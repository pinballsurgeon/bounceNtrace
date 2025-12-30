import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Side = str  # "L" or "R"


@dataclass
class SideStats:
    hits: int = 0
    arcs: int = 0
    distance_px: float = 0.0
    best_hit_speed_px_s: float = 0.0
    best_hit_delta_v_px_s: float = 0.0
    best_arc_height_px: float = 0.0  # max height (y-up, in px)
    best_arc_distance_px: float = 0.0


@dataclass(frozen=True)
class HitEvent:
    frame: int
    t_s: float
    side: Side
    cx_px: float
    cy_px: float
    speed_px_s: float
    delta_v_px_s: float


@dataclass(frozen=True)
class ArcEvent:
    arc_id: int
    side: Side
    start_frame: int
    end_frame: int
    start_t_s: float
    end_t_s: float
    distance_px: float
    max_speed_px_s: float
    max_height_px: float
    start_cx_px: float
    start_cy_px: float
    end_cx_px: float
    end_cy_px: float


@dataclass(frozen=True)
class FrameRow:
    frame: int
    t_s: float
    ok: bool
    cx_px: Optional[float]
    cy_px: Optional[float]
    vx_px_s: Optional[float]
    vy_px_s: Optional[float]
    speed_px_s: Optional[float]
    angle_deg: Optional[float]
    side: Optional[Side]
    is_hit: int
    hit_side: Optional[Side]
    total_distance_px: Optional[float]
    left_distance_px: Optional[float]
    right_distance_px: Optional[float]


def _create_tracker(tracker_name: str):
    name = tracker_name.strip().lower()
    if name in {"csrt", "tracker_csrt"}:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
    if name in {"kcf", "tracker_kcf"}:
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
    raise ValueError(f"Unsupported tracker '{tracker_name}'. Use 'csrt' or 'kcf'.")


def _parse_bbox(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected bbox format 'x,y,w,h'")
    x, y, w, h = (float(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("bbox w/h must be > 0")
    return x, y, w, h


def _downscale_frame(frame: np.ndarray, downscale: float) -> np.ndarray:
    if downscale == 1.0:
        return frame
    h, w = frame.shape[:2]
    out_w = max(1, int(round(w * downscale)))
    out_h = max(1, int(round(h * downscale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _bbox_pad_xywh(
    bbox: Tuple[float, float, float, float], pad_px: float, frame_w: int, frame_h: int
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    x = max(0.0, x - pad_px)
    y = max(0.0, y - pad_px)
    x2 = min(float(frame_w - 1), x2 + pad_px)
    y2 = min(float(frame_h - 1), y2 + pad_px)
    w = max(1.0, x2 - x)
    h = max(1.0, y2 - y)
    return x, y, w, h


def _safe_fourcc(name: str) -> int:
    return cv2.VideoWriter_fourcc(*name)


def _open_writer(path: Path, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    # Prefer MP4, fallback to AVI if needed.
    for suffix, fourcc_name in [(".mp4", "mp4v"), (".avi", "XVID")]:
        out_path = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(out_path), _safe_fourcc(fourcc_name), fps, (w, h))
        if writer.isOpened():
            return writer, out_path
        writer.release()
    raise RuntimeError("Failed to open VideoWriter (mp4v and XVID both failed).")


def _overlay_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color_bgr, alpha: float) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img.shape[1], x + w)
    y1 = min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def _draw_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color_bgr,
    scale: float = 0.7,
    thickness: int = 2,
):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def _draw_polyline(
    img: np.ndarray,
    pts: Iterable[Tuple[int, int]],
    color_bgr,
    thickness: int = 2,
) -> None:
    pts_arr = np.array(list(pts), dtype=np.int32)
    if len(pts_arr) < 2:
        return
    cv2.polylines(img, [pts_arr.reshape(-1, 1, 2)], isClosed=False, color=color_bgr, thickness=thickness)


def _plot_series(
    canvas: np.ndarray,
    values: List[float],
    color_bgr,
    label: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    h, w = canvas.shape[:2]
    pad_l, pad_r, pad_t, pad_b = 44, 10, 18, 22
    x0, x1 = pad_l, w - pad_r
    y0, y1 = pad_t, h - pad_b
    if x1 <= x0 or y1 <= y0 or len(values) < 2:
        return

    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    if vmin is None:
        vmin = float(np.nanmin(vals))
    if vmax is None:
        vmax = float(np.nanmax(vals))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    # add margin
    margin = 0.08 * (vmax - vmin)
    vmin -= margin
    vmax += margin

    # axes
    cv2.line(canvas, (x0, y1), (x1, y1), (160, 160, 160), 1)
    cv2.line(canvas, (x0, y0), (x0, y1), (160, 160, 160), 1)
    _draw_text(canvas, label, (8, 16), (255, 255, 255), scale=0.55, thickness=1)
    _draw_text(canvas, f"{vmax:.0f}", (4, y0 + 12), (180, 180, 180), scale=0.45, thickness=1)
    _draw_text(canvas, f"{vmin:.0f}", (4, y1), (180, 180, 180), scale=0.45, thickness=1)

    pts: List[Tuple[int, int]] = []
    n = len(values)
    for i, v in enumerate(values):
        if not math.isfinite(v):
            continue
        x = x0 + int(round((i / (n - 1)) * (x1 - x0)))
        yn = (v - vmin) / (vmax - vmin)
        y = y1 - int(round(yn * (y1 - y0)))
        pts.append((x, y))
    _draw_polyline(canvas, pts, color_bgr, thickness=2)


def _as_mid_x_px(mid_x: float, frame_w_px: int) -> float:
    # If mid_x <= 1 treat as fraction.
    if mid_x <= 1.0:
        return float(frame_w_px) * float(mid_x)
    return float(mid_x)


def _select_roi_scaled(
    frame: np.ndarray, max_w: int, max_h: int, win_name: str
) -> Tuple[Tuple[float, float, float, float], float]:
    h, w = frame.shape[:2]
    max_w_f = float(max_w) if max_w and max_w > 0 else float("inf")
    max_h_f = float(max_h) if max_h and max_h > 0 else float("inf")
    scale = min(1.0, max_w_f / w, max_h_f / h)
    view = frame
    if scale != 1.0:
        view_w = max(1, int(round(w * scale)))
        view_h = max(1, int(round(h * scale)))
        view = cv2.resize(frame, (view_w, view_h), interpolation=cv2.INTER_AREA)

    view2 = view.copy()
    _draw_text(view2, "Select BALL ROI, press ENTER (ESC cancels)", (10, 30), (255, 255, 255), scale=0.7, thickness=2)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, view2.shape[1], view2.shape[0])
    bbox_view = cv2.selectROI(win_name, view2, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win_name)
    bbox_view = tuple(float(v) for v in bbox_view)
    bbox = tuple(float(v) / scale for v in bbox_view)
    return bbox, scale


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{v:.6f}"


def _write_rows_csv(rows: Iterable[FrameRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame",
        "t_s",
        "ok",
        "cx_px",
        "cy_px",
        "vx_px_s",
        "vy_px_s",
        "speed_px_s",
        "angle_deg",
        "side",
        "is_hit",
        "hit_side",
        "total_distance_px",
        "left_distance_px",
        "right_distance_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "frame": r.frame,
                    "t_s": _fmt(r.t_s),
                    "ok": int(bool(r.ok)),
                    "cx_px": _fmt(r.cx_px),
                    "cy_px": _fmt(r.cy_px),
                    "vx_px_s": _fmt(r.vx_px_s),
                    "vy_px_s": _fmt(r.vy_px_s),
                    "speed_px_s": _fmt(r.speed_px_s),
                    "angle_deg": _fmt(r.angle_deg),
                    "side": r.side or "",
                    "is_hit": int(r.is_hit),
                    "hit_side": r.hit_side or "",
                    "total_distance_px": _fmt(r.total_distance_px),
                    "left_distance_px": _fmt(r.left_distance_px),
                    "right_distance_px": _fmt(r.right_distance_px),
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ball tracker + 2-player scoreboard (hits/arcs), with live graphs rendered into the video."
    )
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument(
        "--output-video",
        default="ball_scoreboard_annot.mp4",
        help="Annotated output video path (default: ball_scoreboard_annot.mp4)",
    )
    ap.add_argument(
        "--output-frames",
        default="ball_scoreboard_frames.csv",
        help="Per-frame CSV output path; set '' to disable (default: ball_scoreboard_frames.csv)",
    )
    ap.add_argument(
        "--output-hits",
        default="ball_scoreboard_hits.csv",
        help="Hit-event CSV output path; set '' to disable (default: ball_scoreboard_hits.csv)",
    )
    ap.add_argument(
        "--output-arcs",
        default="ball_scoreboard_arcs.csv",
        help="Arc CSV output path; set '' to disable (default: ball_scoreboard_arcs.csv)",
    )
    ap.add_argument("--tracker", default="csrt", help="Tracker: csrt or kcf (default: csrt)")
    ap.add_argument("--start-frame", type=int, default=0, help="Start frame index (default: 0)")
    ap.add_argument("--end-frame", type=int, default=-1, help="End frame index inclusive (default: -1 = end)")
    ap.add_argument(
        "--init-bbox",
        default="",
        help="Initial bbox 'x,y,w,h' in start frame. If omitted, an interactive selector opens.",
    )
    ap.add_argument("--init-pad", type=float, default=10.0, help="Pad around ROI in px (default: 10)")
    ap.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Downscale factor for processing/output (e.g. 0.5). CSV remains in original px.",
    )
    ap.add_argument(
        "--select-max-width",
        type=int,
        default=1000,
        help="Max width (px) for ROI selection preview window; set 0 to disable (default: 1000)",
    )
    ap.add_argument(
        "--select-max-height",
        type=int,
        default=900,
        help="Max height (px) for ROI selection preview window; set 0 to disable (default: 900)",
    )
    ap.add_argument(
        "--mid-x",
        type=float,
        default=0.5,
        help="Midline X split for Left/Right. <=1 means fraction of width (default: 0.5).",
    )
    ap.add_argument("--trail", type=int, default=30, help="Trail length (default: 30)")
    ap.add_argument(
        "--dashboard-frac",
        type=float,
        default=0.25,
        help="Fraction of frame height used for bottom graphs overlay (default: 0.25)",
    )
    ap.add_argument(
        "--plot-window-s",
        type=float,
        default=10.0,
        help="Seconds of history to show in graphs (default: 10)",
    )
    ap.add_argument("--smooth", type=int, default=5, help="Velocity smoothing window (frames, default: 5)")
    ap.add_argument(
        "--hit-vy-down",
        type=float,
        default=150.0,
        help="Hit detection: previous vy must be > this (px/s) (default: 150)",
    )
    ap.add_argument(
        "--hit-vy-up",
        type=float,
        default=150.0,
        help="Hit detection: current vy must be < -this (px/s) (default: 150)",
    )
    ap.add_argument(
        "--hit-speed-min",
        type=float,
        default=250.0,
        help="Hit detection: current speed must be >= this (px/s) (default: 250)",
    )
    ap.add_argument(
        "--hit-cooldown-frames",
        type=int,
        default=6,
        help="Min frames between hit events (default: 6)",
    )
    ap.add_argument("--show", action="store_true", help="Show live preview (press 'q' to stop)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    downscale = float(args.downscale)
    if not (0.05 <= downscale <= 1.0):
        raise ValueError("--downscale must be in [0.05, 1.0]")
    dashboard_frac = float(args.dashboard_frac)
    if not (0.0 <= dashboard_frac <= 0.45):
        raise ValueError("--dashboard-frac must be in [0, 0.45]")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(args.start_frame))
    end_frame = int(args.end_frame)
    if end_frame < 0 or end_frame >= total_frames:
        end_frame = max(0, total_frames - 1)
    if start_frame > end_frame:
        raise ValueError("--start-frame must be <= --end-frame")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        raise RuntimeError(f"Failed to read frame {start_frame}")

    h0, w0 = frame0.shape[:2]
    mid_x_px = _as_mid_x_px(float(args.mid_x), w0)
    if not (0 <= mid_x_px <= w0):
        raise ValueError("--mid-x is out of bounds for this video width")

    frame0_p = _downscale_frame(frame0, downscale)
    ph, pw = frame0_p.shape[:2]

    if args.init_bbox.strip():
        bbox0 = _parse_bbox(args.init_bbox.strip())
        bbox0 = (bbox0[0] * downscale, bbox0[1] * downscale, bbox0[2] * downscale, bbox0[3] * downscale)
    else:
        bbox0, _ = _select_roi_scaled(
            frame0_p, int(args.select_max_width), int(args.select_max_height), "ball_scoreboard: select ROI"
        )
    if bbox0[2] <= 0 or bbox0[3] <= 0:
        print("Canceled ROI selection.")
        cap.release()
        return 2

    bbox0 = _bbox_pad_xywh(bbox0, float(args.init_pad), pw, ph)
    bbox0_i = tuple(int(round(v)) for v in bbox0)
    bbox0_i = (
        max(0, bbox0_i[0]),
        max(0, bbox0_i[1]),
        max(1, bbox0_i[2]),
        max(1, bbox0_i[3]),
    )

    tracker = _create_tracker(args.tracker)
    tracker.init(frame0_p, bbox0_i)

    out_video_path = Path(args.output_video)
    writer, final_video_path = _open_writer(out_video_path, fps, (pw, ph))

    # Colors (BGR)
    col_left = (255, 180, 60)
    col_right = (60, 200, 255)
    col_trail = (255, 0, 0)
    col_mid = (120, 120, 120)

    stats: Dict[Side, SideStats] = {"L": SideStats(), "R": SideStats()}
    hit_events: List[HitEvent] = []
    arc_events: List[ArcEvent] = []
    frame_rows: List[FrameRow] = []

    prev_center_px: Optional[Tuple[float, float]] = None  # original px
    prev_vx: Optional[float] = None
    prev_vy: Optional[float] = None
    vel_hist: Deque[Tuple[float, float]] = deque(maxlen=max(1, int(args.smooth)))
    last_hit_frame = -10_000

    current_arc_side: Optional[Side] = None
    arc_id = 0
    arc_start_frame: Optional[int] = None
    arc_start_pos: Optional[Tuple[float, float]] = None
    arc_distance_px = 0.0
    arc_max_speed = 0.0
    arc_max_height = 0.0

    total_distance_px = 0.0

    max_hist = max(10, int(round(float(args.plot_window_s) * fps)))
    hist_speed: Deque[float] = deque(maxlen=max_hist)
    hist_height: Deque[float] = deque(maxlen=max_hist)
    hist_dist: Deque[float] = deque(maxlen=max_hist)

    trail: Deque[Tuple[int, int]] = deque(maxlen=max(1, int(args.trail)))

    def side_for_x(cx_px: float) -> Side:
        return "L" if cx_px < mid_x_px else "R"

    def height_y_up(cy_px: float) -> float:
        return float(h0) - float(cy_px)

    def close_arc(end_frame: int, end_pos: Tuple[float, float]) -> None:
        nonlocal arc_id, arc_start_frame, arc_start_pos, arc_distance_px, arc_max_speed, arc_max_height, current_arc_side
        if current_arc_side is None or arc_start_frame is None or arc_start_pos is None:
            return
        if end_frame <= arc_start_frame:
            return
        arc_id += 1
        e = ArcEvent(
            arc_id=arc_id,
            side=current_arc_side,
            start_frame=arc_start_frame,
            end_frame=end_frame,
            start_t_s=arc_start_frame / fps,
            end_t_s=end_frame / fps,
            distance_px=arc_distance_px,
            max_speed_px_s=arc_max_speed,
            max_height_px=arc_max_height,
            start_cx_px=arc_start_pos[0],
            start_cy_px=arc_start_pos[1],
            end_cx_px=end_pos[0],
            end_cy_px=end_pos[1],
        )
        arc_events.append(e)
        side_stats = stats[current_arc_side]
        side_stats.arcs += 1
        side_stats.distance_px += arc_distance_px
        side_stats.best_arc_height_px = max(side_stats.best_arc_height_px, arc_max_height)
        side_stats.best_arc_distance_px = max(side_stats.best_arc_distance_px, arc_distance_px)

        arc_start_frame = None
        arc_start_pos = None
        arc_distance_px = 0.0
        arc_max_speed = 0.0
        arc_max_height = 0.0
        current_arc_side = None

    def start_arc(side: Side, frame_idx: int, pos: Tuple[float, float], speed: float, height: float) -> None:
        nonlocal current_arc_side, arc_start_frame, arc_start_pos, arc_distance_px, arc_max_speed, arc_max_height
        current_arc_side = side
        arc_start_frame = frame_idx
        arc_start_pos = pos
        arc_distance_px = 0.0
        arc_max_speed = float(speed)
        arc_max_height = float(height)

    def draw_overlay(
        frame_p: np.ndarray,
        frame_idx: int,
        ok: bool,
        bbox_p: Optional[Tuple[float, float, float, float]],
        center_p: Optional[Tuple[float, float]],
        vx: Optional[float],
        vy: Optional[float],
        speed: Optional[float],
        angle: Optional[float],
        is_hit: bool,
        hit_side: Optional[Side],
    ) -> None:
        mid_p = int(round(mid_x_px * downscale))
        cv2.line(frame_p, (mid_p, 0), (mid_p, ph - 1), col_mid, 2)

        if ok and bbox_p is not None:
            x, y, w, h = bbox_p
            cv2.rectangle(
                frame_p,
                (int(round(x)), int(round(y))),
                (int(round(x + w)), int(round(y + h))),
                (0, 255, 0),
                2,
            )
        if ok and center_p is not None:
            cx, cy = int(round(center_p[0])), int(round(center_p[1]))
            cv2.circle(frame_p, (cx, cy), 3, (0, 0, 255), -1)
            trail.append((cx, cy))
        else:
            trail.append((0, 0))

        for i in range(1, len(trail)):
            if trail[i - 1] == (0, 0) or trail[i] == (0, 0):
                continue
            cv2.line(frame_p, trail[i - 1], trail[i], col_trail, 2)

        if ok and center_p is not None and vx is not None and vy is not None and speed is not None and speed > 1:
            cx, cy = center_p
            px_per_frame = (speed / fps) * downscale
            vec_scale = 60.0 / max(20.0, px_per_frame)
            dx = (vx / fps) * downscale * vec_scale
            dy = (vy / fps) * downscale * vec_scale
            p1 = (int(round(cx)), int(round(cy)))
            p2 = (int(round(cx + dx)), int(round(cy + dy)))
            cv2.arrowedLine(frame_p, p1, p2, (0, 255, 255), 2, tipLength=0.25)

        top_h = max(72, int(round(ph * 0.09)))
        _overlay_rect(frame_p, 0, 0, pw, top_h, (0, 0, 0), alpha=0.55)

        left_s = stats["L"]
        right_s = stats["R"]
        left_live_dist = left_s.distance_px + (arc_distance_px if current_arc_side == "L" else 0.0)
        right_live_dist = right_s.distance_px + (arc_distance_px if current_arc_side == "R" else 0.0)

        _draw_text(frame_p, f"LEFT  hits {left_s.hits}", (10, 28), col_left, scale=0.8, thickness=2)
        _draw_text(
            frame_p,
            f"dist {left_live_dist:,.0f}px  best {left_s.best_hit_speed_px_s:,.0f}px/s",
            (10, 56),
            (230, 230, 230),
            scale=0.65,
            thickness=2,
        )
        right_x = max(10, pw - 420)
        _draw_text(frame_p, f"RIGHT hits {right_s.hits}", (right_x + 100, 28), col_right, scale=0.8, thickness=2)
        _draw_text(
            frame_p,
            f"dist {right_live_dist:,.0f}px  best {right_s.best_hit_speed_px_s:,.0f}px/s",
            (right_x, 56),
            (230, 230, 230),
            scale=0.65,
            thickness=2,
        )
        _draw_text(
            frame_p,
            f"total dist {total_distance_px:,.0f}px",
            (max(10, pw // 2 - 140), 28),
            (255, 255, 255),
            scale=0.7,
            thickness=2,
        )
        _draw_text(
            frame_p,
            f"t {frame_idx / fps:,.2f}s",
            (max(10, pw // 2 - 70), 56),
            (200, 200, 200),
            scale=0.65,
            thickness=2,
        )

        status = "OK" if ok else "LOST"
        if is_hit and hit_side:
            status += f" | HIT {hit_side}"
        if speed is not None and angle is not None:
            status += f" | {speed:,.0f}px/s {angle:+.0f}deg"
        _draw_text(
            frame_p,
            status,
            (10, top_h - 10),
            (0, 255, 0) if ok else (0, 0, 255),
            scale=0.65,
            thickness=2,
        )

        if dashboard_frac > 0:
            dash_h = int(round(ph * dashboard_frac))
            y0 = ph - dash_h
            _overlay_rect(frame_p, 0, y0, pw, dash_h, (0, 0, 0), alpha=0.60)
            slot_h = max(1, dash_h // 3)
            for i in range(3):
                yy = y0 + i * slot_h
                cv2.line(frame_p, (0, yy), (pw - 1, yy), (50, 50, 50), 1)

            def plot_into(slot: int, series: Deque[float], color, label: str):
                yy = y0 + slot * slot_h
                region = frame_p[yy : yy + slot_h, 0:pw]
                _plot_series(region, list(series), color, label)

            plot_into(0, hist_speed, (0, 255, 255), "speed (px/s)")
            plot_into(1, hist_height, (255, 180, 180), "height (px, y-up)")
            plot_into(2, hist_dist, (180, 255, 180), "total distance (px)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        if frame_idx == start_frame:
            frame = frame0
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        frame_p = _downscale_frame(frame, downscale)
        ok, bbox_p = tracker.update(frame_p)
        ok = bool(ok)
        bbox_p_t = tuple(float(v) for v in bbox_p) if bbox_p is not None else None

        cx_px: Optional[float] = None
        cy_px: Optional[float] = None
        center_p: Optional[Tuple[float, float]] = None
        if ok and bbox_p_t is not None:
            x, y, w, h = bbox_p_t
            cx_p = x + w / 2
            cy_p = y + h / 2
            center_p = (cx_p, cy_p)
            cx_px = cx_p / downscale
            cy_px = cy_p / downscale

        vx = vy = speed = angle = None
        if ok and cx_px is not None and cy_px is not None and prev_center_px is not None:
            dt = 1.0 / fps
            vx = (cx_px - prev_center_px[0]) / dt
            vy = (cy_px - prev_center_px[1]) / dt
            vel_hist.append((vx, vy))
        elif ok and cx_px is not None and cy_px is not None:
            vel_hist.clear()
            vel_hist.append((0.0, 0.0))

        if ok and len(vel_hist) > 0:
            vx = float(np.median([v[0] for v in vel_hist]))
            vy = float(np.median([v[1] for v in vel_hist]))
            speed = math.hypot(vx, vy)
            angle = math.degrees(math.atan2(-vy, vx))

        if ok and cx_px is not None and cy_px is not None and prev_center_px is not None:
            step_dist = math.hypot(cx_px - prev_center_px[0], cy_px - prev_center_px[1])
            total_distance_px += step_dist
            if current_arc_side is not None:
                arc_distance_px += step_dist

        if ok and speed is not None and current_arc_side is not None:
            arc_max_speed = max(arc_max_speed, float(speed))
        if ok and cy_px is not None and current_arc_side is not None:
            arc_max_height = max(arc_max_height, height_y_up(cy_px))

        is_hit = False
        hit_side: Optional[Side] = None
        delta_v = 0.0
        if (
            ok
            and vx is not None
            and vy is not None
            and speed is not None
            and prev_vy is not None
            and (frame_idx - last_hit_frame) >= int(args.hit_cooldown_frames)
            and prev_vy > float(args.hit_vy_down)
            and vy < -float(args.hit_vy_up)
            and speed >= float(args.hit_speed_min)
            and cx_px is not None
            and cy_px is not None
        ):
            is_hit = True
            hit_side = side_for_x(cx_px)
            if prev_vx is not None and prev_vy is not None:
                delta_v = math.hypot(vx - prev_vx, vy - prev_vy)
            last_hit_frame = frame_idx

            close_arc(frame_idx, (cx_px, cy_px))

            stats[hit_side].hits += 1
            stats[hit_side].best_hit_speed_px_s = max(stats[hit_side].best_hit_speed_px_s, float(speed))
            stats[hit_side].best_hit_delta_v_px_s = max(stats[hit_side].best_hit_delta_v_px_s, float(delta_v))

            start_arc(hit_side, frame_idx, (cx_px, cy_px), float(speed), height_y_up(cy_px))
            hit_events.append(
                HitEvent(
                    frame=frame_idx,
                    t_s=frame_idx / fps,
                    side=hit_side,
                    cx_px=float(cx_px),
                    cy_px=float(cy_px),
                    speed_px_s=float(speed),
                    delta_v_px_s=float(delta_v),
                )
            )

        if ok and speed is not None and cy_px is not None:
            hist_speed.append(float(speed))
            hist_height.append(height_y_up(cy_px))
            hist_dist.append(float(total_distance_px))
        else:
            hist_speed.append(float("nan"))
            hist_height.append(float("nan"))
            hist_dist.append(float(total_distance_px))

        draw_overlay(frame_p, frame_idx, ok, bbox_p_t, center_p, vx, vy, speed, angle, is_hit, hit_side)
        writer.write(frame_p)

        if args.show:
            cv2.imshow("ball_scoreboard", frame_p)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        left_live_dist = stats["L"].distance_px + (arc_distance_px if current_arc_side == "L" else 0.0)
        right_live_dist = stats["R"].distance_px + (arc_distance_px if current_arc_side == "R" else 0.0)
        frame_rows.append(
            FrameRow(
                frame=frame_idx,
                t_s=frame_idx / fps,
                ok=ok,
                cx_px=cx_px,
                cy_px=cy_px,
                vx_px_s=vx,
                vy_px_s=vy,
                speed_px_s=speed,
                angle_deg=angle,
                side=current_arc_side,
                is_hit=int(is_hit),
                hit_side=hit_side,
                total_distance_px=float(total_distance_px),
                left_distance_px=float(left_live_dist),
                right_distance_px=float(right_live_dist),
            )
        )

        if ok and cx_px is not None and cy_px is not None:
            prev_center_px = (cx_px, cy_px)
        prev_vx, prev_vy = vx, vy
        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    last_row = next((r for r in reversed(frame_rows) if r.ok and r.cx_px is not None and r.cy_px is not None), None)
    if last_row is not None:
        close_arc(last_row.frame, (float(last_row.cx_px), float(last_row.cy_px)))

    if args.output_frames.strip():
        _write_rows_csv(frame_rows, Path(args.output_frames))
    if args.output_hits.strip():
        _write_hits_csv(hit_events, Path(args.output_hits))
    if args.output_arcs.strip():
        _write_arcs_csv(arc_events, Path(args.output_arcs))

    print(f"Wrote annotated video: {final_video_path}")
    if args.output_frames.strip():
        print(f"Wrote frames CSV: {args.output_frames}")
    if args.output_hits.strip():
        print(f"Wrote hits CSV: {args.output_hits}")
    if args.output_arcs.strip():
        print(f"Wrote arcs CSV: {args.output_arcs}")
    print(
        f"Summary: L hits={stats['L'].hits} dist={stats['L'].distance_px:,.0f}px | "
        f"R hits={stats['R'].hits} dist={stats['R'].distance_px:,.0f}px"
    )
    return 0


def _write_hits_csv(events: Iterable[HitEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "t_s", "side", "cx_px", "cy_px", "speed_px_s", "delta_v_px_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow(
                {
                    "frame": e.frame,
                    "t_s": _fmt(e.t_s),
                    "side": e.side,
                    "cx_px": _fmt(e.cx_px),
                    "cy_px": _fmt(e.cy_px),
                    "speed_px_s": _fmt(e.speed_px_s),
                    "delta_v_px_s": _fmt(e.delta_v_px_s),
                }
            )


def _write_arcs_csv(events: Iterable[ArcEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arc_id",
        "side",
        "start_frame",
        "end_frame",
        "start_t_s",
        "end_t_s",
        "duration_s",
        "distance_px",
        "max_speed_px_s",
        "max_height_px",
        "start_cx_px",
        "start_cy_px",
        "end_cx_px",
        "end_cy_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow(
                {
                    "arc_id": e.arc_id,
                    "side": e.side,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "start_t_s": _fmt(e.start_t_s),
                    "end_t_s": _fmt(e.end_t_s),
                    "duration_s": _fmt(e.end_t_s - e.start_t_s),
                    "distance_px": _fmt(e.distance_px),
                    "max_speed_px_s": _fmt(e.max_speed_px_s),
                    "max_height_px": _fmt(e.max_height_px),
                    "start_cx_px": _fmt(e.start_cx_px),
                    "start_cy_px": _fmt(e.start_cy_px),
                    "end_cx_px": _fmt(e.end_cx_px),
                    "end_cy_px": _fmt(e.end_cy_px),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
