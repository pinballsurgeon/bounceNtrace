import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackRow:
    frame: int
    t_s: float
    ok: bool
    cx_px: Optional[float]
    cy_px: Optional[float]
    bbox_x_px: Optional[float]
    bbox_y_px: Optional[float]
    bbox_w_px: Optional[float]
    bbox_h_px: Optional[float]
    vx_px_s: Optional[float] = None
    vy_px_s: Optional[float] = None
    speed_px_s: Optional[float] = None
    angle_deg: Optional[float] = None


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
    raise ValueError(
        f"Unsupported tracker '{tracker_name}'. Try 'csrt' (recommended) or 'kcf'."
    )


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


def _finite_diff_velocity(
    rows: Iterable[TrackRow], fps: float
) -> Tuple[TrackRow, ...]:
    dt = 1.0 / fps if fps > 0 else 0.0
    out = []
    prev: Optional[TrackRow] = None
    for row in rows:
        if (
            prev is None
            or not row.ok
            or not prev.ok
            or row.cx_px is None
            or row.cy_px is None
            or prev.cx_px is None
            or prev.cy_px is None
            or dt <= 0
        ):
            out.append(row)
            prev = row
            continue
        vx = (row.cx_px - prev.cx_px) / dt
        vy = (row.cy_px - prev.cy_px) / dt
        speed = math.hypot(vx, vy)
        angle = math.degrees(math.atan2(-vy, vx))  # y up, 0° = +x (right), 90° = up
        out.append(
            TrackRow(
                **{
                    **row.__dict__,
                    "vx_px_s": vx,
                    "vy_px_s": vy,
                    "speed_px_s": speed,
                    "angle_deg": angle,
                }
            )
        )
        prev = row
    return tuple(out)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{v:.6f}"


def _write_csv(rows: Iterable[TrackRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "t_s",
        "ok",
        "cx_px",
        "cy_px",
        "bbox_x_px",
        "bbox_y_px",
        "bbox_w_px",
        "bbox_h_px",
        "vx_px_s",
        "vy_px_s",
        "speed_px_s",
        "angle_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "frame": r.frame,
                    "t_s": _fmt(r.t_s),
                    "ok": int(bool(r.ok)),
                    "cx_px": _fmt(r.cx_px),
                    "cy_px": _fmt(r.cy_px),
                    "bbox_x_px": _fmt(r.bbox_x_px),
                    "bbox_y_px": _fmt(r.bbox_y_px),
                    "bbox_w_px": _fmt(r.bbox_w_px),
                    "bbox_h_px": _fmt(r.bbox_h_px),
                    "vx_px_s": _fmt(r.vx_px_s),
                    "vy_px_s": _fmt(r.vy_px_s),
                    "speed_px_s": _fmt(r.speed_px_s),
                    "angle_deg": _fmt(r.angle_deg),
                }
            )


def _maybe_make_plots(rows: Tuple[TrackRow, ...], plot_path: Optional[Path]) -> None:
    if plot_path is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for --plots. Install it with: pip install matplotlib"
        ) from e

    t = np.array([r.t_s for r in rows], dtype=float)
    x = np.array([np.nan if r.cx_px is None else r.cx_px for r in rows], dtype=float)
    y = np.array([np.nan if r.cy_px is None else r.cy_px for r in rows], dtype=float)
    speed = np.array(
        [np.nan if r.speed_px_s is None else r.speed_px_s for r in rows], dtype=float
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, x, lw=1)
    axes[0].set_ylabel("x (px)")
    axes[1].plot(t, y, lw=1)
    axes[1].set_ylabel("y (px)")
    axes[2].plot(t, speed, lw=1)
    axes[2].set_ylabel("speed (px/s)")
    axes[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _draw_overlay(
    frame: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]],
    ok: bool,
    trail: Deque[Tuple[int, int]],
    speed: Optional[float],
    angle_deg: Optional[float],
    downscale: float,
):
    out = frame
    if bbox is not None and ok:
        x, y, w, h = bbox
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cx = int(round(x + w / 2))
        cy = int(round(y + h / 2))
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        trail.append((cx, cy))
    else:
        trail.append(trail[-1] if trail else (0, 0))

    for i in range(1, len(trail)):
        if trail[i - 1] == (0, 0) or trail[i] == (0, 0):
            continue
        cv2.line(out, trail[i - 1], trail[i], (255, 0, 0), 2)

    text = f"{'OK' if ok else 'LOST'}"
    if speed is not None:
        text += f" | {speed:.1f}px/s"
    if angle_deg is not None:
        text += f" | {angle_deg:.1f}deg"
    cv2.putText(
        out,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if ok else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    if downscale != 1.0:
        cv2.putText(
            out,
            f"downscale={downscale:.2f} (CSV is original px)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Track a ball in a video and export position/velocity/angle to CSV."
    )
    ap.add_argument("--video", required=True, help="Path to input video (e.g. IMG_6775.MOV)")
    ap.add_argument(
        "--output",
        default="ball_track.csv",
        help="CSV output path (default: ball_track.csv)",
    )
    ap.add_argument(
        "--annot",
        default="ball_track_annot.mp4",
        help="Annotated video output path (default: ball_track_annot.mp4). Use '' to disable.",
    )
    ap.add_argument(
        "--plots",
        default="ball_track_plots.png",
        help="Plot output path (default: ball_track_plots.png). Use '' to disable.",
    )
    ap.add_argument("--tracker", default="csrt", help="Tracker: csrt or kcf (default: csrt)")
    ap.add_argument("--start-frame", type=int, default=0, help="Start frame index (default: 0)")
    ap.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="End frame index inclusive (default: -1 for end of video)",
    )
    ap.add_argument(
        "--init-bbox",
        default="",
        help="Initial bbox 'x,y,w,h' in start frame. If omitted, an interactive selector opens.",
    )
    ap.add_argument(
        "--init-pad",
        type=float,
        default=10.0,
        help="Pad (px) added around initial bbox (default: 10)",
    )
    ap.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Downscale factor for processing/annot video (e.g. 0.5). CSV stays in original px.",
    )
    ap.add_argument(
        "--select-max-width",
        type=int,
        default=1000,
        help="Max width (px) for the ROI selection preview window; set 0 to disable (default: 1000)",
    )
    ap.add_argument(
        "--select-max-height",
        type=int,
        default=900,
        help="Max height (px) for the ROI selection preview window; set 0 to disable (default: 900)",
    )
    ap.add_argument(
        "--trail",
        type=int,
        default=30,
        help="Number of previous points to draw (default: 30)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show live tracking window (press 'q' to quit early)",
    )
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    downscale = float(args.downscale)
    if not (0.05 <= downscale <= 1.0):
        raise ValueError("--downscale must be in [0.05, 1.0]")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 30.0

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

    frame0_p = _downscale_frame(frame0, downscale)
    fh, fw = frame0_p.shape[:2]

    if args.init_bbox.strip():
        bbox0 = _parse_bbox(args.init_bbox.strip())
        bbox0 = (bbox0[0] * downscale, bbox0[1] * downscale, bbox0[2] * downscale, bbox0[3] * downscale)
    else:
        max_w = float(args.select_max_width) if args.select_max_width and args.select_max_width > 0 else float("inf")
        max_h = float(args.select_max_height) if args.select_max_height and args.select_max_height > 0 else float("inf")
        select_scale = min(1.0, max_w / fw, max_h / fh)

        selector = frame0_p.copy()
        if select_scale != 1.0:
            sel_w = max(1, int(round(fw * select_scale)))
            sel_h = max(1, int(round(fh * select_scale)))
            selector = cv2.resize(selector, (sel_w, sel_h), interpolation=cv2.INTER_AREA)

        cv2.putText(
            selector,
            "Select the BALL and press ENTER (or ESC to cancel)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        win = "ball_track: select ROI"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, selector.shape[1], selector.shape[0])
        bbox0 = cv2.selectROI(win, selector, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)
        bbox0 = tuple(float(v) / select_scale for v in bbox0)
        if bbox0[2] <= 0 or bbox0[3] <= 0:
            print("Canceled ROI selection.")
            return 2

    bbox0 = _bbox_pad_xywh(bbox0, float(args.init_pad), fw, fh)

    tracker = _create_tracker(args.tracker)
    # Some OpenCV builds on Windows are picky about bbox types at init-time and
    # only accept integer coordinates.
    bbox0_i = tuple(int(round(v)) for v in bbox0)
    bbox0_i = (
        max(0, bbox0_i[0]),
        max(0, bbox0_i[1]),
        max(1, bbox0_i[2]),
        max(1, bbox0_i[3]),
    )
    tracker.init(frame0_p, bbox0_i)

    annot_path = Path(args.annot) if args.annot.strip() else None
    plot_path = Path(args.plots) if args.plots.strip() else None
    out_csv = Path(args.output)

    writer = None
    if annot_path is not None:
        annot_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annot_path), fourcc, fps, (fw, fh))
        if not writer.isOpened():
            writer.release()
            annot_path = annot_path.with_suffix(".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(annot_path), fourcc, fps, (fw, fh))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for annotated output.")

    trail: Deque[Tuple[int, int]] = deque(maxlen=max(1, int(args.trail)))
    rows: list[TrackRow] = []

    prev_center: Optional[Tuple[float, float]] = None
    dt = 1.0 / fps

    def row_from_bbox(frame_idx: int, bbox: Tuple[float, float, float, float], ok: bool) -> TrackRow:
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        cx_o = cx / downscale
        cy_o = cy / downscale
        return TrackRow(
            frame=frame_idx,
            t_s=(frame_idx / fps),
            ok=ok,
            cx_px=cx_o if ok else None,
            cy_px=cy_o if ok else None,
            bbox_x_px=(x / downscale) if ok else None,
            bbox_y_px=(y / downscale) if ok else None,
            bbox_w_px=(w / downscale) if ok else None,
            bbox_h_px=(h / downscale) if ok else None,
        )

    # frame0 already read
    frame_idx = start_frame
    ok0, bbox0_u = tracker.update(frame0_p)
    bbox0_u = tuple(float(v) for v in bbox0_u)
    rows.append(row_from_bbox(frame_idx, bbox0_u, bool(ok0)))
    if ok0:
        prev_center = (rows[-1].cx_px or 0.0, rows[-1].cy_px or 0.0)

    if writer is not None or args.show:
        speed0 = None
        angle0 = None
        _draw_overlay(frame0_p, bbox0_u, bool(ok0), trail, speed0, angle0, downscale)
        if writer is not None:
            writer.write(frame0_p)
        if args.show:
            cv2.imshow("ball_track", frame0_p)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
                rows2 = _finite_diff_velocity(rows, fps)
                _write_csv(rows2, out_csv)
                _maybe_make_plots(rows2, plot_path)
                return 0

    for frame_idx in range(start_frame + 1, end_frame + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_p = _downscale_frame(frame, downscale)
        ok, bbox = tracker.update(frame_p)
        bbox = tuple(float(v) for v in bbox)
        row = row_from_bbox(frame_idx, bbox, bool(ok))

        speed = None
        angle = None
        if ok and row.cx_px is not None and row.cy_px is not None and prev_center is not None and dt > 0:
            vx = (row.cx_px - prev_center[0]) / dt
            vy = (row.cy_px - prev_center[1]) / dt
            speed = math.hypot(vx, vy)
            angle = math.degrees(math.atan2(-vy, vx))
            prev_center = (row.cx_px, row.cy_px)
        elif ok and row.cx_px is not None and row.cy_px is not None:
            prev_center = (row.cx_px, row.cy_px)

        rows.append(row)

        if writer is not None or args.show:
            _draw_overlay(frame_p, bbox, bool(ok), trail, speed, angle, downscale)
            if writer is not None:
                writer.write(frame_p)
            if args.show:
                cv2.imshow("ball_track", frame_p)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    rows2 = _finite_diff_velocity(rows, fps)
    _write_csv(rows2, out_csv)
    _maybe_make_plots(rows2, plot_path)

    if annot_path is not None:
        print(f"Wrote annotated video: {annot_path}")
    print(f"Wrote CSV: {out_csv}")
    if plot_path is not None:
        print(f"Wrote plots: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
