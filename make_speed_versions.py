import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2


def _parse_speeds(tokens: Iterable[str]) -> List[float]:
    speeds: List[float] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            speeds.append(float(part))
    if not speeds:
        raise ValueError("No speeds provided. Example: 90 70 60")
    return speeds


def _pct_tag(pct: float) -> str:
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    s = f"{pct:g}".replace(".", "p")
    return f"{s}pct"


def _open_writer(path: Path, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size

    # Try MP4 first if requested, then AVI fallback.
    candidates: List[Tuple[Path, str]] = []
    if path.suffix.lower() in {".mp4", ".m4v"}:
        candidates.append((path.with_suffix(".mp4"), "mp4v"))
    candidates.append((path.with_suffix(".avi"), "XVID"))

    for out_path, fourcc_name in candidates:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc_name), fps, (w, h))
        if writer.isOpened():
            return writer, out_path
        writer.release()
    raise RuntimeError("Failed to open VideoWriter (mp4v and XVID both failed).")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create multiple playback-speed versions of ball_broadcast_annot.mp4 by changing output FPS."
    )
    ap.add_argument(
        "--video",
        default="ball_broadcast_annot.mp4",
        help="Input video path (default: ball_broadcast_annot.mp4)",
    )
    ap.add_argument(
        "speeds",
        nargs="+",
        help="Speed percents like: 90 70 60 (also supports comma list: 90,70,60)",
    )
    args = ap.parse_args()

    in_path = Path(args.video)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    speeds = _parse_speeds(args.speeds)

    cap0 = cv2.VideoCapture(str(in_path))
    if not cap0.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")
    in_fps = float(cap0.get(cv2.CAP_PROP_FPS) or 0.0)
    if in_fps <= 0:
        in_fps = 30.0
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    if w <= 0 or h <= 0:
        raise RuntimeError("Failed to read video dimensions.")

    print(f"Input: {in_path} | {w}x{h} | fps={in_fps:.3f} | frames={n if n else 'unknown'}")

    for pct in speeds:
        if pct <= 0:
            raise ValueError(f"Invalid speed percent: {pct}")
        if pct < 5 or pct > 400:
            raise ValueError(f"Speed percent out of supported range (5..400): {pct}")

        factor = pct / 100.0
        out_fps = in_fps * factor
        tag = _pct_tag(pct)
        out_path = in_path.with_name(f"{in_path.stem}_{tag}{in_path.suffix}")

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {in_path}")

        writer, final_out_path = _open_writer(out_path, out_fps, (w, h))
        print(f"Writing: {final_out_path} | speed={factor:.3f}x | out_fps={out_fps:.3f}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            writer.write(frame)

        writer.release()
        cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

