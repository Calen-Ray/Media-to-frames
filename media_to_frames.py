"""Convert local GIF and MP4 media into frame sequences suitable for ESP displays.

Place .gif files in input_gifs/, .mp4 files in input_mp4s/, run the script,
and review the generated folders under output_frames/.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from moviepy.editor import VideoFileClip
from PIL import Image, ImageSequence, UnidentifiedImageError

OUTPUT_SIZE = (128, 128)
COLOR_DEPTH = 256
GIF_FPS = 10
INPUT_GIF_DIR = "input_gifs"
INPUT_MP4_DIR = "input_mp4s"
OUTPUT_DIR = "output_frames"
TEMP_GIF_DIR = "temp_gifs"

INPUT_GIF_PATH = Path(INPUT_GIF_DIR)
INPUT_MP4_PATH = Path(INPUT_MP4_DIR)
OUTPUT_PATH = Path(OUTPUT_DIR)
TEMP_GIF_PATH = Path(TEMP_GIF_DIR)

TEST_GIF_NAME = "GIF_TO_FRAME_TEST_CASE.gif"
TEST_MP4_NAME = "MP4_TO_FRAME_TEST_CASE.mp4"

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS  # Pillow < 9

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = RESAMPLE_FILTER


def parse_size(value: str) -> Tuple[int, int]:
    """Parse --size argument formatted as WIDTHxHEIGHT."""
    try:
        width_str, height_str = value.lower().split("x")
        width, height = int(width_str), int(height_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Size must be formatted as WIDTHxHEIGHT, e.g. 240x240") from None

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive integers")

    return width, height


def ensure_directories() -> None:
    """Create required folders for media processing."""
    for folder in (INPUT_GIF_PATH, INPUT_MP4_PATH, OUTPUT_PATH, TEMP_GIF_PATH):
        folder.mkdir(parents=True, exist_ok=True)


def iter_media(directory: Path, suffix: str) -> Iterable[Path]:
    """Yield media files from the given directory that match the suffix."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower())


def quantize_frame(frame: Image.Image, output_size: Tuple[int, int], color_depth: int) -> Image.Image:
    """Resize and quantize a single frame."""
    processed = frame.copy().convert("RGBA")
    processed = processed.resize(output_size, RESAMPLE_FILTER)
    processed = processed.convert("RGB")
    return processed.quantize(colors=color_depth, method=Image.MEDIANCUT)


def close_clip(clip: object) -> None:
    """Safely close a MoviePy clip if it provides a close() method."""
    close_fn = getattr(clip, "close", None)
    if callable(close_fn):
        close_fn()


def save_frames_from_gif(
    gif_path: Path,
    output_dir: Path,
    output_size: Tuple[int, int],
    color_depth: int,
    output_name: str | None = None,
) -> bool:
    """Extract frames from a GIF file and save them as BMP images."""
    destination_name = output_name or gif_path.stem
    frame_dir = output_dir / destination_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    delays: list[int] = []

    print(f"Processing {gif_path.name}...")

    try:
        with Image.open(gif_path) as img:
            total_frames = getattr(img, "n_frames", 1)
            print(f"  Detected {total_frames} frame(s)")

            for index, frame in enumerate(ImageSequence.Iterator(img), start=1):
                delay_ms = frame.info.get("duration", img.info.get("duration", 0)) or 0
                processed = quantize_frame(frame, output_size, color_depth)

                frame_path = frame_dir / f"frame_{index:04d}.bmp"
                processed.save(frame_path, format="BMP")

                delays.append(int(delay_ms))
                print(f"    Saved frame {index:04d}")

    except (UnidentifiedImageError, OSError) as exc:
        logging.warning("Skipping corrupted GIF '%s': %s", gif_path.name, exc)
        shutil.rmtree(frame_dir, ignore_errors=True)
        return False
    except Exception as exc:
        logging.warning("Unexpected error processing '%s': %s", gif_path.name, exc)
        shutil.rmtree(frame_dir, ignore_errors=True)
        return False

    if not delays:
        logging.warning("No frames extracted from '%s'; removing empty output", gif_path.name)
        shutil.rmtree(frame_dir, ignore_errors=True)
        return False

    frames_txt = frame_dir / "frames.txt"
    frames_txt.write_text("\n".join(str(delay) for delay in delays), encoding="utf-8")
    print(f"  Wrote delays to {frames_txt}")

    return True


def convert_mp4_to_gif(mp4_path: Path, target_size: Tuple[int, int], fps: int) -> Path | None:
    """Convert an MP4 to a temporary GIF stored under TEMP_GIF_PATH."""
    print(f"Converting video: {mp4_path.name}...")
    TEMP_GIF_PATH.mkdir(parents=True, exist_ok=True)
    temp_gif_path = TEMP_GIF_PATH / f"{mp4_path.stem}.gif"

    clip = None
    resized = None
    cropped = None

    try:
        clip = VideoFileClip(str(mp4_path))
        target_w, target_h = target_size
        if clip.w == 0 or clip.h == 0:
            raise ValueError("Clip has invalid dimensions and cannot be resized.")

        scale = max(target_w / clip.w, target_h / clip.h)
        resized = clip.resize(scale)
        cropped = resized.crop(
            x_center=resized.w / 2,
            y_center=resized.h / 2,
            width=target_w,
            height=target_h,
        )

        cropped.write_gif(
            str(temp_gif_path),
            fps=fps,
            program="ffmpeg",
            logger=None,
        )
    except Exception as exc:
        logging.warning("Skipping video '%s': %s", mp4_path.name, exc)
        try:
            temp_gif_path.unlink()
        except FileNotFoundError:
            pass
        return None
    finally:
        if cropped is not None:
            close_clip(cropped)
        if resized is not None:
            close_clip(resized)
        if clip is not None:
            close_clip(clip)

    return temp_gif_path


def process_mp4(
    mp4_path: Path,
    output_dir: Path,
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    cleanup_temp: bool,
) -> bool:
    """Convert an MP4 to GIF and then extract frames using the GIF workflow."""
    temp_gif = convert_mp4_to_gif(mp4_path, output_size, fps)
    if temp_gif is None:
        return False

    try:
        print(f"Extracting frames from {mp4_path.name}...")
        return save_frames_from_gif(temp_gif, output_dir, output_size, color_depth, output_name=mp4_path.stem)
    finally:
        if cleanup_temp:
            try:
                temp_gif.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GIF animations or MP4 videos into individual frames suitable for ESP displays."
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        help="Override the output resolution (WIDTHxHEIGHT). Default is 128x128.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Retain temporary GIF files and test outputs for debugging.",
    )
    return parser


def process_media(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    cleanup_temp: bool,
) -> tuple[int, int, int, int]:
    """Process all media found in the input directories."""
    ensure_directories()

    gifs = list(iter_media(INPUT_GIF_PATH, ".gif"))
    videos = list(iter_media(INPUT_MP4_PATH, ".mp4"))

    if not gifs and not videos:
        print(
            "No media files found. Add .gif files to "
            f"'{INPUT_GIF_DIR}' or .mp4 files to '{INPUT_MP4_DIR}' and rerun."
        )
        return 0, 0, 0, 0

    total_media = len(gifs) + len(videos)
    print(
        f"Found {total_media} media file(s). Output size set to {output_size[0]}x{output_size[1]} pixels "
        f"at {fps} FPS for converted videos."
    )

    converted_gifs = 0
    for gif_path in gifs:
        if save_frames_from_gif(gif_path, OUTPUT_PATH, output_size, color_depth):
            converted_gifs += 1

    converted_videos = 0
    for mp4_path in videos:
        if process_mp4(mp4_path, OUTPUT_PATH, output_size, color_depth, fps, cleanup_temp):
            converted_videos += 1

    print(
        f"Finished. Successfully converted {converted_gifs} of {len(gifs)} GIF(s) "
        f"and {converted_videos} of {len(videos)} video(s)."
    )

    return len(gifs), converted_gifs, len(videos), converted_videos


def clean_temp_directory() -> None:
    """Remove all temporary GIF files."""
    shutil.rmtree(TEMP_GIF_PATH, ignore_errors=True)


def validate_test_output(frames_dir: Path) -> tuple[bool, int]:
    """Check that the expected frame outputs exist and return success and frame count."""
    frame_file = frames_dir / "frame_0001.bmp"
    frames_txt = frames_dir / "frames.txt"

    if not frames_dir.exists():
        print("Test case failed: output folder was not created.")
        return False, 0
    if not frame_file.exists():
        print("Test case failed: first frame file is missing.")
        return False, 0
    if not frames_txt.exists():
        print("Test case failed: frames.txt not generated.")
        return False, 0

    delays = [line for line in frames_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not delays:
        print("Test case failed: frames.txt does not contain delay entries.")
        return False, 0

    frame_count = len(list(frames_dir.glob("frame_*.bmp")))
    return True, frame_count


def run_gif_test(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    no_cleanup: bool,
) -> None:
    test_gif = Path(TEST_GIF_NAME)
    if not test_gif.exists():
        print(f"GIF test skipped: '{TEST_GIF_NAME}' not found in the working directory.")
        return

    print("Running built-in GIF test case...")
    ensure_directories()

    frames_dir = OUTPUT_PATH / test_gif.stem
    preexisting_frames = frames_dir.exists()

    destination = INPUT_GIF_PATH / test_gif.name
    preexisting_input = destination.exists()
    shutil.copy2(test_gif, destination)

    cleanup_temp = not no_cleanup
    process_media(output_size, color_depth, fps, cleanup_temp)

    success, frame_count = validate_test_output(frames_dir)
    if success:
        print(f"Test case passed: {TEST_GIF_NAME} converted successfully with {frame_count} frame(s).")

    if not no_cleanup:
        if not preexisting_frames and frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if not preexisting_input and destination.exists():
            destination.unlink()


def run_mp4_test(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    no_cleanup: bool,
) -> None:
    test_mp4 = Path(TEST_MP4_NAME)
    if not test_mp4.exists():
        print(f"MP4 test skipped: '{TEST_MP4_NAME}' not found in the working directory.")
        return

    print("Running built-in MP4 test case...")
    ensure_directories()

    frames_dir = OUTPUT_PATH / Path(TEST_MP4_NAME).stem
    preexisting_frames = frames_dir.exists()

    destination = INPUT_MP4_PATH / test_mp4.name
    preexisting_input = destination.exists()
    shutil.copy2(test_mp4, destination)

    cleanup_temp = not no_cleanup
    process_media(output_size, color_depth, fps, cleanup_temp)

    success, frame_count = validate_test_output(frames_dir)
    if success:
        print(f"MP4 test case passed: {TEST_MP4_NAME} converted successfully with {frame_count} frame(s).")

    if not no_cleanup:
        if not preexisting_frames and frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if not preexisting_input and destination.exists():
            destination.unlink()


def run_built_in_tests(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    no_cleanup: bool,
) -> None:
    run_gif_test(output_size, color_depth, fps, no_cleanup)
    run_mp4_test(output_size, color_depth, fps, no_cleanup)

    if not no_cleanup:
        clean_temp_directory()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    output_size = args.size or OUTPUT_SIZE
    color_depth = max(1, min(COLOR_DEPTH, 256))
    fps = max(1, GIF_FPS)

    cleanup_temp = not args.no_cleanup

    process_media(output_size, color_depth, fps, cleanup_temp)

    run_built_in_tests(output_size, color_depth, fps, args.no_cleanup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
