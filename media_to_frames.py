"""Convert local GIF and MP4 media into frame sequences suitable for ESP displays.

Place .gif files in input_gifs/, .mp4 files in input_mp4s/, run the script,
and review the generated folders under output_frames/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

try:  # MoviePy >= 2.2.1 exposes VideoFileClip at the package top level.
    from moviepy import VideoFileClip
except ImportError:  # Older releases keep it under moviepy.editor.
    from moviepy.editor import VideoFileClip  # type: ignore[attr-defined]

from PIL import Image, UnidentifiedImageError

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

PIL_DITHER = {
    "none": Image.Dither.NONE,
    "floyd": Image.Dither.FLOYDSTEINBERG,
    "ordered": Image.Dither.ORDERED,
}


def _build_rgb332_palette() -> List[int]:
    palette: List[int] = []
    for r in range(8):
        rr = int(round(r * 255 / 7))
        for g in range(8):
            gg = int(round(g * 255 / 7))
            for b in range(4):
                bb = int(round(b * 255 / 3))
                palette.extend([rr, gg, bb])
    return palette


RGB332_PALETTE_IMAGE = Image.new("P", (1, 1))
RGB332_PALETTE_IMAGE.putpalette(_build_rgb332_palette())


@dataclass
class BinaryOptions:
    enabled: bool = False
    pixel_format: str = "rgb565"
    endian: str = "little"
    dither: str = "none"
    pack_order: str = "row-major"
    single_file: bool = False
    bin_subdir: str = "bin"

    def with_overrides(self, **kwargs: object) -> "BinaryOptions":
        return replace(self, **kwargs)


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


def composite_gif_frames(im: Image.Image) -> Iterator[Tuple[Image.Image, int]]:
    """Yield fully composited RGBA frames and their delays from an animated GIF."""
    try:
        im.seek(0)
    except EOFError:
        return

    while True:
        duration = int(im.info.get("duration", 0) or 0)
        frame = im.copy().convert("RGBA")
        base = Image.new("RGBA", im.size, (0, 0, 0, 255))
        base.alpha_composite(frame)
        yield base, duration
        try:
            im.seek(im.tell() + 1)
        except EOFError:
            break


def quantize_for_bmp(rgb_image: Image.Image, color_depth: int) -> Image.Image:
    """Quantize an RGB image for BMP output."""
    colors = max(1, min(color_depth, 256))
    return rgb_image.quantize(colors=colors, method=Image.MEDIANCUT)


def quantize_frame(frame: Image.Image, output_size: Tuple[int, int], color_depth: int) -> Image.Image:
    """Resize and quantize a single frame for BMP output."""
    resized = frame.resize(output_size, RESAMPLE_FILTER)
    rgb = resized.convert("RGB")
    return quantize_for_bmp(rgb, color_depth)


def bytes_per_frame(width: int, height: int, pixel_format: str) -> int:
    if pixel_format == "rgb565":
        return width * height * 2
    if pixel_format == "rgb332":
        return width * height
    if pixel_format == "mono1":
        return ((width + 7) // 8) * height
    raise ValueError(f"Unsupported pixel format: {pixel_format}")


def convert_image_to_buffer(
    image: Image.Image,
    options: BinaryOptions,
    width: int,
    height: int,
) -> bytes:
    """Convert an RGB image to the requested binary framebuffer format."""
    if options.pixel_format == "rgb565":
        rgb = image.convert("RGB")
        data = rgb.tobytes()
        total_pixels = width * height
        out = bytearray(total_pixels * 2)
        little_endian = options.endian == "little"
        for idx in range(total_pixels):
            r = data[idx * 3]
            g = data[idx * 3 + 1]
            b = data[idx * 3 + 2]
            value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            if little_endian:
                out[idx * 2] = value & 0xFF
                out[idx * 2 + 1] = (value >> 8) & 0xFF
            else:
                out[idx * 2] = (value >> 8) & 0xFF
                out[idx * 2 + 1] = value & 0xFF
        return bytes(out)

    if options.pixel_format == "rgb332":
        rgb = image.convert("RGB")
        dither_mode = PIL_DITHER[options.dither]
        paletted = rgb.quantize(palette=RGB332_PALETTE_IMAGE, dither=dither_mode)
        return paletted.tobytes()

    if options.pixel_format == "mono1":
        gray = image.convert("L")
        dither_mode = PIL_DITHER[options.dither]
        mono = gray.convert("1", dither=dither_mode)
        return mono.tobytes()

    raise ValueError(f"Unsupported pixel format: {options.pixel_format}")


class BinaryFrameWriter:
    """Handle writing binary frame buffers and metadata."""

    def __init__(self, frame_dir: Path, options: BinaryOptions, width: int, height: int) -> None:
        self.options = options
        self.width = width
        self.height = height
        self.single_file = options.single_file
        self.frame_count = 0
        self.expected_length = bytes_per_frame(width, height, options.pixel_format)
        self.filenames: List[str] = []
        self.frame_offsets: List[int] = []
        self.current_offset = 0
        self._bin_handle = None
        self._tmp_bin_path: Path | None = None
        self.bin_dir = self._prepare_bin_dir(frame_dir, options.bin_subdir)

    @staticmethod
    def _prepare_bin_dir(frame_dir: Path, subdir: str) -> Path:
        name = subdir.strip() or "bin"
        sanitized = Path(name).name  # drop nested components
        dest = frame_dir / sanitized
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def add_frame(self, image: Image.Image, index: int) -> None:
        buffer = convert_image_to_buffer(image, self.options, self.width, self.height)
        if len(buffer) != self.expected_length:
            raise ValueError(
                f"Unexpected buffer length {len(buffer)} (expected {self.expected_length}) for frame {index}"
            )

        if self.single_file:
            if self._bin_handle is None:
                self._tmp_bin_path = self.bin_dir / "frames.bin.tmp"
                self._bin_handle = open(self._tmp_bin_path, "wb")
            self.frame_offsets.append(self.current_offset)
            self._bin_handle.write(buffer)
            self.current_offset += len(buffer)
            self.filenames = ["frames.bin"]
        else:
            filename = f"frame_{index:04d}.bin"
            tmp_path = self.bin_dir / f"{filename}.tmp"
            final_path = self.bin_dir / filename
            with open(tmp_path, "wb") as fh:
                fh.write(buffer)
            tmp_path.replace(final_path)
            self.filenames.append(filename)

        self.frame_count += 1

    def finalize(self, delays: List[int]) -> None:
        if self.single_file and self._bin_handle and self._tmp_bin_path:
            self._bin_handle.close()
            final_path = self.bin_dir / "frames.bin"
            self._tmp_bin_path.replace(final_path)
        metadata = {
            "width": self.width,
            "height": self.height,
            "pixel_format": self.options.pixel_format,
            "endian": self.options.endian,
            "dither": self.options.dither,
            "pack_order": self.options.pack_order,
            "frame_count": self.frame_count,
            "delays_ms": [int(d) for d in delays],
            "filenames": self.filenames,
        }
        if self.single_file:
            metadata["frame_offsets"] = self.frame_offsets
            metadata["frame_length_bytes"] = self.expected_length
        metadata_path = self.bin_dir / "metadata.json"
        tmp_metadata = metadata_path.with_suffix(".json.tmp")
        with open(tmp_metadata, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        tmp_metadata.replace(metadata_path)

    def abort(self) -> None:
        shutil.rmtree(self.bin_dir, ignore_errors=True)


def save_frames_from_gif(
    gif_path: Path,
    output_dir: Path,
    output_size: Tuple[int, int],
    color_depth: int,
    binary_options: BinaryOptions,
    output_name: str | None = None,
) -> bool:
    """Extract frames from a GIF file, save BMP frames, and optionally emit binaries."""
    destination_name = output_name or gif_path.stem
    frame_dir = output_dir / destination_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    delays: List[int] = []
    binary_writer = BinaryFrameWriter(frame_dir, binary_options, output_size[0], output_size[1]) if binary_options.enabled else None

    try:
        with Image.open(gif_path) as img:
            for index, (frame_rgba, delay_ms) in enumerate(composite_gif_frames(img), start=1):
                resized = frame_rgba.resize(output_size, RESAMPLE_FILTER)
                rgb_frame = resized.convert("RGB")
                bmp_frame = quantize_for_bmp(rgb_frame, color_depth)

                frame_path = frame_dir / f"frame_{index:04d}.bmp"
                bmp_frame.save(frame_path, format="BMP")

                delays.append(int(delay_ms))
                if binary_writer:
                    binary_writer.add_frame(rgb_frame, index)

    except (UnidentifiedImageError, OSError) as exc:
        logging.warning("Skipping corrupted GIF '%s': %s", gif_path.name, exc)
        if binary_writer:
            binary_writer.abort()
        shutil.rmtree(frame_dir, ignore_errors=True)
        return False
    except Exception as exc:
        logging.warning("Unexpected error processing '%s': %s", gif_path.name, exc)
        if binary_writer:
            binary_writer.abort()
        shutil.rmtree(frame_dir, ignore_errors=True)
        return False

    if not delays:
        logging.warning("No frames extracted from '%s'; removing empty output", gif_path.name)
        if binary_writer:
            binary_writer.abort()
        shutil.rmtree(frame_dir, ignore_errors=True)
        return False

    frames_txt = frame_dir / "frames.txt"
    frames_txt.write_text("\n".join(str(delay) for delay in delays), encoding="utf-8")

    if binary_writer:
        binary_writer.finalize(delays)

    return True


def close_clip(clip: object) -> None:
    """Safely close a MoviePy clip if it provides a close() method."""
    close_fn = getattr(clip, "close", None)
    if callable(close_fn):
        close_fn()


def convert_mp4_to_gif(mp4_path: Path, target_size: Tuple[int, int], fps: int) -> Path | None:
    """Convert an MP4 to a temporary GIF stored under TEMP_GIF_PATH."""
    print(f"Converting video: {mp4_path.name}...")
    TEMP_GIF_PATH.mkdir(parents=True, exist_ok=True)
    temp_gif_path = TEMP_GIF_PATH / f"{mp4_path.stem}.gif"

    clip = None
    try:
        clip = VideoFileClip(str(mp4_path))
        cropped = clip.resize(target_size)
        cropped.write_gif(
            str(temp_gif_path),
            fps=fps,
            program="ffmpeg",
            logger=None,
        )
    except Exception as exc:
        logging.warning("Skipping video '%s': %s", mp4_path.name, exc)
        if temp_gif_path.exists():
            temp_gif_path.unlink()
        return None
    finally:
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
    binary_options: BinaryOptions,
) -> bool:
    """Convert an MP4 to GIF and then extract frames using the GIF workflow."""
    temp_gif = convert_mp4_to_gif(mp4_path, output_size, fps)
    if temp_gif is None:
        return False

    try:
        print(f"Extracting frames from {mp4_path.name}...")
        return save_frames_from_gif(
            temp_gif,
            output_dir,
            output_size,
            color_depth,
            binary_options,
            output_name=mp4_path.stem,
        )
    finally:
        if cleanup_temp and temp_gif.exists():
            temp_gif.unlink()


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
    parser.add_argument(
        "--emit-binary",
        action="store_true",
        help="Emit binary framebuffer output alongside image frames.",
    )
    parser.add_argument(
        "--binary-format",
        choices=("rgb565", "rgb332", "mono1"),
        default="rgb565",
        help="Binary pixel format to emit when --emit-binary is used (default: rgb565).",
    )
    parser.add_argument(
        "--endian",
        choices=("little", "big"),
        default="little",
        help="Endianness for multi-byte pixel formats (default: little).",
    )
    parser.add_argument(
        "--dither",
        choices=("none", "floyd", "ordered"),
        default="none",
        help="Dithering mode for rgb332 and mono1 conversions (default: none).",
    )
    parser.add_argument(
        "--pack-order",
        choices=("row-major",),
        default="row-major",
        help="Frame buffer scanning order (reserved for future use).",
    )
    parser.add_argument(
        "--bin-onefile",
        action="store_true",
        help="Emit a single frames.bin file plus metadata instead of per-frame binaries.",
    )
    return parser


def process_media(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    cleanup_temp: bool,
    binary_options: BinaryOptions,
) -> Tuple[int, int, int, int]:
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
        print(f"Processing {gif_path.name}...")
        if save_frames_from_gif(gif_path, OUTPUT_PATH, output_size, color_depth, binary_options):
            converted_gifs += 1

    converted_videos = 0
    for mp4_path in videos:
        if process_mp4(mp4_path, OUTPUT_PATH, output_size, color_depth, fps, cleanup_temp, binary_options):
            converted_videos += 1

    print(
        f"Finished. Successfully converted {converted_gifs} of {len(gifs)} GIF(s) "
        f"and {converted_videos} of {len(videos)} video(s)."
    )

    return len(gifs), converted_gifs, len(videos), converted_videos


def clean_temp_directory() -> None:
    """Remove all temporary GIF files."""
    shutil.rmtree(TEMP_GIF_PATH, ignore_errors=True)


def validate_binary_output(frames_dir: Path, binary_options: BinaryOptions) -> None:
    """Validate binary framebuffer output for tests."""
    bin_dir = frames_dir / (binary_options.bin_subdir.strip() or "bin")
    if not bin_dir.exists():
        raise AssertionError(f"Binary output missing at {bin_dir}")

    metadata_path = bin_dir / "metadata.json"
    if not metadata_path.exists():
        raise AssertionError("metadata.json missing from binary output")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_count = metadata.get("frame_count", 0)
    delays = metadata.get("delays_ms", [])
    if frame_count <= 0:
        raise AssertionError("metadata.json reports zero frames")
    if len(delays) != frame_count:
        raise AssertionError("delays_ms length does not match frame_count")
    if metadata.get("pixel_format") != binary_options.pixel_format:
        raise AssertionError("Pixel format in metadata does not match expected value")
    if metadata.get("endian") != binary_options.endian:
        raise AssertionError("Endian in metadata does not match expected value")

    width = metadata.get("width")
    height = metadata.get("height")
    expected_length = bytes_per_frame(width, height, binary_options.pixel_format)

    filenames = metadata.get("filenames", [])
    frames_bin = bin_dir / "frames.bin"
    if binary_options.single_file:
        if not frames_bin.exists():
            raise AssertionError("frames.bin missing despite --bin-onefile")
        if frames_bin.stat().st_size != expected_length * frame_count:
            raise AssertionError("frames.bin size does not match expected total length")
        offsets = metadata.get("frame_offsets", [])
        frame_len = metadata.get("frame_length_bytes")
        if len(offsets) != frame_count:
            raise AssertionError("frame_offsets length does not match frame_count")
        if frame_len != expected_length:
            raise AssertionError("frame_length_bytes incorrect in metadata")
    else:
        if len(filenames) != frame_count:
            raise AssertionError("Number of binary frame files does not match frame_count")
        for name in filenames:
            bin_path = bin_dir / name
            if not bin_path.exists():
                raise AssertionError(f"Frame binary missing: {name}")
            if bin_path.stat().st_size != expected_length:
                raise AssertionError(f"Unexpected frame size in {name}")


def run_gif_test(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    no_cleanup: bool,
    binary_options: BinaryOptions,
) -> None:
    test_gif = Path(TEST_GIF_NAME)
    if not test_gif.exists():
        print(f"GIF test skipped: '{TEST_GIF_NAME}' not found in the working directory.")
        return

    print("Running built-in GIF test case...")
    ensure_directories()

    frames_dir = OUTPUT_PATH / test_gif.stem
    if frames_dir.exists() and not no_cleanup:
        shutil.rmtree(frames_dir, ignore_errors=True)

    destination = INPUT_GIF_PATH / test_gif.name
    preexisting_input = destination.exists()
    shutil.copy2(test_gif, destination)

    cleanup_temp = not no_cleanup
    process_media(output_size, color_depth, fps, cleanup_temp, binary_options)

    frame_file = frames_dir / "frame_0001.bmp"
    frames_txt = frames_dir / "frames.txt"

    if not frames_dir.exists():
        print("Test case failed: output folder was not created.")
    elif not frame_file.exists():
        print("Test case failed: first frame file is missing.")
    elif not frames_txt.exists():
        print("Test case failed: frames.txt not generated.")
    else:
        delays = [line for line in frames_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not delays:
            print("Test case failed: frames.txt does not contain delay entries.")
        else:
            if binary_options.enabled:
                validate_binary_output(frames_dir, binary_options)
            frame_count = len(list(frames_dir.glob("frame_*.bmp")))
            print(
                f"Test case passed: {TEST_GIF_NAME} converted successfully with {frame_count} frame(s)."
            )

    if not no_cleanup:
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if not preexisting_input and destination.exists():
            destination.unlink()


def run_mp4_test(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    no_cleanup: bool,
    binary_options: BinaryOptions,
) -> None:
    test_mp4 = Path(TEST_MP4_NAME)
    if not test_mp4.exists():
        print(f"MP4 test skipped: '{TEST_MP4_NAME}' not found in the working directory.")
        return

    print("Running built-in MP4 test case...")
    ensure_directories()

    frames_dir = OUTPUT_PATH / Path(TEST_MP4_NAME).stem
    if frames_dir.exists() and not no_cleanup:
        shutil.rmtree(frames_dir, ignore_errors=True)

    destination = INPUT_MP4_PATH / test_mp4.name
    preexisting_input = destination.exists()
    shutil.copy2(test_mp4, destination)

    cleanup_temp = not no_cleanup
    process_media(output_size, color_depth, fps, cleanup_temp, binary_options)

    frame_file = frames_dir / "frame_0001.bmp"
    frames_txt = frames_dir / "frames.txt"

    if not frames_dir.exists():
        print("MP4 test case failed: output folder was not created.")
    elif not frame_file.exists():
        print("MP4 test case failed: first frame file is missing.")
    elif not frames_txt.exists():
        print("MP4 test case failed: frames.txt not generated.")
    else:
        delays = [line for line in frames_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not delays:
            print("MP4 test case failed: frames.txt does not contain delay entries.")
        else:
            if binary_options.enabled:
                validate_binary_output(frames_dir, binary_options)
            frame_count = len(list(frames_dir.glob("frame_*.bmp")))
            print(
                f"MP4 test case passed: {TEST_MP4_NAME} converted successfully with {frame_count} frame(s)."
            )

    if not no_cleanup:
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if not preexisting_input and destination.exists():
            destination.unlink()


def run_built_in_tests(
    output_size: Tuple[int, int],
    color_depth: int,
    fps: int,
    no_cleanup: bool,
    binary_options: BinaryOptions,
) -> None:
    run_gif_test(output_size, color_depth, fps, no_cleanup, binary_options)
    run_mp4_test(output_size, color_depth, fps, no_cleanup, binary_options)

    if binary_options.enabled:
        mono_options = binary_options.with_overrides(pixel_format="mono1", dither="ordered", bin_subdir="bin_mono1")
        run_gif_test(output_size, color_depth, fps, no_cleanup, mono_options)
        print("Binary pipeline test passed: mono1 format validated.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    output_size = args.size or OUTPUT_SIZE
    color_depth = max(1, min(COLOR_DEPTH, 256))
    fps = max(1, GIF_FPS)

    cleanup_temp = not args.no_cleanup

    binary_options = BinaryOptions(
        enabled=args.emit_binary,
        pixel_format=args.binary_format,
        endian=args.endian,
        dither=args.dither,
        pack_order=args.pack_order,
        single_file=args.bin_onefile,
    )

    process_media(output_size, color_depth, fps, cleanup_temp, binary_options)

    if cleanup_temp:
        clean_temp_directory()

    run_built_in_tests(output_size, color_depth, fps, args.no_cleanup, binary_options)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
