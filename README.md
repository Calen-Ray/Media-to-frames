media_to_frames
================

Command-line tooling to convert animated GIFs and MP4 videos into ESP32/ESP8266-friendly image frame sets for microcontroller displays.

`media_to_frames` uses Pillow and MoviePy to resize, quantize, and export frame sequences, letting you feed LED matrices, TFT panels, or other embedded displays without manual asset wrangling.

## Features

- Convert `.gif` animations into sequential bitmap frames
- Convert `.mp4` videos via temporary GIF generation before frame extraction
- Resize and palette-quantize frames for low-memory boards
- Override output dimensions and GIF FPS from the command line
- Auto-create required working directories on first run
- Emit optional binary framebuffers (`--emit-binary`) in RGB565, RGB332, or mono1 formats
- Built-in regression test using `GIF_TO_FRAME_TEST_CASE.gif`
- Plays nicely with typical ESP32 / ESP8266 image-display pipelines

## Requirements

- Python >= 3.8
- Pillow >= 10.0
- MoviePy >= 2.2.1 (import the clip class with `from moviepy import VideoFileClip`; `moviepy.editor` is deprecated in 2.2.1)

```bash
pip install pillow moviepy
```

If MoviePy cannot locate FFmpeg, install it separately (for example, `brew install ffmpeg` on macOS or use your OS package manager).

## Usage

```bash
python media_to_frames.py
```

Override output size (and derived GIF FPS) when needed:

```bash
python media_to_frames.py --size 240x240
```

Workflow:

1. Place source GIFs into `input_gifs/`
2. Place MP4 clips into `input_mp4s/`
3. Run the script
4. Generated assets appear under `output_frames/<media_name>/`
5. Each folder contains ordered frames (`frame_0001.bmp`, `frame_0002.bmp`, ...) plus a `timing.txt` file with per-frame delays (and optional inline comments). Add `--emit-legacy-frames-file` if you also need the older `frames.txt` format.

### Binary Output (optional)

Enable framebuffer generation alongside images:

```bash
python media_to_frames.py --emit-binary --binary-format rgb565
```

- Formats: `rgb565`, `rgb332`, or `mono1`
- Endianness for RGB565: `--endian little|big` (default little)
- Dithering for RGB332 / mono1: `--dither none|floyd|ordered`
- Emit one file per frame (default) or a single `frames.bin` with `--bin-onefile`
- Metadata lives in `output_frames/<media>/bin/metadata.json`, describing dimensions, delays, filenames, and offsets

Example microcontroller flow (per-frame files):

1. Read `metadata.json` to get width, height, pixel format, and delays
2. Stream each `frame_XXXX.bin` into your framebuffer
3. Delay according to `delays_ms[index]`

Mono1 frames use MSB-first packing per row; RGB565 honors `--endian`.

## Test Case

Running the script automatically executes regression tests:

- GIF test always runs with `GIF_TO_FRAME_TEST_CASE.gif`
- MP4 test runs when `MP4_TO_FRAME_TEST_CASE.mp4` is present
- When `--emit-binary` is supplied, additional checks validate `bin/` outputs (including a mono1/ordered run)
- Timing file validation ensures each line looks like `frame_0001.bmp    40ms`
- The test harness prints `Test case passed` on success or a descriptive failure message otherwise
- Test artifacts are cleaned up unless `--no-cleanup` is provided

## Troubleshooting

- **`ModuleNotFoundError: No module named 'moviepy.editor'`**  
  Import MoviePy's clip class directly: `from moviepy import VideoFileClip`

- **MoviePy cannot find FFmpeg**  
  Install FFmpeg (`brew install ffmpeg`, `apt-get install ffmpeg`, or download from [ffmpeg.org](https://ffmpeg.org))

- **Frames too large for ESP8266 memory**  
  Reduce `--size`, drop GIF FPS, or choose `rgb332`/`mono1` for binary output

## Project Structure

```
media_to_frames/
|-- input_gifs/
|-- input_mp4s/
|-- output_frames/
|   `-- example_gif/
|       |-- frame_0001.bmp
|       |-- frame_0002.bmp
|       |-- timing.txt
|       `-- bin/
|           |-- frame_0001.bin
|           |-- frame_0002.bin
|           `-- metadata.json
|-- temp_gifs/
|-- media_to_frames.py
`-- GIF_TO_FRAME_TEST_CASE.gif
```

## License & Attribution

Distributed under the MIT License - see [`LICENSE`](./LICENSE).

Built on the shoulders of [Pillow](https://python-pillow.org) and [MoviePy](https://zulko.github.io/moviepy/); review their licenses for redistribution guidance.

## Future Improvements

- Support exporting PNG frame sequences alongside BMP
- Provide a command-line preview of converted animations
- Emit batch compression metrics and metadata summaries for generated frame sets
- `timing.txt` lists per-frame delays in a friendly format (`frame_0001.bmp    40ms`), and `--emit-legacy-frames-file` restores the legacy `frames.txt` output when needed.
