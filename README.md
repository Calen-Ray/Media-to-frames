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
- Built-in regression test using `GIF_TO_FRAME_TEST_CASE.gif`
- Plays nicely with typical ESP32 / ESP8266 image-display pipelines

## Requirements

- Python ≥ 3.8
- Pillow ≥ 10.0
- MoviePy ≥ 2.2.1 (import the clip class with `from moviepy import VideoFileClip`; `moviepy.editor` is deprecated in 2.2.1)

```bash
pip install pillow moviepy
```

If MoviePy cannot locate FFmpeg, install it separately (for example, `brew install ffmpeg` on macOS or your system package manager).

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
5. Each folder contains ordered frames (`frame_0001.bmp`, `frame_0002.bmp`, …) plus a `frames.txt` file with per-frame delays in milliseconds

## Test Case

Running the script automatically executes a regression test using `GIF_TO_FRAME_TEST_CASE.gif`:

- Confirms conversion completes successfully
- Verifies `frames.txt` exists with at least one delay entry
- Ensures `frame_0001.bmp` (and subsequent frames) are generated
- Prints `Test case passed` on success, or a descriptive failure message otherwise
- Cleans up generated test assets unless `--no-cleanup` is provided

If `MP4_TO_FRAME_TEST_CASE.mp4` is present, an MP4 conversion test runs as well.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'moviepy.editor'`**  
  Import MoviePy’s clip class directly: `from moviepy import VideoFileClip`

- **MoviePy cannot find FFmpeg**  
  Install FFmpeg (`brew install ffmpeg`, `apt-get install ffmpeg`, or download from [ffmpeg.org](https://ffmpeg.org))

- **Frames too large for ESP8266 memory**  
  Reduce `--size` or lower the GIF FPS constant to shrink generated assets

## Project Structure

```
media_to_frames/
├── input_gifs/
├── input_mp4s/
├── output_frames/
│   └── example_gif/
│       ├── frame_0001.bmp
│       ├── frame_0002.bmp
│       └── frames.txt
├── temp_gifs/
├── media_to_frames.py
└── GIF_TO_FRAME_TEST_CASE.gif
```

## License & Attribution

Distributed under the MIT License — see [`LICENSE`](./LICENSE).

Built on the shoulders of [Pillow](https://python-pillow.org) and [MoviePy](https://zulko.github.io/moviepy/); review their licenses for redistribution guidance.

## Future Improvements

- Support exporting PNG frame sequences alongside BMP
- Provide a command-line preview of converted animations
- Emit batch compression metrics and metadata summaries for generated frame sets

