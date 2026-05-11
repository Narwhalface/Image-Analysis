import os
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import typer

from azure_kinect_video_player.image_scaler import map_uint16_to_uint8
from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper

app = typer.Typer()


@lru_cache(maxsize=1)
def ensure_ffmpeg_on_path() -> Path:
    ffmpeg_executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = Path(__file__).resolve().parent / ".tools" / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    local_ffmpeg = ffmpeg_dir / "ffmpeg.exe"
    if not local_ffmpeg.exists():
        shutil.copy2(ffmpeg_executable, local_ffmpeg)

    local_ffprobe = ffmpeg_dir / "ffprobe.exe"
    if not local_ffprobe.exists():
        ffprobe_executable = find_ffprobe_executable(ffmpeg_executable)
        shutil.copy2(ffprobe_executable, local_ffprobe)

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    ffmpeg_dir_str = str(ffmpeg_dir)
    if ffmpeg_dir_str not in path_entries:
        os.environ["PATH"] = os.pathsep.join([ffmpeg_dir_str, *path_entries]) if path_entries else ffmpeg_dir_str

    return local_ffmpeg


def find_ffprobe_executable(ffmpeg_executable: Path) -> Path:
    sibling_ffprobe = ffmpeg_executable.with_name("ffprobe.exe")
    if sibling_ffprobe.exists():
        return sibling_ffprobe

    for candidate in iter_ffprobe_candidates():
        if candidate.exists():
            return candidate

    raise RuntimeError("Unable to find ffprobe.exe. Install FFmpeg so ffprobe is available.")


def iter_ffprobe_candidates():
    direct_match = shutil.which("ffprobe")
    if direct_match:
        yield Path(direct_match)

    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    program_files = Path(os.environ.get("ProgramFiles", ""))
    search_roots = [
        local_appdata / "Microsoft" / "WinGet" / "Packages",
        local_appdata / "Microsoft" / "WindowsApps",
        program_files,
    ]

    for root in search_roots:
        if not root.exists():
            continue
        yield from root.rglob("ffprobe.exe")


@app.command()
def app_main(
    video_filename: Path = typer.Argument(..., help="Path to an Azure Kinect .mkv recording"),
    realtime_wait: bool = typer.Option(True, help="Wait for each frame in realtime"),
    rgb: bool = typer.Option(True, help="Display the colour stream"),
    depth: bool = typer.Option(True, help="Display the depth stream"),
    ir: bool = typer.Option(False, help="Display the IR stream"),
    init_threshold: int = typer.Option(500, help="Initial far-depth threshold in millimetres"),
    init_near_threshold: int = typer.Option(250, help="Initial near-depth threshold in millimetres"),
    min_detected_points: int = typer.Option(100, help="Minimum detected depth pixels before drawing a box"),
):
    video_filename = Path(video_filename)
    ensure_ffmpeg_on_path()
    far_threshold = max(0, init_threshold)
    near_threshold = max(0, init_near_threshold)
    max_threshold_value = 32767
    display_depth_max = 10000
    click_label: tuple[int, int, str] | None = None
    current_depth_image: np.ndarray | None = None
    out_of_sensor_mask: np.ma.MaskedArray | None = None
    first_frame = True

    def on_far_threshold_change(value: int) -> None:
        nonlocal far_threshold
        far_threshold = value

    def on_near_threshold_change(value: int) -> None:
        nonlocal near_threshold
        near_threshold = value

    def depth_click_handler(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        nonlocal current_depth_image, click_label
        if event != cv2.EVENT_LBUTTONDOWN or current_depth_image is None:
            return
        if y >= current_depth_image.shape[0] or x >= current_depth_image.shape[1]:
            return
        value = int(current_depth_image[y, x])
        click_label = (x, y, f"{value}mm")
        print(f"Depth at ({x}, {y}): {value}mm")

    # Load the video file
    playback_wrapper = AzureKinectPlaybackWrapper(
        video_filename,
        realtime_wait = realtime_wait,
        auto_start = False,
        rgb = rgb,
        depth = depth,
        ir = ir,
    )

    # Create display windows
    if rgb:
        cv2.namedWindow("Colour", cv2.WINDOW_NORMAL)  # note: RGB window
    if depth:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)   # note: depth window
        cv2.namedWindow("Thresholded Depth", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Far Threshold", "Thresholded Depth", far_threshold, max_threshold_value, on_far_threshold_change)
        cv2.createTrackbar("Near Threshold", "Thresholded Depth", near_threshold, max_threshold_value, on_near_threshold_change)
        cv2.setMouseCallback("Depth", depth_click_handler)
    if ir:
        cv2.namedWindow("IR", cv2.WINDOW_NORMAL)

    start_time = time.time()
    playback_wrapper.start()  # note: start reading frames

    try:
        # Loop through frames
        for colour_image, depth_image, ir_image in playback_wrapper.grab_frame():

            # Stop if no frames left
            if colour_image is None and depth_image is None and ir_image is None:
                break
               
            if depth and depth_image is not None:
                current_depth_image = depth_image
                _, thresholded_depth_image = cv2.threshold(
                    depth_image,
                    far_threshold,
                    max_threshold_value,
                    cv2.THRESH_BINARY,
                )

                if first_frame:
                    out_of_sensor_mask = np.ma.masked_where(
                        thresholded_depth_image == 0,
                        thresholded_depth_image,
                    )
                    first_frame = False

                valid_depth_mask = depth_image > 0
                if near_threshold > 0:
                    thresholded_depth_image[
                        valid_depth_mask & (depth_image < near_threshold)
                    ] = max_threshold_value

                depth_image_8bit = map_uint16_to_uint8(depth_image, 0, display_depth_max)
                thresholded_depth_8bit = map_uint16_to_uint8(
                    thresholded_depth_image,
                    0,
                    max_threshold_value,
                )

                depth_image_colour = cv2.cvtColor(depth_image_8bit, cv2.COLOR_GRAY2BGR)
                if out_of_sensor_mask is not None:
                    depth_image_colour[out_of_sensor_mask.mask] = (0, 0, 255)

                points = np.argwhere(thresholded_depth_image == 0)
                if out_of_sensor_mask is not None and len(points) > 0:
                    bb_points = points[
                        ~out_of_sensor_mask.mask[points[:, 0], points[:, 1]]
                    ]
                else:
                    bb_points = points

                if len(bb_points) >= min_detected_points:
                    depth_image_colour[bb_points[:, 0], bb_points[:, 1]] = (255, 0, 0)
                    y, x, h, w = cv2.boundingRect(bb_points)
                    if (
                        w > 0
                        and h > 0
                        and w < depth_image.shape[1]
                        and h < depth_image.shape[0]
                    ):
                        cv2.rectangle(
                            depth_image_colour,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2,
                        )

                if click_label is not None:
                    x, y, text = click_label
                    cv2.putText(
                        depth_image_colour,
                        text,
                        (x + 10, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                cv2.putText(
                    depth_image_colour,
                    f"Near: {near_threshold}mm  Far: {far_threshold}mm",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow("Depth", depth_image_colour)
                cv2.imshow("Thresholded Depth", thresholded_depth_8bit)

            # ==============================
            # RGB DISPLAY (OPTIONAL)
            # ==============================

            if rgb and colour_image is not None:
                cv2.imshow("Colour", colour_image)  # note: raw RGB image

            if ir and ir_image is not None:
                cv2.imshow("IR", ir_image)

            # Wait for key press
            key = cv2.waitKey(1)

            if key == ord("q") or key == 27:  # note: press q or ESC to quit
                break

    except KeyboardInterrupt:
        pass

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f}s")

    # Cleanup
    cv2.destroyAllWindows()   # note: close all windows
    playback_wrapper.stop()   # note: stop video playback

    return 0


if __name__ == "__main__":
    sys.exit(app())