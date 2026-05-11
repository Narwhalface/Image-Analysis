import json
import os
import shutil
import sys
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import typer

from azure_kinect_video_player.image_scaler import map_uint16_to_uint8
from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper

from vi2026_pythonpackage.features import extract_multimodal_features, normalise_to_uint8, prepare_feature_image

app = typer.Typer()


class DetectorType(str, Enum):
    NONE = "none"
    ORB = "orb"
    HARRIS = "harris"
    SIFT = "sift"
    HOG = "hog"


class DatasetType(str, Enum):
    TRAINING = "training"
    TESTING = "testing"


WORKSPACE_ROOT = Path(__file__).resolve().parent
LABEL_PREVIEW_DIRNAME = "_label_previews"


def resolve_classifier_model_path(model_path: Path | None) -> Path | None:
    if model_path is not None:
        return model_path

    default_candidates = [
        WORKSPACE_ROOT / "training_data_massive" / "classifier.yml",
        WORKSPACE_ROOT / "classifier.yml",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate
    return None


def load_classifier_metadata(model_path: Path, metadata_path: Path | None) -> dict:
    resolved_metadata = metadata_path if metadata_path is not None else model_path.with_suffix(".json")
    if not resolved_metadata.exists():
        raise RuntimeError(f"Could not find classifier metadata: {resolved_metadata}")
    return json.loads(resolved_metadata.read_text(encoding="utf-8"))


def predict_object_class(
    crop: np.ndarray,
    svm: cv2.ml.SVM,
    orb: cv2.ORB,
    class_names: list[str],
    nfeatures: int,
) -> str:
    feature_vector = extract_multimodal_features(crop, orb, nfeatures=nfeatures)
    _, prediction = svm.predict(feature_vector.reshape(1, -1).astype(np.float32))
    class_index = int(prediction.flatten()[0])
    if class_index < 0 or class_index >= len(class_names):
        return "Unknown"
    return class_names[class_index]


def resolve_recording_path(video_filename: Path | None, dataset: DatasetType) -> Path:
    if video_filename is not None:
        return Path(video_filename)

    default_filename = {
        DatasetType.TRAINING: "kinect-training-set.mkv",
        DatasetType.TESTING: "kinect-testing-set.mkv",
    }[dataset]
    default_path = WORKSPACE_ROOT / default_filename
    if not default_path.exists():
        raise RuntimeError(f"Could not find default {dataset.value} recording: {default_path}")
    return default_path


def resolve_output_dir(output_dir: Path | None, dataset: DatasetType) -> Path:
    if output_dir is not None:
        return output_dir

    default_dirname = {
        DatasetType.TRAINING: "training_data",
        DatasetType.TESTING: "testing_data",
    }[dataset]
    return WORKSPACE_ROOT / default_dirname


def extract_orb_features(img: np.ndarray, orb: cv2.ORB, nfeatures: int = 100) -> np.ndarray:
    _, descriptors = orb.detectAndCompute(img, None)
    feature = np.zeros(nfeatures * 32, dtype=np.float32)
    if descriptors is not None:
        flat = descriptors.flatten().astype(np.float32)
        feature[: min(len(flat), len(feature))] = flat[: min(len(flat), len(feature))]
    return feature


def extract_sift_features(img: np.ndarray, sift: cv2.SIFT, nfeatures: int = 100) -> np.ndarray:
    _, descriptors = sift.detectAndCompute(img, None)
    feature = np.zeros(nfeatures * 128, dtype=np.float32)
    if descriptors is not None:
        flat = descriptors.flatten().astype(np.float32)
        feature[: min(len(flat), len(feature))] = flat[: min(len(flat), len(feature))]
    return feature


def extract_harris_features(img: np.ndarray, fixed_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    grey = np.float32(img)
    corners = cv2.cornerHarris(grey, blockSize=2, ksize=3, k=0.04)
    corners_resized = cv2.resize(corners, fixed_size)
    return corners_resized.flatten().astype(np.float32)


def extract_hog_features(
    img: np.ndarray,
    cell_size: tuple[int, int] = (8, 8),
    block_size: tuple[int, int] = (2, 2),
    nbins: int = 9,
) -> tuple[np.ndarray, np.ndarray | None]:
    if img.shape[0] < 16 or img.shape[1] < 16:
        return np.zeros(nbins, dtype=np.float32), None

    win_size = (
        img.shape[1] // cell_size[1] * cell_size[1],
        img.shape[0] // cell_size[0] * cell_size[0],
    )
    if win_size[0] < 16 or win_size[1] < 16:
        return np.zeros(nbins, dtype=np.float32), None

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins,
    )
    trimmed = img[: win_size[1], : win_size[0]]
    hog_features_cv = hog.compute(trimmed)

    n_cells = (trimmed.shape[0] // cell_size[0], trimmed.shape[1] // cell_size[1])
    hog_features = hog_features_cv.reshape(
        n_cells[1] - block_size[1] + 1,
        n_cells[0] - block_size[0] + 1,
        block_size[0],
        block_size[1],
        nbins,
    ).transpose((1, 0, 2, 3, 4))

    gradients = np.zeros((n_cells[0], n_cells[1], nbins), dtype=np.float32)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=np.int32)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[
                off_y : n_cells[0] - block_size[0] + off_y + 1,
                off_x : n_cells[1] - block_size[1] + off_x + 1,
            ] += hog_features[:, :, off_y, off_x, :]
            cell_count[
                off_y : n_cells[0] - block_size[0] + off_y + 1,
                off_x : n_cells[1] - block_size[1] + off_x + 1,
            ] += 1

    gradients /= np.maximum(cell_count, 1)
    histogram = gradients.mean(axis=(0, 1)).astype(np.float32)
    show_image = gradients[:, :, min(5, nbins - 1)]
    visualisation = cv2.resize(
        show_image,
        (show_image.shape[1] * 4, show_image.shape[0] * 4),
        interpolation=cv2.INTER_NEAREST,
    )
    visualisation = cv2.normalize(visualisation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return histogram, visualisation


def build_feature_visualisation(
    crop: np.ndarray,
    detector_type: DetectorType,
    orb: cv2.ORB | None,
    sift: cv2.SIFT | None,
    nfeatures: int,
) -> tuple[np.ndarray | None, str | None]:
    feature_img = prepare_feature_image(crop)

    if detector_type == DetectorType.NONE:
        return None, None

    if detector_type == DetectorType.ORB and orb is not None:
        keypoints, _ = orb.detectAndCompute(feature_img, None)
        feature_vector = extract_orb_features(feature_img, orb, nfeatures=nfeatures)
        visualisation = cv2.drawKeypoints(
            feature_img,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return visualisation, f"ORB: {len(keypoints)} keypoints, {len(feature_vector)} values"

    if detector_type == DetectorType.SIFT and sift is not None:
        keypoints, _ = sift.detectAndCompute(feature_img, None)
        feature_vector = extract_sift_features(feature_img, sift, nfeatures=nfeatures)
        visualisation = cv2.drawKeypoints(
            feature_img,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return visualisation, f"SIFT: {len(keypoints)} keypoints, {len(feature_vector)} values"

    if detector_type == DetectorType.HARRIS:
        corners = cv2.cornerHarris(np.float32(feature_img), blockSize=2, ksize=3, k=0.04)
        corners_dilated = cv2.dilate(corners, None)
        visualisation = cv2.cvtColor(feature_img, cv2.COLOR_GRAY2BGR)
        visualisation[corners_dilated > 0.01 * corners_dilated.max()] = (0, 255, 0)
        feature_vector = extract_harris_features(feature_img)
        return visualisation, f"Harris: {len(feature_vector)} values"

    if detector_type == DetectorType.HOG:
        feature_vector, visualisation = extract_hog_features(feature_img)
        return visualisation, f"HOG: {len(feature_vector)} values"

    return None, None


def save_frame_image(image: np.ndarray, output_dir: Path, frame_count: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_count:06d}.png"
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Failed to save frame image to {output_path}")
    return output_path


def save_label_preview(image: np.ndarray, output_dir: Path, frame_count: int) -> Path:
    preview_dir = output_dir / LABEL_PREVIEW_DIRNAME
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / f"frame_{frame_count:06d}.png"
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Failed to save preview image to {output_path}")
    return output_path


def should_save_frame(
    image: np.ndarray,
    frame_count: int,
    last_saved_image: np.ndarray | None,
    last_saved_frame: int | None,
    min_save_gap: int,
    min_crop_change: float,
) -> bool:
    if last_saved_frame is not None and frame_count - last_saved_frame < min_save_gap:
        return False

    if last_saved_image is None:
        return True

    resized_previous = cv2.resize(last_saved_image, (64, 64), interpolation=cv2.INTER_AREA)
    resized_current = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(resized_previous, resized_current)
    mean_difference = float(np.mean(difference))
    return mean_difference >= min_crop_change


def prepare_panel_image(image: np.ndarray | None, title: str, fallback_shape: tuple[int, int]) -> np.ndarray:
    height, width = fallback_shape
    if image is None:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        panel = image.copy()
        if panel.dtype == np.uint16:
            panel = cv2.normalize(panel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif panel.dtype != np.uint8:
            panel = cv2.normalize(panel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)

    cv2.putText(
        panel,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return panel


def build_saved_screenshot(
    colour_image: np.ndarray | None,
    depth_display: np.ndarray,
    ir_image: np.ndarray | None,
) -> np.ndarray:
    panel_height, panel_width = depth_display.shape[:2]
    colour_panel = prepare_panel_image(colour_image, "Colour", (panel_height, panel_width))
    depth_panel = prepare_panel_image(depth_display, "Depth", (panel_height, panel_width))
    ir_panel = prepare_panel_image(ir_image, "IR", (panel_height, panel_width))
    spacer = np.zeros((panel_height, 20, 3), dtype=np.uint8)
    return np.hstack([colour_panel, spacer, depth_panel, spacer.copy(), ir_panel])


def crop_modality_image(
    image: np.ndarray | None,
    x: int,
    y: int,
    w: int,
    h: int,
    reference_shape: tuple[int, int],
) -> np.ndarray | None:
    if image is None:
        return None

    ref_height, ref_width = reference_shape
    image_height, image_width = image.shape[:2]
    scale_x = image_width / ref_width
    scale_y = image_height / ref_height

    left = max(0, min(image_width - 1, int(round(x * scale_x))))
    top = max(0, min(image_height - 1, int(round(y * scale_y))))
    right = max(left + 1, min(image_width, int(round((x + w) * scale_x))))
    bottom = max(top + 1, min(image_height, int(round((y + h) * scale_y))))
    return image[top:bottom, left:right].copy()


def build_multimodal_crop(
    colour_crop: np.ndarray | None,
    depth_crop: np.ndarray,
    ir_crop: np.ndarray | None,
) -> np.ndarray:
    panel_height, panel_width = depth_crop.shape[:2]

    def prepare_crop_panel(image: np.ndarray | None) -> np.ndarray:
        if image is None or image.size == 0:
            panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        else:
            panel = normalise_to_uint8(image)
            if panel.ndim == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        return panel

    colour_panel = prepare_crop_panel(colour_crop)
    depth_panel = prepare_crop_panel(depth_crop)
    ir_panel = prepare_crop_panel(ir_crop)
    return np.hstack([colour_panel, depth_panel, ir_panel])


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
    video_filename: Path | None = typer.Argument(None, help="Path to an Azure Kinect .mkv recording"),
    dataset: DatasetType = typer.Option(DatasetType.TRAINING, help="Use the bundled training or testing recording when no video path is provided"),
    realtime_wait: bool = typer.Option(True, help="Wait for each frame in realtime"),
    rgb: bool = typer.Option(True, help="Display the colour stream"),
    depth: bool = typer.Option(True, help="Display the depth stream"),
    ir: bool = typer.Option(False, help="Display the IR stream"),
    init_threshold: int = typer.Option(500, help="Initial far-depth threshold in millimetres"),
    init_near_threshold: int = typer.Option(250, help="Initial near-depth threshold in millimetres"),
    min_detected_points: int = typer.Option(100, help="Minimum detected depth pixels before drawing a box"),
    output_dir: Path | None = typer.Option(None, help="Folder where detected depth crops are saved"),
    class_label: str | None = typer.Option(None, help="Optional class label subfolder for saved crops"),
    save_every_detection: bool = typer.Option(False, help="Save each detected crop automatically"),
    min_save_gap: int = typer.Option(15, help="Minimum number of frames between automatic saves"),
    min_crop_change: float = typer.Option(8.0, help="Minimum mean pixel difference before saving another crop"),
    detector_type: DetectorType = typer.Option(DetectorType.NONE, help="Feature detector to visualise on the crop"),
    nfeatures: int = typer.Option(100, help="Maximum number of ORB or SIFT keypoints to use"),
    classifier_model: Path | None = typer.Option(None, help="Optional path to a trained classifier model for live prediction"),
    classifier_metadata: Path | None = typer.Option(None, help="Optional path to the classifier metadata JSON file"),
):
    video_filename = resolve_recording_path(video_filename, dataset)
    output_dir = resolve_output_dir(output_dir, dataset)
    classifier_model = resolve_classifier_model_path(classifier_model)
    ensure_ffmpeg_on_path()
    far_threshold = max(0, init_threshold)
    near_threshold = max(0, init_near_threshold)
    max_threshold_value = 32767
    display_depth_max = 10000
    click_label: tuple[int, int, str] | None = None
    current_depth_image: np.ndarray | None = None
    out_of_sensor_mask: np.ma.MaskedArray | None = None
    first_frame = True
    frame_count = 0
    saved_output_dir = output_dir / class_label if class_label else output_dir
    last_detected_crop: np.ndarray | None = None
    last_display_frame: np.ndarray | None = None
    last_colour_frame: np.ndarray | None = None
    last_ir_frame: np.ndarray | None = None
    last_feature_summary: str | None = None
    last_prediction_label: str | None = None
    last_saved_image: np.ndarray | None = None
    last_saved_frame: int | None = None

    print(f"Using recording: {video_filename}")
    print(f"Saving crops to: {saved_output_dir}")
    print(f"Auto-save gap: {min_save_gap} frames, crop change threshold: {min_crop_change}")

    orb = cv2.ORB_create(nfeatures=nfeatures) if detector_type == DetectorType.ORB else None
    sift = None
    if detector_type == DetectorType.SIFT:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not include SIFT support.")
        sift = cv2.SIFT_create(nfeatures=nfeatures)

    classifier = None
    classifier_orb = None
    classifier_class_names: list[str] | None = None
    classifier_nfeatures = nfeatures
    if classifier_model is not None:
        if not classifier_model.exists():
            raise RuntimeError(f"Could not find classifier model: {classifier_model}")
        metadata = load_classifier_metadata(classifier_model, classifier_metadata)
        if metadata.get("feature_extractor") not in {"orb", "orb_colour_ir"}:
            raise RuntimeError("Live classification only supports ORB-based models.")
        classifier_class_names = list(metadata["class_names"])
        classifier_nfeatures = int(metadata["nfeatures"])
        classifier = cv2.ml.SVM_load(str(classifier_model))
        classifier_orb = cv2.ORB_create(nfeatures=classifier_nfeatures)
        print(f"Live classifier: {classifier_model}")
        print(f"Live classes: {', '.join(classifier_class_names)}")

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
        if detector_type != DetectorType.NONE:
            cv2.namedWindow("Features", cv2.WINDOW_NORMAL)
    if ir:
        cv2.namedWindow("IR", cv2.WINDOW_NORMAL)

    start_time = time.time()
    playback_wrapper.start()  # note: start reading frames

    try:
        # Loop through frames
        for colour_image, depth_image, ir_image in playback_wrapper.grab_frame():
            frame_count += 1
            last_detected_crop = None
            last_display_frame = None
            last_colour_frame = None
            last_ir_frame = None
            feature_visualisation = None
            last_feature_summary = None
            last_prediction_label = None

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
                    detection_mask = np.zeros(depth_image.shape, dtype=np.uint8)
                    detection_mask[bb_points[:, 0], bb_points[:, 1]] = 255
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(detection_mask, connectivity=8)

                    if num_labels <= 1:
                        continue

                    component_areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_label = int(np.argmax(component_areas)) + 1
                    largest_component = labels == largest_label
                    component_points = np.argwhere(largest_component)

                    if len(component_points) < min_detected_points:
                        continue

                    depth_image_colour[component_points[:, 0], component_points[:, 1]] = (255, 0, 0)
                    component_points_xy = component_points[:, [1, 0]].astype(np.int32)
                    x, y, w, h = cv2.boundingRect(component_points_xy)
                    if (
                        w > 0
                        and h > 0
                        and w < depth_image.shape[1]
                        and h < depth_image.shape[0]
                    ):
                        depth_crop = depth_image[y : y + h, x : x + w].copy()
                        colour_crop = crop_modality_image(colour_image, x, y, w, h, depth_image.shape[:2])
                        ir_crop = crop_modality_image(ir_image, x, y, w, h, depth_image.shape[:2])
                        detected_crop = build_multimodal_crop(colour_crop, depth_crop, ir_crop)
                        last_detected_crop = detected_crop
                        cv2.rectangle(
                            depth_image_colour,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2,
                        )
                        if detector_type != DetectorType.NONE:
                            feature_visualisation, last_feature_summary = build_feature_visualisation(
                                detected_crop,
                                detector_type,
                                orb,
                                sift,
                                nfeatures,
                            )
                        if classifier is not None and classifier_orb is not None and classifier_class_names is not None:
                            last_prediction_label = predict_object_class(
                                detected_crop,
                                classifier,
                                classifier_orb,
                                classifier_class_names,
                                classifier_nfeatures,
                            )
                            cv2.putText(
                                depth_image_colour,
                                last_prediction_label,
                                (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
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

                cv2.putText(
                    depth_image_colour,
                    "Press S to save crop",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                if last_feature_summary is not None:
                    cv2.putText(
                        depth_image_colour,
                        last_feature_summary,
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                if classifier is not None:
                    status_text = f"Prediction: {last_prediction_label or 'No object detected'}"
                    cv2.putText(
                        depth_image_colour,
                        status_text,
                        (10, 125 if save_every_detection else 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                if save_every_detection:
                    cv2.putText(
                        depth_image_colour,
                        f"Save gap: {min_save_gap}f  Change: {min_crop_change:.1f}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                last_display_frame = depth_image_colour.copy()
                if colour_image is not None:
                    last_colour_frame = colour_image.copy()
                if ir_image is not None:
                    last_ir_frame = ir_image.copy()

                if save_every_detection and last_detected_crop is not None:
                    saved_crop = last_detected_crop.copy()
                    saved_preview = build_saved_screenshot(
                        last_colour_frame,
                        last_display_frame,
                        last_ir_frame,
                    )
                    if should_save_frame(
                        prepare_feature_image(saved_crop),
                        frame_count,
                        last_saved_image,
                        last_saved_frame,
                        min_save_gap,
                        min_crop_change,
                    ):
                        output_path = save_frame_image(saved_crop, saved_output_dir, frame_count)
                        save_label_preview(saved_preview, saved_output_dir, frame_count)
                        last_saved_image = prepare_feature_image(saved_crop)
                        last_saved_frame = frame_count
                        print(f"Saved crop: {output_path}")

                cv2.imshow("Depth", depth_image_colour)
                cv2.imshow("Thresholded Depth", thresholded_depth_8bit)
                if feature_visualisation is not None:
                    cv2.imshow("Features", feature_visualisation)

            # ==============================
            # RGB DISPLAY (OPTIONAL)
            # ==============================

            if rgb and colour_image is not None:
                last_colour_frame = colour_image.copy()
                cv2.imshow("Colour", colour_image)  # note: raw RGB image

            if ir and ir_image is not None:
                last_ir_frame = ir_image.copy()
                cv2.imshow("IR", ir_image)

            # Wait for key press
            key = cv2.waitKey(1)

            if key == ord("q") or key == 27:  # note: press q or ESC to quit
                break
            if key == ord("s") and last_detected_crop is not None:
                saved_crop = last_detected_crop.copy()
                saved_preview = build_saved_screenshot(
                    last_colour_frame,
                    last_display_frame,
                    last_ir_frame,
                )
                output_path = save_frame_image(saved_crop, saved_output_dir, frame_count)
                save_label_preview(saved_preview, saved_output_dir, frame_count)
                last_saved_image = prepare_feature_image(saved_crop)
                last_saved_frame = frame_count
                print(f"Saved crop: {output_path}")

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