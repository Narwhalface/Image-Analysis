import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import typer

app = typer.Typer()

WORKSPACE_ROOT = Path(__file__).resolve().parent
LABEL_PREVIEW_DIRNAME = "_label_previews"
REJECTED_LABEL = "Rejected"


def parse_labels(label_text: str | None) -> list[str]:
    if label_text:
        labels = [label.strip() for label in label_text.split(",") if label.strip()]
    else:
        labels = []
    return labels


def get_unlabeled_images(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.glob("*.png") if path.is_file())


def get_preview_image_path(data_dir: Path, image_path: Path) -> Path:
    return data_dir / LABEL_PREVIEW_DIRNAME / image_path.name


def prepare_display_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def draw_overlay(display_image: np.ndarray, image_path: Path, index: int, total: int, labels: list[str], using_preview: bool) -> np.ndarray:
    lines = [
        f"Image {index + 1}/{total}: {image_path.name}",
        "Showing saved preview." if using_preview else "Showing crop only for this frame.",
        "Press a number key for labels 1-9.",
        "Press L to enter a label name or number.",
        "Press N to create a new label.",
        f"Press R to move to {REJECTED_LABEL}.",
        "Press S to skip, Q or Esc to quit.",
    ]
    if labels:
        lines.extend([f"{position}: {label}" for position, label in enumerate(labels[:9], start=1)])
        if len(labels) > 9:
            lines.append(f"... plus {len(labels) - 9} more labels via L")
    else:
        lines.append("No labels yet. Press N to create the first one.")

    sidebar_width = 420
    output = np.zeros((display_image.shape[0], display_image.shape[1] + sidebar_width, 3), dtype=np.uint8)
    output[:, : display_image.shape[1]] = display_image

    y = 25
    for line in lines:
        cv2.putText(
            output,
            line,
            (display_image.shape[1] + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        y += 28
    return output


def prompt_for_label(image_path: Path, label_names: list[str], prompt: str) -> str | None:
    cv2.destroyWindow("Label Training Images")
    try:
        if label_names:
            print("Available labels:")
            for position, label in enumerate(label_names, start=1):
                print(f"  {position}: {label}")
        response = input(f"{prompt} for {image_path.name}: ").strip()
    finally:
        cv2.namedWindow("Label Training Images", cv2.WINDOW_NORMAL)

    if not response:
        print("No label entered.")
        return None

    if response.isdigit():
        selected_index = int(response) - 1
        if 0 <= selected_index < len(label_names):
            return label_names[selected_index]
        print("Invalid label number.")
        return None

    return response


def move_image_to_label(data_dir: Path, image_path: Path, label_name: str, label_names: list[str]) -> None:
    if label_name not in label_names:
        label_names.append(label_name)
        print(f"Added label {len(label_names)}: {label_name}")

    label_dir = data_dir / label_name
    label_dir.mkdir(parents=True, exist_ok=True)
    destination = label_dir / image_path.name
    shutil.move(str(image_path), str(destination))
    print(f"Moved {image_path.name} -> {label_name}")


@app.command()
def main(
    data_dir: Path = typer.Argument(WORKSPACE_ROOT / "training_data", help="Folder containing unlabeled PNG crops"),
    labels: str | None = typer.Option(None, help="Optional comma-separated label names, assigned to keys 1-9"),
) -> int:
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        return 1

    try:
        label_names = parse_labels(labels)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    images = get_unlabeled_images(data_dir)
    if not images:
        print(f"No unlabeled PNG images found in {data_dir}")
        return 0

    if label_names:
        print("Label key mapping:")
        for position, label in enumerate(label_names, start=1):
            print(f"  {position}: {label}")
    else:
        print("No initial labels provided. Press N in the window to create labels as you go.")

    cv2.namedWindow("Label Training Images", cv2.WINDOW_NORMAL)

    index = 0
    while index < len(images):
        image_path = images[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            index += 1
            continue

        preview_path = get_preview_image_path(data_dir, image_path)
        preview_image = None
        if preview_path.exists():
            preview_image = cv2.imread(str(preview_path), cv2.IMREAD_UNCHANGED)

        display_source = preview_image if preview_image is not None else image
        display_image = prepare_display_image(display_source)
        display_image = draw_overlay(display_image, image_path, index, len(images), label_names, preview_image is not None)
        cv2.imshow("Label Training Images", display_image)

        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            index += 1
            continue
        if key in (ord("r"), ord("R")):
            move_image_to_label(data_dir, image_path, REJECTED_LABEL, label_names)
            images.pop(index)
            continue
        if key in (ord("l"), ord("L")):
            label_name = prompt_for_label(image_path, label_names, "Enter label name or number")
            if label_name is None:
                continue

            move_image_to_label(data_dir, image_path, label_name, label_names)
            images.pop(index)
            continue
        if key in (ord("n"), ord("N")):
            label_name = prompt_for_label(image_path, label_names, "Enter new label")
            if label_name is None:
                continue

            move_image_to_label(data_dir, image_path, label_name, label_names)
            images.pop(index)
            continue

        selected_index = key - ord("1")
        if 0 <= selected_index < len(label_names):
            label_dir = data_dir / label_names[selected_index]
            label_dir.mkdir(parents=True, exist_ok=True)
            destination = label_dir / image_path.name
            shutil.move(str(image_path), str(destination))
            print(f"Moved {image_path.name} -> {label_names[selected_index]}")
            images.pop(index)
            continue

        print("Unrecognised key. Use 1-9, L, N, R, S, Q, or Esc.")

    cv2.destroyAllWindows()
    print(f"Remaining unlabeled images: {len(images)}")
    return 0


if __name__ == "__main__":
    sys.exit(app())