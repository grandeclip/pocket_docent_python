from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np

from pocket_docent.model.encode_image_model import DINOv2Model


@dataclass
class Artwork:
    artist: str
    serial_number: int
    image_path: Path
    embedding: np.ndarray


def register_artwork(model: DINOv2Model, image_path: Path) -> Artwork:
    artist = image_path.parent.name
    serial_number = int(image_path.stem.split("_")[-1])

    image = model.preprocess(image_path)
    embedding = model.inference(image)
    output = model.postprocess(embedding)

    return Artwork(artist, serial_number, image_path, output)


def generator_image_path(artwork_dir: Path) -> Iterator[Path]:
    for artist_dir in artwork_dir.iterdir():
        for image_path in artist_dir.glob("*.jpg"):
            yield image_path


def main() -> None:
    # current path is src/pocket_docent
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "dinov2_vits14.onnx"
    artwork_dir = (
        project_root / "assets" / "best_artworks_of_all_time" / "images" / "images"
    )

    model = DINOv2Model(model_path)
    model.warmup()
    cnt = 0

    start_time = perf_counter()

    for image_path in generator_image_path(artwork_dir):
        artwork = register_artwork(model, image_path)
        print(artwork.artist, artwork.serial_number)
        cnt += 1

    end_time = perf_counter()

    time_taken = end_time - start_time

    print(f"Total artworks: {cnt}")
    print(f"Time taken: {time_taken:.2f}s\t{cnt / time_taken:.2f} artwork/s")


if __name__ == "__main__":
    main()
