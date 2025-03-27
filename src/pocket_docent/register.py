from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator, List

import faiss
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

    artworks: List[Artwork] = []
    embeddings = []

    for image_path in generator_image_path(artwork_dir):
        artwork = register_artwork(model, image_path)
        artworks.append(artwork)
        embeddings.append(artwork.embedding)
        cnt += 1

        if cnt % 100 == 0:
            print(f"Progress: {cnt} artworks")

    end_time = perf_counter()

    time_taken = end_time - start_time

    print(f"Total artworks: {cnt}")
    print(f"Artworks time taken: {time_taken:.2f}s\t{time_taken / cnt:.2f}s/artwork")

    start_time = perf_counter()

    # FAISS 인덱스 생성 및 저장
    embeddings_array = np.array(embeddings).astype("float32")
    dimension = embeddings_array.shape[1]  # embedding의 차원, 384
    assert dimension == 384

    # L2 거리를 사용하는 인덱스 생성
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # 인덱스 저장
    index_path = project_root / "db" / "artwork_index.faiss"
    faiss.write_index(index, str(index_path))

    # 메타데이터 저장 (아티스트, 시리얼 번호, 이미지 경로)
    metadata = [
        (artwork.artist, artwork.serial_number, artwork.image_path.as_posix())
        for artwork in artworks
    ]
    metadata_path = project_root / "db" / "artwork_metadata.npy"
    np.save(str(metadata_path), metadata)

    end_time = perf_counter()

    time_taken = end_time - start_time

    print(f"Save index and metadata time taken: {time_taken:.2f}s")


if __name__ == "__main__":
    main()
