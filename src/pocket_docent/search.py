import argparse
from pathlib import Path
from typing import List

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np

from pocket_docent.model.encode_image_model import DINOv2Model
from pocket_docent.model.utils import ModelType


def show_image(
    query_image_path: Path,
    scores: List[float],
    similar_artworks: List[List[str]],
) -> None:
    query_image = cv2.imread(query_image_path.as_posix())
    similar_images = [cv2.imread(a[2]) for a in similar_artworks]

    # Longest width for all images
    max_width = max(img.shape[1] for img in [query_image, *similar_images])

    # Resize images to the longest width while maintaining the aspect ratio
    query_image = cv2.resize(
        query_image,
        (max_width, int(query_image.shape[0] * max_width / query_image.shape[1])),
    )

    similar_images = [
        cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1])))
        for img in similar_images
    ]

    # Batch images 2x3
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, ax in enumerate(axes.ravel()):
        if i == 0:
            ax.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
            ax.set_title("Query Image")
        elif i <= len(similar_images):
            rank = i - 1
            artist = similar_artworks[rank][0]
            serial_number = similar_artworks[rank][1]
            score = scores[rank]
            text = f"{rank + 1} | {artist} {serial_number} | {score:.2f}"
            ax.imshow(cv2.cvtColor(similar_images[i - 1], cv2.COLOR_BGR2RGB))
            ax.set_title(text)

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for similar artworks using a query image."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelType.DINOV2_VITS14.value,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "-q",
        "--query-image",
        type=str,
        required=True,
        help="Path to the query image.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    # current path is src/pocket_docent
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / f"{ModelType(args.model).name.lower()}.onnx"

    model_name = model_path.stem
    index_path = project_root / "db" / f"{model_name}_index.faiss"

    metadata_path = project_root / "db" / "artworks_metadata.npy"
    query_image_path = Path(args.query_image)

    # 검색 예시
    index = faiss.read_index(str(index_path))
    metadata = np.load(str(metadata_path), allow_pickle=True)

    model = DINOv2Model(model_path)
    model.warmup()

    query_image = model.preprocess(query_image_path)
    query_embedding = model.inference(query_image)
    query_embedding = model.postprocess(query_embedding)

    # query_embedding을 2D array로 변환 (1 x d 형태)
    query_embedding = query_embedding.reshape(1, -1).astype("float32")

    # 쿼리 임베딩으로 검색
    scores, indices = index.search(query_embedding, k=5)  # 상위 5개 유사 이미지 검색
    similar_artworks = metadata[indices[0]]  # artwork metadata

    show_image(query_image_path, scores[0], similar_artworks)


if __name__ == "__main__":
    main()
