import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import tqdm


def load_onnx_model(
    model_path: Path,
    providers: Optional[List[str]] = None,
) -> ort.InferenceSession:
    ort_session = ort.InferenceSession(model_path.as_posix(), providers=providers)

    return ort_session


def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
) -> np.ndarray:
    """
    Preprocess the input image.

    Args:
        image (numpy.ndarray): Input image.
        input_size (tuple): Target input size. Defaults to (224, 224).
        mean (tuple): Mean values for normalization. Defaults to (123.675, 116.28, 103.53).
        std (tuple): Standard deviation values for normalization. Defaults to (58.395, 57.12, 57.375).

    Returns:
        numpy.ndarray: Preprocessed input data.
    """
    input_data = cv2.resize(image, input_size)
    input_data = (input_data - np.array(mean)) / np.array(std)
    input_data = input_data.astype(np.float32)
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    return input_data


def compare_features(embedding_1: np.ndarray, embedding_2: np.ndarray) -> float:
    embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
    embedding_2 = embedding_2 / np.linalg.norm(embedding_2)

    return (embedding_1 @ embedding_2.T)[0][0]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image feature comparison using an ONNX model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vits14.onnx",
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the folder containing images.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    # current path is src/pocket_docent
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / args.model

    ort_session = load_onnx_model(model_path, providers=["CPUExecutionProvider"])

    images = list(Path(args.image_dir).glob("*.jpeg"))
    images.sort()

    embedding_list = []

    for image_path in tqdm.tqdm(images, total=len(images), desc="Processing images..."):
        image = cv2.imread(image_path.as_posix())
        input_data = preprocess_image(image)

        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outputs = ort_session.run(None, ort_inputs)[0]

        embedding_list.append(ort_outputs)

    num_embeddings = len(embedding_list)

    for probe_id in range(num_embeddings):
        for gallery_id in range(probe_id + 1, num_embeddings):
            similarity = compare_features(
                embedding_list[probe_id],
                embedding_list[gallery_id],
            )

            print(
                f"Similarity of {images[probe_id].name} and "
                f"{images[gallery_id].name}: {similarity:.6f}"
            )


if __name__ == "__main__":
    main()
