import argparse
from pathlib import Path

import tqdm

from pocket_docent.model.encode_image_model import DINOv2Model, similarity
from pocket_docent.model.utils import ModelType


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image feature comparison using an ONNX model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelType.DINOV2_VITS14.value,
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
    model_path = project_root / "models" / f"{ModelType(args.model).name.lower()}.onnx"

    model = DINOv2Model(model_path)
    model.warmup()

    images = list(Path(args.image_dir).glob("*.jpeg"))
    images.sort()

    embedding_list = []

    for image_path in tqdm.tqdm(images, total=len(images), desc="Processing images..."):
        image = model.preprocess(image_path)
        embedding = model.inference(image)
        output = model.postprocess(embedding)

        embedding_list.append(output)

    num_embeddings = len(embedding_list)
    print(f"Shape of embedding: {embedding_list[0].shape}")

    for probe_id in range(num_embeddings):
        for gallery_id in range(probe_id + 1, num_embeddings):
            score = similarity(
                embedding_list[probe_id],
                embedding_list[gallery_id],
            )

            print(
                f"Similarity of {images[probe_id].name} and "
                f"{images[gallery_id].name}: {score:.6f}"
            )


if __name__ == "__main__":
    main()
