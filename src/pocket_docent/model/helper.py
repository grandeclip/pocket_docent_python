from pathlib import Path

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def download_grounding_dino_tiny(target_dir: Path):
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    processor.save_pretrained(target_dir.as_posix())
    model.save_pretrained(target_dir.as_posix())


if __name__ == "__main__":
    target_dir = (
        Path(__file__).parent.parent.parent.parent / "models" / "grounding_dino_tiny"
    )
    download_grounding_dino_tiny(target_dir)
