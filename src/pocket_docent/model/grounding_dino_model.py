from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding


class GroundingDINOModel:

    def __init__(self, model_path: Path | str) -> None:
        self.model_path: Path = (
            model_path if isinstance(model_path, Path) else Path(model_path)
        )

        self.device: torch.device = self.__get_device()
        self.model_id: str = self.model_path.as_posix()
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForZeroShotObjectDetection] = None

    def warmup(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

    def preprocess(
        self,
        image: np.ndarray | Path | str,
        text: str,
    ) -> Tuple[np.ndarray, BatchEncoding]:
        if isinstance(image, Path):
            image = cv2.imread(image.as_posix())
        elif isinstance(image, str):
            image = cv2.imread(image)

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        return image, inputs

    def inference(self, inputs: BatchEncoding) -> Any:
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs

    def postprocess(
        self,
        image: np.ndarray,
        inputs: BatchEncoding,
        outputs: Any,
    ) -> np.ndarray:
        detection = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.shape],
        )[0]

        boxes: torch.Tensor = detection["boxes"]
        boxes = boxes.cpu().numpy().astype(np.int32)

        top1_box = boxes[0]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_image = rgb_image[top1_box[1] : top1_box[3], top1_box[0] : top1_box[2]]

        return cropped_image

    @staticmethod
    def __get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
