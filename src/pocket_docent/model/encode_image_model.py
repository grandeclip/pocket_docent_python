from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class DINOv2Model:

    def __init__(self, model_path: Path | str) -> None:
        self.model_path: str = (
            model_path.as_posix() if isinstance(model_path, Path) else model_path
        )
        self.session: Optional[ort.InferenceSession] = None

        # Configuration for preprocessing
        self.input_size: Tuple[int, int] = (224, 224)
        self.mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
        self.std: Tuple[float, float, float] = (58.395, 57.12, 57.375)

    def warmup(self) -> None:
        providers: List[str] = self.__set_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)

    def preprocess(self, image: np.ndarray | Path | str) -> np.ndarray:
        if self.session is None:
            raise ValueError("Model is not initialized")

        if isinstance(image, Path):
            image = cv2.imread(image.as_posix())
        elif isinstance(image, str):
            image = cv2.imread(image)

        image_data = cv2.resize(image, self.input_size)  # (224, 224, 3)
        image_data = (image_data - np.array(self.mean)) / np.array(self.std)
        image_data = image_data.astype(np.float32)
        image_data = np.transpose(image_data, (2, 0, 1))  # (3, 224, 224)
        image_data = np.expand_dims(image_data, axis=0)  # (1, 3, 224, 224)

        return image_data

    def inference(self, image_data: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise ValueError("Model is not initialized")

        inputs = {self.session.get_inputs()[0].name: image_data}
        embedding = self.session.run(None, inputs)[0]

        return embedding

    def postprocess(self, embedding: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise ValueError("Model is not initialized")

        return embedding.flatten()

    def __set_providers(self) -> List[str]:
        providers = ort.get_available_providers()

        remove_providers: List[str] = [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        ]

        result = [p for p in providers if p not in remove_providers]

        return result


def similarity(embedding_1: np.ndarray, embedding_2: np.ndarray) -> float:
    embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
    embedding_2 = embedding_2 / np.linalg.norm(embedding_2)

    return embedding_1 @ embedding_2.T
