# Pocket Docent

이미지로 미술품을 검색

## Environment

### Install `rye`

- [Rye 설치 가이드](https://rye.astral.sh/guide/installation/) 참고

    ```bash
    curl -sSf https://rye.astral.sh/get | bash
    ```

- 설정된 Python version 및 필요한 dependencies 설치

    ```bash
    rye sync
    ```

### Download models

#### DINO v2

[DINO v2 small](https://huggingface.co/facebook/dinov2-small) 모델은 Vision Transformer(ViT)로써 self-supervised 방식으로 방대한 이미지 컬렉션에서 사전 학습된 transformer encoder model 입니다. 여기서는 [onnx](https://huggingface.co/sefaburak/dinov2-small-onnx) 형태로 변환한 모델 파일을 사용합니다.

- Download to [models](./models) directory

    ```bash
    curl -L https://huggingface.co/sefaburak/dinov2-small-onnx/resolve/main/dinov2_vits14.onnx -o models/dinov2_vits14.onnx
    ```

## Data

초기 테스트 데이터는 구글 검색을 통해 [생각하는 사람](https://ko.wikipedia.org/wiki/%EC%83%9D%EA%B0%81%ED%95%98%EB%8A%94_%EC%82%AC%EB%9E%8C), [모나리자](https://ko.wikipedia.org/wiki/%EB%AA%A8%EB%82%98%EB%A6%AC%EC%9E%90), [반가사유상](https://ko.wikipedia.org/wiki/%EB%B0%98%EA%B0%80%EC%82%AC%EC%9C%A0%EC%83%81) 이미지를 다운받았습니다.

## Citations

```bibtex
misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Licenses

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) is licensed under the Apache License 2.0
- [sefaburakokcu/dinov2_onnx](https://github.com/sefaburakokcu/dinov2_onnx/blob/main/LICENSE) is licensed under the Apache License 2.0
