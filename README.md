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

[DINO v2 small](https://huggingface.co/facebook/dinov2-small) 모델은 Vision Transformer(ViT)로써 self-supervised 방식으로 방대한 이미지 컬렉션에서 사전 학습된 transformer encoder model 입니다. 여기서는 onnx 형태로 변환한 모델 파일을 사용합니다.

- Download to [models](./models) directory
  - [small](https://huggingface.co/onnx-community/dinov2-small): 88.5 MB
  - [base](https://huggingface.co/Xenova/dinov2-base): 346.6 MB

    ```bash
    curl -L https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx -o models/dinov2_vits14.onnx
    curl -L https://huggingface.co/onnx-community/dinov2-base/resolve/main/onnx/model.onnx -o models/dinov2_vitb14.onnx
    ```

## Data

### Sample data

초기 테스트 데이터는 구글 검색을 통해 [생각하는 사람](https://ko.wikipedia.org/wiki/생각하는_사람), [모나리자](https://ko.wikipedia.org/wiki/모나리자), [반가사유상](https://ko.wikipedia.org/wiki/반가사유상), [피카소](https://ko.wikipedia.org/wiki/파블로_피카소) 이미지를 다운받았습니다.

### Best Artworks of All Time

[Kaggle Dataset - Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/) 영향력있는 예술가 50명의 작품을 모았으며, 작품 이미지가 예술가별 숫자로 구분하여 제공되고 있습니다. 예술가에 대한 메타 정보가 있으나 각 작품에 대한 메타 정보는 없습니다.

- Download to [assets/best_artworks_of_all_time](./assets/best_artworks_of_all_time) directory, and then unzip

    ```bash
    curl -L https://www.kaggle.com/api/v1/datasets/download/ikarus777/best-artworks-of-all-time -o assets/best-artworks-of-all-time.zip && \
    unzip assets/best-artworks-of-all-time.zip -d assets/best_artworks_of_all_time
    ```

## How to use

### 등록

[FAISS 의 cosine similarity](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)를 사용하여 인덱싱

```bash
# example
rye run register --model small
```

### 검색

인덱싱 등록이 cosine similarity 로 되어 있으므로 인덱싱에 맞게 검색

```bash
# example
rye run search --model base -q assets/sample_images/mo_0.jpeg
```

## Result

미술품이 정확환 크기로 잘려져 있지 않는한 이미지 검색을 하는데 성능이 많이 떨어집니다. 조명 등의 이슈로 톤이 달라져도 비교적 잘 검색합니다.

| Sample | DINOv2 small | DINOv2 base |
|:------:|:------------:|:-----------:|
| [mo_0.jpeg](./assets/sample_images/mo_0.jpeg) | ![result_dinov2_s_mo_0](./assets/docs/result_dinov2_s_mo_0.jpg) | ![result_dinov2_b_mo_0](./assets/docs/result_dinov2_b_mo_0.jpg) |
| [mo_2.jpeg](./assets/sample_images/mo_2.jpeg) | ![result_dinov2_s_mo_2](./assets/docs/result_dinov2_s_mo_2.jpg) | ![result_dinov2_b_mo_2](./assets/docs/result_dinov2_b_mo_2.jpg) |
| [pi_1.jpeg](./assets/sample_images/pi_1.jpeg) | ![result_dinov2_s_pi_1](./assets/docs/result_dinov2_s_pi_1.jpg) | ![result_dinov2_b_pi_1](./assets/docs/result_dinov2_b_pi_1.jpg) |

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
- [ikarus777/best-artworks-of-all-time](https://creativecommons.org/licenses/by-nc-sa/4.0/) is licensed under the CC BY-NC-SA 4.0
