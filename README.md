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

### Best Artworks of All Time

[Kaggle Dataset - Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/) 영향력있는 예술가 50명의 작품을 모았으며, 작품 이미지가 예술가별 숫자로 구분하여 제공되고 있습니다. 예술가에 대한 메타 정보가 있으나 각 작품에 대한 메타 정보는 없습니다.

## How to use

초기 테스트 데이터가 [assets/sample_images](./assets/sample_images/) 에 있다고 가정하면:

```bash
rye run service --image-dir assets/sample_images
```

<details>
<summary>결과</summary>

```text
Similarity of bs_0.jpeg and bs_1.jpeg: 0.765640
Similarity of bs_0.jpeg and bs_2.jpeg: 0.675734
Similarity of bs_0.jpeg and bs_3.jpeg: 0.772154
Similarity of bs_0.jpeg and mo_0.jpeg: 0.194971
Similarity of bs_0.jpeg and mo_1.jpeg: 0.137492
Similarity of bs_0.jpeg and mo_2.jpeg: 0.256037
Similarity of bs_0.jpeg and mo_3.jpeg: 0.137595
Similarity of bs_0.jpeg and mo_4.jpeg: 0.162791
Similarity of bs_0.jpeg and tp_0.jpeg: 0.417260
Similarity of bs_0.jpeg and tp_1.jpeg: 0.258903
Similarity of bs_0.jpeg and tp_2.jpeg: 0.066621
Similarity of bs_0.jpeg and tp_3.jpeg: 0.243741
Similarity of bs_0.jpeg and tp_4.jpeg: 0.238415
Similarity of bs_1.jpeg and bs_2.jpeg: 0.753469
Similarity of bs_1.jpeg and bs_3.jpeg: 0.909024
Similarity of bs_1.jpeg and mo_0.jpeg: 0.236961
Similarity of bs_1.jpeg and mo_1.jpeg: 0.167419
Similarity of bs_1.jpeg and mo_2.jpeg: 0.288664
Similarity of bs_1.jpeg and mo_3.jpeg: 0.157980
Similarity of bs_1.jpeg and mo_4.jpeg: 0.164963
Similarity of bs_1.jpeg and tp_0.jpeg: 0.530288
Similarity of bs_1.jpeg and tp_1.jpeg: 0.341173
Similarity of bs_1.jpeg and tp_2.jpeg: 0.138752
Similarity of bs_1.jpeg and tp_3.jpeg: 0.356877
Similarity of bs_1.jpeg and tp_4.jpeg: 0.324237
Similarity of bs_2.jpeg and bs_3.jpeg: 0.800658
Similarity of bs_2.jpeg and mo_0.jpeg: 0.216880
Similarity of bs_2.jpeg and mo_1.jpeg: 0.172101
Similarity of bs_2.jpeg and mo_2.jpeg: 0.253723
Similarity of bs_2.jpeg and mo_3.jpeg: 0.133605
Similarity of bs_2.jpeg and mo_4.jpeg: 0.156046
Similarity of bs_2.jpeg and tp_0.jpeg: 0.540113
Similarity of bs_2.jpeg and tp_1.jpeg: 0.280114
Similarity of bs_2.jpeg and tp_2.jpeg: 0.180296
Similarity of bs_2.jpeg and tp_3.jpeg: 0.385650
Similarity of bs_2.jpeg and tp_4.jpeg: 0.370535
Similarity of bs_3.jpeg and mo_0.jpeg: 0.208123
Similarity of bs_3.jpeg and mo_1.jpeg: 0.187374
Similarity of bs_3.jpeg and mo_2.jpeg: 0.250088
Similarity of bs_3.jpeg and mo_3.jpeg: 0.137194
Similarity of bs_3.jpeg and mo_4.jpeg: 0.165795
Similarity of bs_3.jpeg and tp_0.jpeg: 0.553816
Similarity of bs_3.jpeg and tp_1.jpeg: 0.343698
Similarity of bs_3.jpeg and tp_2.jpeg: 0.164702
Similarity of bs_3.jpeg and tp_3.jpeg: 0.375859
Similarity of bs_3.jpeg and tp_4.jpeg: 0.370548
Similarity of mo_0.jpeg and mo_1.jpeg: 0.516511
Similarity of mo_0.jpeg and mo_2.jpeg: 0.689861
Similarity of mo_0.jpeg and mo_3.jpeg: 0.357155
Similarity of mo_0.jpeg and mo_4.jpeg: 0.481262
Similarity of mo_0.jpeg and tp_0.jpeg: 0.151120
Similarity of mo_0.jpeg and tp_1.jpeg: 0.089457
Similarity of mo_0.jpeg and tp_2.jpeg: 0.052222
Similarity of mo_0.jpeg and tp_3.jpeg: 0.054163
Similarity of mo_0.jpeg and tp_4.jpeg: 0.118406
Similarity of mo_1.jpeg and mo_2.jpeg: 0.658458
Similarity of mo_1.jpeg and mo_3.jpeg: 0.768398
Similarity of mo_1.jpeg and mo_4.jpeg: 0.770338
Similarity of mo_1.jpeg and tp_0.jpeg: 0.173380
Similarity of mo_1.jpeg and tp_1.jpeg: 0.076879
Similarity of mo_1.jpeg and tp_2.jpeg: 0.090563
Similarity of mo_1.jpeg and tp_3.jpeg: 0.078829
Similarity of mo_1.jpeg and tp_4.jpeg: 0.105926
Similarity of mo_2.jpeg and mo_3.jpeg: 0.611093
Similarity of mo_2.jpeg and mo_4.jpeg: 0.663631
Similarity of mo_2.jpeg and tp_0.jpeg: 0.155075
Similarity of mo_2.jpeg and tp_1.jpeg: 0.059101
Similarity of mo_2.jpeg and tp_2.jpeg: 0.068464
Similarity of mo_2.jpeg and tp_3.jpeg: 0.060886
Similarity of mo_2.jpeg and tp_4.jpeg: 0.114672
Similarity of mo_3.jpeg and mo_4.jpeg: 0.641933
Similarity of mo_3.jpeg and tp_0.jpeg: 0.134715
Similarity of mo_3.jpeg and tp_1.jpeg: 0.058136
Similarity of mo_3.jpeg and tp_2.jpeg: 0.011659
Similarity of mo_3.jpeg and tp_3.jpeg: 0.043614
Similarity of mo_3.jpeg and tp_4.jpeg: 0.057288
Similarity of mo_4.jpeg and tp_0.jpeg: 0.099501
Similarity of mo_4.jpeg and tp_1.jpeg: 0.092150
Similarity of mo_4.jpeg and tp_2.jpeg: 0.061716
Similarity of mo_4.jpeg and tp_3.jpeg: 0.031618
Similarity of mo_4.jpeg and tp_4.jpeg: 0.063088
Similarity of tp_0.jpeg and tp_1.jpeg: 0.491725
Similarity of tp_0.jpeg and tp_2.jpeg: 0.362347
Similarity of tp_0.jpeg and tp_3.jpeg: 0.689296
Similarity of tp_0.jpeg and tp_4.jpeg: 0.666327
Similarity of tp_1.jpeg and tp_2.jpeg: 0.255971
Similarity of tp_1.jpeg and tp_3.jpeg: 0.584015
Similarity of tp_1.jpeg and tp_4.jpeg: 0.430376
Similarity of tp_2.jpeg and tp_3.jpeg: 0.344742
Similarity of tp_2.jpeg and tp_4.jpeg: 0.312794
Similarity of tp_3.jpeg and tp_4.jpeg: 0.623825
```

</details>

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
- [ikarus777/best-artworks-of-all-time](https://creativecommons.org/licenses/by-nc-sa/4.0/) is licensed under the CC BY-NC-SA 4.0
