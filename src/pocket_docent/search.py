from pathlib import Path

import faiss
import numpy as np

from pocket_docent.model.encode_image_model import DINOv2Model


def main() -> None:
    # current path is src/pocket_docent
    project_root = Path(__file__).parent.parent.parent
    index_path = project_root / "db" / "artwork_index.faiss"
    metadata_path = project_root / "db" / "artwork_metadata.npy"
    query_image_path = project_root / "assets" / "sample_images" / "pi_1.jpeg"

    # 검색 예시
    index = faiss.read_index(str(index_path))
    metadata = np.load(str(metadata_path), allow_pickle=True)

    model = DINOv2Model(project_root / "models" / "dinov2_vits14.onnx")
    model.warmup()

    query_image = model.preprocess(query_image_path)
    query_embedding = model.inference(query_image)
    query_embedding = model.postprocess(query_embedding)

    # query_embedding을 2D array로 변환 (1 x d 형태)
    query_embedding = query_embedding.reshape(1, -1).astype("float32")

    # 쿼리 임베딩으로 검색
    scores, indices = index.search(query_embedding, k=5)  # 상위 5개 유사 이미지 검색
    similar_artworks = metadata[indices[0]]  # 검색된 아트워크의 메타데이터

    print(f"Query image: {query_image_path}")
    print(
        f"Similar artworks:\n{'\n'.join([f'{s:.2f}\t{a[0]}\t{a[1]}' for s, a in zip(scores[0], similar_artworks)])}"
    )


if __name__ == "__main__":
    main()
