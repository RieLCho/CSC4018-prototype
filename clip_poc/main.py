import torch
import clip
from PIL import Image

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 로드
image_path = "modern_room.jpg"  # 사용자가 업로드한 이미지 경로
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 가구 태그: COCO, ADE20K, CLIP 데이터셋 기반 확장
furniture_labels = [
    "a bed", "a sofa", "a table", "a chair", "a bookshelf", "a mirror", "a lamp",
    "a nightstand", "a rug", "a plant", "a curtain", "a picture frame", "a desk",
    "a coffee table", "a dresser", "a cabinet", "a wardrobe", "a carpet", "a cushion"
]

# 분위기 태그 확장
style_labels = [
    "modern", "cozy", "luxurious", "minimalist", "rustic", "bohemian", "industrial", "vintage",
    "traditional", "bright", "dark", "neutral", "warm", "wooden", "colorful", "elegant", "chic",
    "artistic", "casual", "functional", "romantic"
]

# 텍스트 토큰화
furniture_text = clip.tokenize(furniture_labels).to(device)
style_text = clip.tokenize(style_labels).to(device)

# 예측 수행
with torch.no_grad():
    image_features = model.encode_image(image)
    furniture_text_features = model.encode_text(furniture_text)
    style_text_features = model.encode_text(style_text)

    # 가구 태그와 분위기 태그에 대한 유사도 계산
    furniture_logits_per_image = image_features @ furniture_text_features.T
    style_logits_per_image = image_features @ style_text_features.T

    # 확률 계산
    furniture_probs = furniture_logits_per_image.softmax(dim=-1).cpu().numpy()
    style_probs = style_logits_per_image.softmax(dim=-1).cpu().numpy()

# 결과 출력
print("Furniture Predictions:")
for label, prob in sorted(zip(furniture_labels, furniture_probs[0]), key=lambda x: x[1], reverse=True):
    print(f"{label}: {prob:.4f}")

print("\nStyle Predictions:")
for label, prob in sorted(zip(style_labels, style_probs[0]), key=lambda x: x[1], reverse=True):
    print(f"{label}: {prob:.4f}")

# 추가: 결과 시각화
def visualize_top_predictions(labels, probs, top_k=5, category="Furniture"):
    print(f"\nTop {top_k} {category} Predictions:")
    for label, prob in sorted(zip(labels, probs[0]), key=lambda x: x[1], reverse=True)[:top_k]:
        print(f"{label}: {prob:.4f}")

# Top-5 결과 출력
visualize_top_predictions(furniture_labels, furniture_probs, top_k=5, category="Furniture")
visualize_top_predictions(style_labels, style_probs, top_k=5, category="Style")
