import torch
import clip
from PIL import Image

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 로드 및 전처리
image = preprocess(Image.open("cozy_room.jpg")).unsqueeze(0).to(device)

# 라벨 준비
furniture_labels = ["a bed", "a sofa", "a table", "a chair", "a bookshelf"]
style_labels = ["modern", "cozy", "luxurious", "minimalist", "rustic"]

furniture_text = clip.tokenize(furniture_labels).to(device)
style_text = clip.tokenize(style_labels).to(device)

with torch.no_grad():
    # 이미지와 텍스트 특징 추출
    image_features = model.encode_image(image)
    furniture_text_features = model.encode_text(furniture_text)
    style_text_features = model.encode_text(style_text)

    # 유사도 계산
    logits_furniture = (image_features @ furniture_text_features.T).softmax(dim=-1).cpu().numpy()
    logits_style = (image_features @ style_text_features.T).softmax(dim=-1).cpu().numpy()

# 결과 출력
print("Furniture Predictions:")
for label, prob in zip(furniture_labels, logits_furniture[0]):
    print(f"{label}: {prob:.4f}")

print("\nStyle Predictions:")
for label, prob in zip(style_labels, logits_style[0]):
    print(f"{label}: {prob:.4f}")
