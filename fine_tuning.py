import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CocoCaptions

# CLIP 모델 및 프로세서 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 데이터셋 준비 (COCO 데이터셋 예제)
dataset = CocoCaptions(
    root="/path/to/coco/images",  # 이미지 디렉토리
    annFile="/path/to/coco/annotations/captions_train2017.json",  # 캡션 파일
    transform=processor.image_processor,
)

# 데이터 로더
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# 파인튜닝 루프
for epoch in range(5):  # Epoch 수 설정
    model.train()
    for batch in dataloader:
        images, captions = batch
        inputs = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도
        labels = torch.arange(len(images)).to(device)  # 일치하는 쌍에 대한 라벨
        
        loss = criterion(logits_per_image, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# 모델 저장
model.save_pretrained("./fine_tuned_clip")
