"""
AlexNet 구현 - 논문 "ImageNet Classification with Deep Convolutional Neural Networks" 기반
torchvision.models.alexnet 사용하지 않고 PyTorch 내부 함수로 직접 구현
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Section 3.3 - Local Response Normalization (LRN) - pytorch 함수가 존재하지만 직접 구현
# 뉴런 간 경쟁을 만들어 강한 뉴런은 더 강하게, 약한 뉴런은 더 약하게
class LocalResponseNorm(nn.Module):
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        # x: (batch, channels, H, W)
        # 각 채널에 대해 인접 채널들의 활성값 제곱합으로 정규화
        sq = x.pow(2)
        pad = self.size // 2
        # 채널 축으로 패딩 후 평균 풀링으로 인접 채널 합산
        sq_padded = torch.nn.functional.pad(sq, (0, 0, 0, 0, pad, pad))
        # 채널 방향 sliding window 합산
        scale = torch.zeros_like(x)
        for i in range(self.size):
            scale += sq_padded[:, i:i + x.size(1), :, :]
        scale = self.k + self.alpha * scale
        return x / scale.pow(self.beta)


# AlexNet 모델
# Section 3.5 - 5개의 convolution, 3개의 FC, 1000짜리 softmax
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # ---- Feature Extraction (Convolutional Layers) ----

        # Conv1: 11x11x3, stride=4 -> 224x224x3 → 55x55x96
        # 논문에서는 GPU 2개로 48채널씩 나눴지만, 단일 GPU이므로 96채널 통합
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)

        # Conv2: 5x5x48, stride=1, padding=2 -> 27x27x96 → 27x27x256
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)

        # Conv3: 3x3x256, stride=1, padding=1 -> 13x13x256 → 13x13x384
        # 논문: 오직 3번째 레이어만 두 GPU 간 연결
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        # Conv4: 3x3x192, stride=1, padding=1 -> 13x13x384 → 13x13x384
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)

        # Conv5: 3x3x192, stride=1, padding=1 -> 13x13x384 → 13x13x256
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        # Section 3.1 - ReLU (non-saturating, 빠른 학습)
        self.relu = nn.ReLU(inplace=True)

        # Section 3.3 - LRN: Conv1, Conv2 뒤에 적용
        self.lrn = LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        # Section 3.4 - Overlapping Pooling: stride=2, size=3 (s < z)
        # Conv1 LRN 후, Conv2 LRN 후, Conv5 후에 적용
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # ---- Classifier (Fully Connected Layers) ----

        # Section 4.2 - Dropout: FC1, FC2에 적용 (p=0.5)
        self.dropout = nn.Dropout(p=0.5)

        # Conv5 MaxPool 출력: 6x6x256 = 9216
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Section 5 - 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Section 5 - Details of Learning:
        - 가중치: zero-mean Gaussian, std=0.01
        - Bias: Conv2, Conv4, Conv5, FC → 1로 초기화 (양수 ReLU 입력으로 초기 학습 가속)
        - 나머지 Bias: 0으로 초기화
        """
        # Conv1: weight N(0, 0.01), bias 0
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv1.bias, 0)

        # Conv2: weight N(0, 0.01), bias 1
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv2.bias, 1)

        # Conv3: weight N(0, 0.01), bias 0
        nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv3.bias, 0)

        # Conv4: weight N(0, 0.01), bias 1
        nn.init.normal_(self.conv4.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv4.bias, 1)

        # Conv5: weight N(0, 0.01), bias 1
        nn.init.normal_(self.conv5.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv5.bias, 1)

        # FC layers: weight N(0, 0.01), bias 1
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(fc.weight, mean=0, std=0.01)
            nn.init.constant_(fc.bias, 1)

    def forward(self, x):
        """
        입력: x (batch, 3, 224, 224)

        흐름 요약 (논문 Section 3.5):
        Conv1 → ReLU → LRN → MaxPool
        Conv2 → ReLU → LRN → MaxPool
        Conv3 → ReLU
        Conv4 → ReLU
        Conv5 → ReLU → MaxPool
        FC1 → ReLU → Dropout
        FC2 → ReLU → Dropout
        FC3 (1000-way output)
        """
        # Conv1: (B,3,224,224) -> (B,96,55,55) -> LRN -> (B,96,27,27) B는 Batchsize
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.maxpool(x)

        # Conv2: (B,96,27,27) -> (B,256,27,27) -> LRN -> (B,256,13,13)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.maxpool(x)

        # Conv3: (B,256,13,13) -> (B,384,13,13)
        x = self.conv3(x)
        x = self.relu(x)

        # Conv4: (B,384,13,13) -> (B,384,13,13)
        x = self.conv4(x)
        x = self.relu(x)

        # Conv5: (B,384,13,13) -> (B,256,13,13) -> (B,256,6,6)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten: (B, 256*6*6) = (B, 9216)
        x = torch.flatten(x, 1)

        # FC1: 9216 -> 4096 + Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # FC2: 4096 -> 4096 + Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # FC3: 4096 -> num_classes
        x = self.fc3(x)

        return x


# ============================================================
# Section 4.1 - Data Augmentation
# Method1: 256x256 → random 224x224 crop + horizontal flip
# Method2: PCA color augmentation (간소화 버전)
# ============================================================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transforms():
    """
    테스트 시: 5개 패치(4 corners + center) + 좌우반전 = 10개 → 평균 예측
    여기서는 간소화하여 center crop만 사용
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def ten_crop_predict(model, image_tensor_256, device):
    #논문 Section 4.1 - 테스트 시 10-crop 예측
    #256x256 이미지에서 5개 패치(4 corners + center) × 좌우반전 = 10개

    crops = []
    # 4 corners + center (224x224)
    positions = [
        (0, 0), (0, 32), (32, 0), (32, 32),  # corners
        (16, 16),  # center
    ]
    for top, left in positions:
        crop = image_tensor_256[:, top:top+224, left:left+224]
        crops.append(crop)
        crops.append(torch.flip(crop, dims=[2]))  # horizontal flip(좌우반전)

    batch = torch.stack(crops).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    return outputs.mean(dim=0)


# Section 5 - Training (SGD, batch=128, momentum=0.9, weight_decay=0.0005)
def train(model, train_loader, val_loader, device, num_epochs=90):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Section 5: SGD, lr=0.01, momentum=0.9, weight_decay=0.0005
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Section 5: validation error가 개선 안 되면 lr을 10으로 나눔 (총 3번)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4, #4 epoch동안 그대로면 lr 낮추기(임의로 수정 원래는 10)
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {running_loss/(batch_idx+1):.4f} "
                    f"Acc: {100.*correct/total:.2f}%"
                )

        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {running_loss/len(train_loader):.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    print(f"  Validation - Loss: {avg_loss:.4f}, Acc: {100.*correct/total:.2f}%")
    return avg_loss


# CIFAR-10으로 테스트 (ImageNet 대신 가벼운 데이터셋)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 생성 (CIFAR-10은 10 classes)
    model = AlexNet(num_classes=10)

    # 모델 구조 확인
    print("\n=== AlexNet Architecture ===")
    print(model)

    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # shape 테스트
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")

    # CIFAR-10 데이터 로드 (ImageNet 대신)
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    print("\nDownloading CIFAR-10...")
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform,
    )

    # Section 5: batch_size=128
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 학습 시작 (CIFAR-10이므로 epoch 줄임)
    print("\n=== Training Start ===")
    train(model, train_loader, val_loader, device, num_epochs=20)


if __name__ == "__main__":
    main()
