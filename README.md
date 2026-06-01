# Paper Reviews

논문을 읽고 요약 및 구현하는 저장소입니다.

## 목록

### 1. DCGAN
- **논문**: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- **내용**: DCGAN 논문 리뷰
- **파일**: `1. UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS_review.pdf`

### 2. AlexNet
- **논문**: ImageNet Classification with Deep Convolutional Neural Networks (Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton)
- **내용**: 논문 리뷰 + PyTorch로 직접 구현
- **파일**:
  - `2. ImageNet Classification with Deep Convolutional Neural Networks.pdf` - 논문 요약
  - `2. Alexnet 구현/alexnet.py` - AlexNet 구현 코드

#### AlexNet 구현 상세
`torchvision.models.alexnet`을 사용하지 않고 PyTorch 기본 함수(`nn.Conv2d`, `nn.Linear` 등)로 직접 구현했습니다.

**논문 반영 사항:**
- **Architecture (Section 3.5)**: 5 Conv + 3 FC + Softmax (총 58M 파라미터)
- **ReLU (Section 3.1)**: tanh/sigmoid 대신 사용하여 학습 속도 향상
- **LRN (Section 3.3)**: Local Response Normalization 직접 구현 (뉴런 간 경쟁)
- **Overlapping Pooling (Section 3.4)**: stride=2, kernel_size=3 (s < z)
- **Dropout (Section 4.2)**: FC1, FC2에 p=0.5 적용
- **Data Augmentation (Section 4.1)**: Random crop (224x224) + Horizontal flip, 10-crop 테스트
- **Training (Section 5)**: SGD (lr=0.01, momentum=0.9, weight_decay=0.0005), batch_size=128
- **Weight Init (Section 5)**: N(0, 0.01), Conv2/4/5/FC bias=1, 나머지 bias=0

**실행 방법:**
```bash
python alexnet.py
```
CIFAR-10 데이터셋을 자동 다운로드하여 학습합니다. GPU 사용 시 Google Colab 권장.
