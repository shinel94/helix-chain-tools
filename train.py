"""
Helix Tone AI Model Training Module
====================================
이 모듈은 오디오 파일과 프리셋 라벨 데이터셋을 학습시키는 메인 훈련 루프입니다.
Multi-task Loss (분류 손실 + 마스킹 적용 회귀 손실) 계산 로직이 구현되어 있습니다.

[손실 함수 수식 정의]
Total Loss = λ_cls * L_cls + λ_reg * L_reg

1. Classification Loss (L_cls):
   L_cls = (1 / N) * ∑_{s=0}^{N-1} CrossEntropy(y_cls^(s), ŷ_cls^(s))

2. Masked Continuous Regression Loss (L_reg):
   L_reg = ∑_{s=0}^{N-1} [ (∑_{k=0}^{K-1} M^{(s,k)} * SmoothL1(y_reg^{(s,k)}, ŷ_reg^{(s,k)})) / (∑_{k=0}^{K-1} M^{(s,k)} + ε) ]
   (비활성화되거나 해당 모델에 존재하지 않는 노브 손실은 Mask M에 의해 0으로 배제됨)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from models.helix_net import HelixToneNet
from dataset import create_dataloader, create_in_memory_dataloader


class MultiTaskHelixLoss(nn.Module):
    """
    Multi-Task Loss Module for Helix Tone Prediction
    ------------------------------------------------
    CrossEntropy (Model Selection Classification) + Masked Smooth L1 (Knob Regression)
    """

    def __init__(
        self, lambda_cls: float = config.LAMBDA_CLS, lambda_reg: float = config.LAMBDA_REG
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        cls_logits_list,
        reg_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
        reg_masks: torch.Tensor,
    ):
        """
        Args:
            cls_logits_list: 슬롯별 Class Logits 리스트 [(B, C_0), (B, C_1), ...]
            reg_preds: (B, NUM_SLOTS, MAX_KNOBS) 예측 노브 값
            cls_targets: (B, NUM_SLOTS) 실제 이펙터/앰프 Class Index
            reg_targets: (B, NUM_SLOTS, MAX_KNOBS) 실제 노브 값 [0, 1]
            reg_masks: (B, NUM_SLOTS, MAX_KNOBS) 유효 노브 마스크 [0.0 or 1.0]
        """
        batch_size = cls_targets.size(0)

        # 1. Classification Loss (각 슬롯별 Cross Entropy의 합)
        total_cls_loss = 0.0
        for slot_idx, logits in enumerate(cls_logits_list):
            slot_cls_target = cls_targets[:, slot_idx]
            total_cls_loss += self.ce_loss(logits, slot_cls_target)

        total_cls_loss = total_cls_loss / config.NUM_SLOTS

        # 2. Masked Regression Loss (유효한 노브 파라미터 위치만 계산)
        raw_reg_loss = self.smooth_l1(reg_preds, reg_targets)  # (B, NUM_SLOTS, MAX_KNOBS)
        masked_reg_loss = raw_reg_loss * reg_masks

        # 마스크 요소의 개수로 정규화
        num_valid_knobs = reg_masks.sum() + 1e-6
        total_reg_loss = masked_reg_loss.sum() / num_valid_knobs

        # 3. 최종 Multi-Task Combined Loss
        total_loss = (self.lambda_cls * total_cls_loss) + (self.lambda_reg * total_reg_loss)

        return total_loss, total_cls_loss.item(), total_reg_loss.item()


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0

    for mel_spec, cls_targets, reg_targets, reg_masks in dataloader:
        mel_spec = mel_spec.to(device)
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device)
        reg_masks = reg_masks.to(device)

        optimizer.zero_grad()

        # Forward Pass
        cls_logits_list, reg_preds = model(mel_spec)

        # Loss 계산
        loss, cls_l, reg_l = criterion(
            cls_logits_list, reg_preds, cls_targets, reg_targets, reg_masks
        )

        # Backward Pass & Optimization
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += cls_l
        running_reg_loss += reg_l

    num_batches = len(dataloader)
    return (
        running_loss / num_batches,
        running_cls_loss / num_batches,
        running_reg_loss / num_batches,
    )


def train_loop(
    data_dir: str = "dataset",
    output_model_path: str = "checkpoints/best_helix_net.pth",
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    use_pretrained: bool = False,
    pretrained_model_name: str = "AST",
    freeze_backbone: bool = False,
    use_in_memory: bool = False,
    num_synthetic_samples: int = 100,
):
    """
    Helix Tone AI 모델 훈련 시작 함수
    -----------------------------------
    Args:
        use_in_memory (bool): True 설정 시, 디스크 파일 읽기 없이 메모리 상에서 즉시 합성 데이터 생성 및 훈련 진행
        num_synthetic_samples (int): use_in_memory=True 일 때 생성할 합성 샘플 개수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[알림] 학습 디바이스: {device}")

    # DataLoader 준비 (인메모리 vs 디스크 파일)
    if use_in_memory:
        print(f"[데이터] 파일 저장 없이 메모리 상에서 {num_synthetic_samples}개 합성 데이터를 직접 생성하여 즉시 학습합니다.")
        dataloader = create_in_memory_dataloader(
            num_samples=num_synthetic_samples, batch_size=batch_size, shuffle=True
        )
    else:
        dataloader = create_dataloader(data_dir=data_dir, batch_size=batch_size, shuffle=True)
        if len(dataloader) == 0:
            print(f"[경고] {data_dir} 경로에 데이터가 없습니다. 인메모리 합성 데이터셋으로 전환합니다.")
            dataloader = create_in_memory_dataloader(
                num_samples=num_synthetic_samples, batch_size=batch_size, shuffle=True
            )

    # 모델, 손실함수, 옵티마이저 초기화
    model = HelixToneNet(
        embed_dim=512,
        use_pretrained=use_pretrained,
        pretrained_model_name=pretrained_model_name,
        freeze_backbone=freeze_backbone,
    ).to(device)

    criterion = MultiTaskHelixLoss().to(device)

    # 동결(Freeze) 옵션 고려: requires_grad=True 인 학습 대상 파라미터만 옵티마이저에 전달
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    best_loss = float("inf")

    print("==================================================")
    print("      Helix Tone AI PyTorch Training Loop        ")
    print("==================================================")

    for epoch in range(1, num_epochs + 1):
        loss, cls_loss, reg_loss = train_epoch(
            model, dataloader, optimizer, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"| Total Loss: {loss:.4f} | Cls Loss: {cls_loss:.4f} | Reg Loss: {reg_loss:.4f}"
        )

        # Best Model Checkpoint 저장
        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                output_model_path,
            )
            print(f" -> Best Checkpoint 저장 완료! ({output_model_path})")

    print(f"\n[완료] 학습이 완료되었습니다. 최저 Loss: {best_loss:.4f}")


if __name__ == "__main__":
    # 독립 실행 시 기본 dataset 폴더에서 훈련
    dataset_directory = "dataset"
    if os.path.exists(dataset_directory):
        train_loop(data_dir=dataset_directory)
    else:
        print(f"[안내] {dataset_directory} 폴더가 없습니다. 먼저 데이터 세트를 준비해 주세요.")
