"""
Helix Tone Multi-Task Model Architecture
=========================================
오디오 백본에서 추출된 임베딩 벡터를 입력받아
5개 이펙터 체인 슬롯의 (1) 이펙터 종류(Classification)와 (2) 노브 파라미터 값(Regression)을
동시에 예측하는 Multi-task Deep Learning Network 구현체입니다.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn

import config
from models.backbone import AudioToneBackbone, PretrainedAudioBackbone


class HelixToneNet(nn.Module):
    """
    Multi-Task Neural Network for Helix Native Preset Prediction
    --------------------------------------------------------------
    Input: Log-Mel-Spectrogram (B, 1, F, T) 또는 Audio Tensor
    Outputs:
      - cls_logits: Dict[slot_name, Tensor (B, num_classes)]
      - reg_preds: Tensor (B, NUM_SLOTS, MAX_KNOBS) -> values strictly in [0, 1] via Sigmoid
    """

    def __init__(
        self,
        embed_dim: int = 512,
        use_pretrained: bool = False,
        pretrained_model_name: str = "AST",
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # 1. 백본 모델 설정 (커스텀 SE-ResNet vs Hugging Face 사전 학습 모델)
        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.backbone = PretrainedAudioBackbone(
                model_name_or_path=pretrained_model_name,
                embed_dim=embed_dim,
                freeze_backbone=freeze_backbone,
            )
        else:
            self.backbone = AudioToneBackbone(in_channels=1, embed_dim=embed_dim)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

        self.slot_names = config.SLOT_NAMES
        self.num_slots = config.NUM_SLOTS
        self.max_knobs = config.MAX_KNOBS_PER_SLOT

        # 1. Classification Heads (슬롯별 이펙터/앰프 모델 분류)
        self.cls_heads = nn.ModuleList()
        for slot in self.slot_names:
            num_classes = len(config.MODEL_CATALOG[slot])
            head = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes),
            )
            self.cls_heads.append(head)

        # 2. Regression Heads (슬롯별 Continuous Knob Value 회귀)
        self.reg_heads = nn.ModuleList()
        for _ in range(self.num_slots):
            head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.max_knobs),
                nn.Sigmoid(),  # 노브 예측값을 [0, 1] 구간으로 엄격히 바운딩
            )
            self.reg_heads.append(head)

    def forward(
        self, mel_spec: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            mel_spec (torch.Tensor): (B, 1, Mel_Bins, Time_Frames)

        Returns:
            cls_logits_list (List[torch.Tensor]): 슬롯별 Logits 텐서 리스트 [(B, C_0), (B, C_1), ...]
            reg_preds (torch.Tensor): (B, NUM_SLOTS, MAX_KNOBS) [0, 1] 범위 회귀 예측값
        """
        # 백본을 통한 공유 임베딩 피처 추출
        feat_embed = self.backbone(mel_spec)  # (B, embed_dim)

        cls_logits_list = []
        for head in self.cls_heads:
            logits = head(feat_embed)
            cls_logits_list.append(logits)

        reg_preds_list = []
        for head in self.reg_heads:
            reg_out = head(feat_embed)  # (B, MAX_KNOBS)
            reg_preds_list.append(reg_out)

        # (B, NUM_SLOTS, MAX_KNOBS) 형태로 스택
        reg_preds = torch.stack(reg_preds_list, dim=1)

        return cls_logits_list, reg_preds
