"""
Audio Deep Learning Backbone Module
====================================
기타 톤의 미세한 음향적 특성(디스토션 배음 비선형성, 앰프 EQ 주파수 특성,
캐비닛 공간감, 시공간적 잔향 tail 등)을 다각도로 추출하기 위한
SE-ResNet 기반 파이토치 커스텀 오디오 백본 네트워크입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


PRETRAINED_MODEL_MAP = {
    "AST": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "Wav2Vec2": "facebook/wav2vec2-base-960h",
    "HuBERT": "facebook/hubert-base-ls960",
}


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    ---------------------------------
    채널 간 상호작용(주파수 밴드 및 특성 맵 중요도)을 적응적으로 재가중(Re-weighting)하여
    특정 앰프/이펙터 주파수 톤 강조점을 효과적으로 포착합니다.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        b, c, _, _ = x.size()
        squeeze = x.mean(dim=(-2, -1))  # Global Average Pooling (B, C)
        excitation = F.relu(self.fc1(squeeze), inplace=True)
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        return x * excitation


class ConvBlock(nn.Module):
    """
    Residual Convolutional Block with SE Attention
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeAndExcitation(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.gelu(out)
        return out


class AudioToneBackbone(nn.Module):
    """
    Audio Tone Feature Extractor Backbone
    --------------------------------------
    Log-Mel-Spectrogram (B, 1, F, T) -> Feature Embedding Vector (B, embed_dim)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 512):
        super().__init__()

        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=7, stride=(2, 2), padding=3, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # SE-ResNet Blocks Layer
        self.layer1 = ConvBlock(32, 64, stride=1)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.layer4 = ConvBlock(256, 512, stride=2)

        # Multi-scale Pooling (AvgPool + MaxPool Fusion)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Final Embedding Projection Layer
        self.fc_proj = nn.Sequential(
            nn.Linear(512 * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 1, Mel_Bins, Time_Frames)
        Output: (B, embed_dim)
        """
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global Avg + Max Pooling 융합 (공간/시간상 강한 서스테인/어택과 평균 음향 특성 보존)
        avg_feat = self.avg_pool(out).flatten(1)
        max_feat = self.max_pool(out).flatten(1)
        concat_feat = torch.cat([avg_feat, max_feat], dim=1)

        embedding = self.fc_proj(concat_feat)
        return embedding


class PretrainedAudioBackbone(nn.Module):
    """
    Hugging Face 사전 학습 오디오 모델 백본 (AST, Wav2Vec 2.0, HuBERT 등)
    -------------------------------------------------------------------
    사전 학습된 백본에서 추출된 Feature Map을 어댑터(Adapter) 레이어를 통해
    기존 HelixToneNet 헤드의 입력 차원(embed_dim)으로 투영(Projection)합니다.
    """

    def __init__(
        self,
        model_name_or_path: str = "AST",
        embed_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers 라이브러리가 설치되어 있지 않습니다. "
                "'pip install transformers' 명령어로 설치 후 사용해 주세요."
            )

        # 모델 별칭 또는 HF 레포지토리 경로 해소
        self.hf_model_id = PRETRAINED_MODEL_MAP.get(
            model_name_or_path, model_name_or_path
        )
        print(f"[백본 설정] HuggingFace 사전 학습 모델 로딩: {self.hf_model_id}")

        # Hugging Face 인코더 모델 로드
        self.encoder = AutoModel.from_pretrained(self.hf_model_id)

        # 1. 백본 파라미터 동결(Freeze) 여부 설정
        if freeze_backbone:
            print("[백본 설정] 백본 파라미터가 동결(Freeze)되었습니다. (requires_grad = False)")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("[백본 설정] 백본 미세 조정(Fine-tuning) 모드로 설정되었습니다. (requires_grad = True)")

        # 2. 백본 출력 Hidden Dimension 자동 감지
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "dim", 768)

        # 3. 이펙터 예측 헤드의 입력 차원(embed_dim)에 맞추는 어댑터(Adapter) 레이어
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Log-Mel-Spectrogram (B, 1, F, T) 또는 Raw Waveform (B, T)
        Returns:
            torch.Tensor: (B, embed_dim) 사전에 지정한 임베딩 차원의 피처
        """
        # HF 모델 입력 포맷 조정 (Spectrogram: (B, 1, F, T) -> (B, T, F))
        if "ast" in self.hf_model_id.lower() or "ast" in str(type(self.encoder)).lower():
            # AST (Audio Spectrogram Transformer)는 128 Mel Bins x 1024 Time Frames 해상도 및
            # 1214개 고정 위치 임베딩(position_embeddings)에 맞춰 설계되어 있습니다.
            # 입력 Spectrogram을 AST 기대 규격(128, 1024)으로 보간(Interpolate)합니다.
            if x.dim() == 4:
                x = F.interpolate(x, size=(128, 1024), mode="bilinear", align_corners=False)
                x_in = x.squeeze(1).transpose(1, 2)  # (B, 1024, 128)
            elif x.dim() == 3:
                x_4d = x.unsqueeze(1)
                x = F.interpolate(x_4d, size=(128, 1024), mode="bilinear", align_corners=False)
                x_in = x.squeeze(1).transpose(1, 2)  # (B, 1024, 128)
            else:
                x_in = x
        else:
            if x.dim() == 4:
                x_in = x.squeeze(1).transpose(1, 2)
            elif x.dim() == 3:
                x_in = x.transpose(1, 2)
            else:
                x_in = x

        # HF 인코더 추론 (AST / Wav2Vec2 / HuBERT 입력 키워드 대응)
        try:
            outputs = self.encoder(input_values=x_in)
        except (TypeError, ValueError):
            outputs = self.encoder(input_features=x_in)

        # Feature 추출 (pooler_output 또는 last_hidden_state의 Global Average Pooling)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state.mean(dim=1)
        else:
            feat = outputs[0].mean(dim=1)

        # Adapter를 통해 기존 헤드의 embed_dim으로 투영
        embedding = self.adapter(feat)
        return embedding

