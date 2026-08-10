"""
Custom PyTorch Dataset & DataLoader Module
===========================================
지정된 폴더에서 (.wav 오디오 파일, .hlx 프리셋 파일) 매칭 쌍을 탐색하여
배치(Batch) 단위 텐서로 제공하는 PyTorch 커스텀 Dataset 모듈입니다.
"""

import os
import glob
from typing import Dict, Any, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils.audio_utils import load_and_preprocess_audio, AudioFeatureExtractor
from utils.hlx_parser import HLXParser


class HelixDataset(Dataset):
    """
    Helix Tone Paired Audio-Preset Dataset
    --------------------------------------
    학습 데이터 폴더 구조:
      data_dir/
        ├── audio_001.wav
        ├── audio_001.hlx (또는 preset_001.hlx)
        ├── audio_002.wav
        └── audio_002.hlx ...
    """

    def __init__(self, data_dir: str, is_train: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.parser = HLXParser()
        self.feature_extractor = AudioFeatureExtractor()

        # 데이터 디렉토리에서 .wav 및 매칭되는 .hlx 파일 목록 수집
        self.samples: List[Tuple[str, str]] = []
        wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))

        for wav_path in wav_files:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            # 동일한 베이스 네임의 .hlx 파일 혹은 'preset_' 접두어가 붙은 .hlx 탐색
            hlx_candidate1 = os.path.join(data_dir, f"{base_name}.hlx")
            hlx_candidate2 = os.path.join(
                data_dir, f"{base_name.replace('audio_', 'preset_')}.hlx"
            )

            if os.path.exists(hlx_candidate1):
                self.samples.append((wav_path, hlx_candidate1))
            elif os.path.exists(hlx_candidate2):
                self.samples.append((wav_path, hlx_candidate2))

        if len(self.samples) == 0:
            print(
                f"[경고] {data_dir} 경로에서 매칭되는 (.wav, .hlx) 데이터 쌍을 찾지 못했습니다."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mel_spec (torch.Tensor): Log-Mel-Spectrogram (1, N_MELS, Time_Frames)
            cls_target (torch.Tensor): (NUM_SLOTS,) LongTensor
            reg_target (torch.Tensor): (NUM_SLOTS, MAX_KNOBS) FloatTensor
            reg_mask (torch.Tensor): (NUM_SLOTS, MAX_KNOBS) FloatTensor
        """
        wav_path, hlx_path = self.samples[idx]

        # 1. 오디오 로드 및 고정 길이 Waveform 변환 (1, Num_Samples)
        waveform = load_and_preprocess_audio(wav_path)

        # 2. Mel-Spectrogram 추출 (1, N_MELS, Time_Frames)
        with torch.no_grad():
            mel_spec = self.feature_extractor(waveform.unsqueeze(0)).squeeze(0)

        # 3. .hlx JSON 라벨 파싱 및 벡터화
        cls_target, reg_target, reg_mask = self.parser.parse_hlx_file(hlx_path)

        return mel_spec, cls_target, reg_target, reg_mask


def create_dataloader(
    data_dir: str,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader 인스턴스 생성 헬퍼 함수 (디스크 파일 기반)
    """
    dataset = HelixDataset(data_dir=data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return dataloader


# 인메모리 데이터셋 및 DataLoader 모듈 가져오기 (디스크 저장 없이 직접 학습 시 사용)
from generate_synthetic_demo_data import (
    SyntheticHelixDataset,
    create_in_memory_dataloader,
)

