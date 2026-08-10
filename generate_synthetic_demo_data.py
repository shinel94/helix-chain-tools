"""
Synthetic Data & Test Environment Generator Module
===================================================
1. config.py의 다양한 이펙터/앰프/캐비닛 모델(Knob가 있는 모델 및 Knob가 없는 고정 모델)을 
   반영하여 다채로운 합성 기타 오디오 및 .hlx 프리셋을 생성합니다.
2. 디스크 파일 저장 방식(create_synthetic_dataset)과 
   메모리 상에서 즉시 PyTorch DataLoader로 사용하는 방식(create_in_memory_dataloader)을 모두 지원합니다.
"""

import os
import json
import random
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils.audio_utils import AudioFeatureExtractor


def generate_single_synthetic_audio_and_preset(sample_idx: int, seed: int = None):
    """
    단일 합성 기타 오디오 파형(Waveform)과 이에 대응하는 Helix .hlx 딕셔너리,
    그리고 학습 라벨 텐서(cls_target, reg_target, reg_mask)를 생성합니다.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sample_rate = config.SAMPLE_RATE
    duration = config.DURATION
    num_samples = config.NUM_SAMPLES
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 1. 다양한 기타 주파수 (E2 ~ E5 노트 범위: 82Hz ~ 660Hz) 및 서스테인 엔벨로프 생성
    base_notes = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0, 659.25]
    fund_freq = random.choice(base_notes) * (1.0 + random.uniform(-0.05, 0.05))
    
    # 기타 현 피킹 이펙트 (Exponential Decay Attack Envelope)
    decay_env = np.exp(-t * random.uniform(1.5, 4.0))
    
    # 기본 하모닉 톤 조합
    raw_signal = (
        0.5 * np.sin(2 * np.pi * fund_freq * t)
        + 0.25 * np.sin(2 * np.pi * fund_freq * 2 * t)
        + 0.15 * np.sin(2 * np.pi * fund_freq * 3 * t)
        + 0.10 * np.sin(2 * np.pi * fund_freq * 4 * t)
    ) * decay_env

    # 2. 이펙터 모델 무작위 선택 및 라벨 생성
    cls_target = torch.zeros(config.NUM_SLOTS, dtype=torch.long)
    reg_target = torch.zeros((config.NUM_SLOTS, config.MAX_KNOBS_PER_SLOT), dtype=torch.float32)
    reg_mask = torch.zeros((config.NUM_SLOTS, config.MAX_KNOBS_PER_SLOT), dtype=torch.float32)

    dsp0_blocks = {}
    block_counter = 0

    processed_signal = raw_signal.copy()

    for slot_idx, slot_name in enumerate(config.SLOT_NAMES):
        candidate_models = config.MODEL_CATALOG[slot_name]
        
        # 모델 무작위 선택 (0번 index인 "None" 포함)
        cls_id = random.randint(0, len(candidate_models) - 1)
        cls_target[slot_idx] = cls_id
        selected_model = candidate_models[cls_id]

        if selected_model == "None":
            continue

        block_key = f"block{block_counter}"
        block_type = config.BLOCK_TYPE_MAP.get(slot_name, 0)

        block_data = {
            "@type": block_type,
            "@model": selected_model,
            "@enabled": True,
            "@position": slot_idx,
            "@path": 0,
        }

        # 노브가 있는 모델인 경우 노브 값 생성 및 오디오 신호 변조 시뮬레이션
        if config.KNOB_SCHEMA.get(selected_model):
            schema_knobs = config.KNOB_SCHEMA[selected_model]
            knob_keys = list(schema_knobs.keys())[: config.MAX_KNOBS_PER_SLOT]

            for knob_idx, knob_name in enumerate(knob_keys):
                knob_info = schema_knobs[knob_name]
                min_val = knob_info["min"]
                max_val = knob_info["max"]

                # 무작위 Normalized 노브 값 [0.0, 1.0]
                norm_val = round(random.uniform(0.05, 0.95), 4)
                real_val = norm_val * (max_val - min_val) + min_val
                
                if min_val.is_integer() and max_val.is_integer() and (max_val - min_val > 5):
                    real_val = round(real_val)
                else:
                    real_val = round(real_val, 4)

                block_data[knob_name] = real_val
                reg_target[slot_idx, knob_idx] = norm_val
                reg_mask[slot_idx, knob_idx] = 1.0

            # 간단한 음향 특성 변조 (오디오 톤 다양화 시뮬레이션)
            if slot_name in ["Distortion", "Amp"]:
                drive_val = reg_target[slot_idx, 0].item()  # 첫 번째 노브(Drive/Gain)
                gain_factor = 1.0 + drive_val * 4.0
                processed_signal = np.tanh(processed_signal * gain_factor)
        else:
            # 노브가 없는 모델 (고정 온/오프/캐비닛 프리셋 모델 등): reg_mask는 0.0 유지
            pass

        dsp0_blocks[block_key] = block_data
        block_counter += 1

    # 오디오 신호 Normalize
    max_amp = np.max(np.abs(processed_signal))
    if max_amp > 1e-5:
        processed_signal = (processed_signal / max_amp) * 0.85

    # .hlx JSON 메타 구조
    preset_name = f"Synthetic_Tone_{sample_idx:04d}"
    hlx_dict = {
        "schema": "helix",
        "version": 67108864,
        "meta": {"name": preset_name, "application": "Helix Native", "app_version": "3.50.0"},
        "data": {
            "device": 2162694,
            "meta": {"name": preset_name, "application": "Helix Native", "app_version": "3.50.0"},
            "tone": {"dsp0": dsp0_blocks, "global": {"@topology0": "A"}},
        },
    }

    return processed_signal, hlx_dict, cls_target, reg_target, reg_mask


def create_synthetic_dataset(output_dir: str = "dataset", num_samples: int = 20):
    """
    다양한 스타일의 합성 기타 오디오(.wav) 및 프리셋(.hlx) 파일을 지정 디렉토리에 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = config.SAMPLE_RATE

    print(f"[데이터 생성] '{output_dir}' 폴더에 {num_samples}개의 다양화된 (.wav, .hlx) 데이터 세트를 생성합니다...")

    for i in range(1, num_samples + 1):
        sample_id = f"{i:03d}"
        signal, hlx_dict, _, _, _ = generate_single_synthetic_audio_and_preset(sample_idx=i, seed=i*100)

        # .wav 저장
        wav_path = os.path.join(output_dir, f"audio_{sample_id}.wav")
        wavfile.write(wav_path, sample_rate, (signal * 32767).astype(np.int16))

        # .hlx 저장
        hlx_path = os.path.join(output_dir, f"audio_{sample_id}.hlx")
        with open(hlx_path, "w", encoding="utf-8") as f:
            json.dump(hlx_dict, f, indent=2, ensure_ascii=False)

    print(f"[성공] 총 {num_samples}개 세트 저장 완료.")


class SyntheticHelixDataset(Dataset):
    """
    파일을 디스크에 저장하지 않고 메모리(RAM) 상에서 즉시 합성 오디오와 라벨 텐서를
    생성하여 학습에 공급하는 PyTorch 인메모리 Dataset
    """

    def __init__(self, num_samples: int = 100):
        super().__init__()
        self.num_samples = num_samples
        self.feature_extractor = AudioFeatureExtractor()
        self.samples = []

        print(f"[메모리 데이터셋] {num_samples}개의 인메모리 데이터 세트를 즉시 생성 중...")
        for i in range(num_samples):
            signal, _, cls_target, reg_target, reg_mask = generate_single_synthetic_audio_and_preset(
                sample_idx=i, seed=i + 42
            )
            # Waveform np.ndarray -> PyTorch Spectrogram Tensor (1, N_MELS, T)
            waveform_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, NUM_SAMPLES)
            with torch.no_grad():
                mel_spec = self.feature_extractor(waveform_tensor.unsqueeze(0)).squeeze(0)  # (1, N_MELS, T)

            self.samples.append((mel_spec, cls_target, reg_target, reg_mask))

        print(f"[메모리 데이터셋] {num_samples}개 샘플 즉시 생성 및 캐싱 완료!")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.samples[idx]


def create_in_memory_dataloader(
    num_samples: int = 100,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """
    디스크 I/O 없이 메모리 상에서 바로 학습할 수 있는 DataLoader 반환 함수
    """
    dataset = SyntheticHelixDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    create_synthetic_dataset(num_samples=10)
