"""
Synthetic Data & Test Environment Generator Module
===================================================
1. config.py의 전면 개편된 Helix Native 이펙터, 앰프, 캐비닛 모델 카탈로그와 KnobSchemaDict를 반영하여
   다채롭고 정교한 합성 기타 오디오(.wav) 및 .hlx 프리셋을 생성합니다.
2. 각 이펙터/앰프/캐비닛 카테고리의 모든 모델들이 무작위 균등 샘플링되어 다양하게 포함되도록 구성되어 있습니다.
3. 디스크 파일 저장 방식(create_synthetic_dataset)과 메모리 상에서 즉시 PyTorch DataLoader로 사용하는 방식(create_in_memory_dataloader)을 모두 지원합니다.
"""

import os
import json
import random
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils.audio_utils import AudioFeatureExtractor


# 슬롯별 활성화 기본 확률 (구조적 프리셋 생성용)
SLOT_ACTIVATION_PROBS = {
    "Amp": 0.90,
    "Cab": 0.88,
    "Distortion": 0.60,
    "Reverb": 0.70,
    "Delay": 0.50,
    "Modulation": 0.45,
    "Dynamics": 0.40,
    "EQ": 0.35,
    "Pitch_Synth": 0.25,
    "Filter": 0.20,
    "Wah": 0.25,
    "Volume_Pan": 0.20,
}


def _apply_sos_filter(audio_signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """SciPy SOS (Second-Order Sections) 필터를 무한 임펄스 응답 안전하게 적용하는 헬퍼 함수"""
    try:
        filtered = signal.sosfilt(sos, audio_signal)
        if np.isnan(filtered).any() or np.isinf(filtered).any():
            return audio_signal
        return filtered
    except Exception:
        return audio_signal


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

    # -------------------------------------------------------------
    # 1. 다채로운 합성 기타 입력 신호 생성 (음고, 피킹 스타일, 픽업 톤)
    # -------------------------------------------------------------
    # E2 (82.41 Hz) ~ E5 (659.25 Hz) 기타 음역대 노트
    base_notes = [82.41, 110.0, 146.83, 164.81, 196.0, 220.0, 246.94, 293.66, 329.63, 392.0, 440.0, 523.25, 659.25]
    fund_freq = random.choice(base_notes) * random.uniform(0.98, 1.02)

    playing_style = random.choice(["sustained", "strum", "rhythmic", "swell"])
    pickup_type = random.choice(["bridge", "neck", "piezo"])

    if pickup_type == "bridge":
        harmonics = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    elif pickup_type == "neck":
        harmonics = [1.0, 0.3, 0.1, 0.05]
    else:  # piezo
        harmonics = [1.0, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]

    # 기본 파형 조합
    raw_signal = np.zeros(num_samples)
    if playing_style == "strum":
        # 코드 파형 (기본음 + 3도/5도 음)
        chord_ratios = [1.0, 1.2599, 1.4983]  # 루트, 장3도, 완전5도
        for chord_r in chord_ratios:
            f = fund_freq * chord_r
            for h_idx, h_amp in enumerate(harmonics, start=1):
                raw_signal += h_amp * np.sin(2 * np.pi * f * h_idx * t)
    else:
        for h_idx, h_amp in enumerate(harmonics, start=1):
            raw_signal += h_amp * np.sin(2 * np.pi * fund_freq * h_idx * t)

    # 엔벨로프 적용
    if playing_style == "sustained":
        decay_rate = random.uniform(1.2, 3.5)
        env = np.exp(-t * decay_rate)
    elif playing_style == "rhythmic":
        # 팜뮤팅 펄스 엔벨로프
        pulse = np.sin(2 * np.pi * random.uniform(2.0, 5.0) * t)
        env = np.clip(pulse, 0.05, 1.0) * np.exp(-t * 2.0)
    elif playing_style == "swell":
        attack_rate = random.uniform(2.0, 6.0)
        env = (1.0 - np.exp(-t * attack_rate)) * np.exp(-t * 1.5)
    else:  # strum
        env = np.exp(-t * random.uniform(1.0, 2.5))

    raw_signal = raw_signal * env

    # -------------------------------------------------------------
    # 2. 이펙터 모델 다양하게 선택 및 라벨 텐서 초기화
    # -------------------------------------------------------------
    cls_target = torch.zeros(config.NUM_SLOTS, dtype=torch.long)
    reg_target = torch.zeros((config.NUM_SLOTS, config.MAX_KNOBS_PER_SLOT), dtype=torch.float32)
    reg_mask = torch.zeros((config.NUM_SLOTS, config.MAX_KNOBS_PER_SLOT), dtype=torch.float32)

    dsp0_blocks = {}
    block_counter = 0
    processed_signal = raw_signal.copy()

    for slot_idx, slot_name in enumerate(config.SLOT_NAMES):
        candidate_models = config.MODEL_CATALOG[slot_name]

        # 활성화 여부 결정 (카테고리별 확률 기반)
        activation_prob = SLOT_ACTIVATION_PROBS.get(slot_name, 0.5)
        is_active = random.random() < activation_prob

        if not is_active or len(candidate_models) <= 1:
            cls_target[slot_idx] = 0  # "None" / Bypass
            continue

        # non-None 모델들 (1번 index부터 끝까지) 중에서 균등하게 무작위 선택!
        # 이를 통해 100여 개 앰프, 50여 개 캐비닛, 40여 개 디스토션 등 모든 모델이 다양하게 활용됩니다.
        cls_id = random.randint(1, len(candidate_models) - 1)
        cls_target[slot_idx] = cls_id
        selected_model = candidate_models[cls_id]

        block_key = f"block{block_counter}"
        block_type = config.BLOCK_TYPE_MAP.get(slot_name, 0)

        block_data = {
            "@type": block_type,
            "@model": selected_model,
            "@enabled": True,
            "@position": slot_idx,
            "@path": 0,
        }

        # -------------------------------------------------------------
        # 3. 노브 값 생성 & 오디오 DSP 신호 변조 시뮬레이션
        # -------------------------------------------------------------
        model_knob_schema = config.KNOB_SCHEMA.get(selected_model, {})

        if model_knob_schema:
            knob_keys = list(model_knob_schema.keys())[: config.MAX_KNOBS_PER_SLOT]

            for knob_idx, knob_name in enumerate(knob_keys):
                knob_info = model_knob_schema[knob_name]
                min_val = knob_info["min"]
                max_val = knob_info["max"]

                norm_val = round(random.uniform(0.05, 0.95), 4)
                real_val = norm_val * (max_val - min_val) + min_val

                if min_val.is_integer() and max_val.is_integer() and (max_val - min_val > 5):
                    real_val = round(real_val)
                else:
                    real_val = round(real_val, 4)

                block_data[knob_name] = real_val
                reg_target[slot_idx, knob_idx] = norm_val
                reg_mask[slot_idx, knob_idx] = 1.0

        # 슬롯 카테고리별 정교한 DSP 음향 변조 파이프라인
        if slot_name == "Dynamics":
            # 컴프레서 / 노이즈게이트 효과
            thresh_norm = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.5
            ratio_norm = reg_target[slot_idx, 1].item() if reg_mask[slot_idx, 1] > 0 else 0.5
            comp_factor = 1.0 / (1.0 + ratio_norm * 3.0)
            processed_signal = np.sign(processed_signal) * (np.abs(processed_signal) ** comp_factor)

        elif slot_name == "Pitch_Synth":
            # 옥타브 / 피치 시프트 하모닉스 추가
            pitch_norm = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.5
            mix_norm = reg_target[slot_idx, 1].item() if reg_mask[slot_idx, 1] > 0 else 0.3
            shift_ratio = 0.5 if pitch_norm < 0.5 else 2.0
            sub_octave = np.sin(2 * np.pi * fund_freq * shift_ratio * t) * env
            processed_signal = (1.0 - mix_norm * 0.4) * processed_signal + (mix_norm * 0.4) * sub_octave

        elif slot_name in ["Filter", "Wah"]:
            # 필터 / 와와 스위핑
            pos_norm = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.5
            center_freq = 300.0 + pos_norm * 3500.0
            low_f = max(50.0, center_freq * 0.7)
            high_f = min(sample_rate * 0.45, center_freq * 1.4)
            sos = signal.butter(2, [low_f, high_f], btype="bandpass", fs=sample_rate, output="sos")
            processed_signal = _apply_sos_filter(processed_signal, sos)

        elif slot_name == "Distortion":
            # 드라이브 / 디스토션 비선형 새츄레이션
            gain_val = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.5
            gain_factor = 1.0 + gain_val * 8.0
            if "Fuzz" in selected_model or "Bitcrusher" in selected_model:
                # Fuzz / Bitcrusher의 하드 클리핑
                processed_signal = np.clip(processed_signal * gain_factor, -0.6, 0.6)
            else:
                # 일반 Overdrive / Distortion의 Soft Tanh Clipping
                processed_signal = np.tanh(processed_signal * gain_factor)

        elif slot_name == "Amp":
            # 앰프 튜브 새츄레이션 + 톤 스택 EQ
            drive_val = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.5
            treble_val = reg_target[slot_idx, 3].item() if reg_mask[slot_idx, 3] > 0 else 0.5
            amp_gain = 1.2 + drive_val * 5.0
            processed_signal = np.tanh(processed_signal * amp_gain)
            # High Treble Cut/Boost Filter
            cut_freq = 2000.0 + treble_val * 4000.0
            sos = signal.butter(2, cut_freq, btype="lowpass", fs=sample_rate, output="sos")
            processed_signal = _apply_sos_filter(processed_signal, sos)

        elif slot_name == "Cab":
            # 캐비닛 주파수 응답 (Low cut ~80Hz, High cut ~6kHz)
            low_cut = 70.0 + (reg_target[slot_idx, 4].item() if reg_mask[slot_idx, 4] > 0 else 0.2) * 80.0
            high_cut = 4000.0 + (reg_target[slot_idx, 5].item() if reg_mask[slot_idx, 5] > 0 else 0.5) * 5000.0
            sos = signal.butter(2, [low_cut, high_cut], btype="bandpass", fs=sample_rate, output="sos")
            processed_signal = _apply_sos_filter(processed_signal, sos)

        elif slot_name == "EQ":
            # EQ Low/High band 컷/부스트
            low_gain = (reg_target[slot_idx, 0].item() - 0.5) * 2.0 if reg_mask[slot_idx, 0] > 0 else 0.0
            if low_gain > 0.1:
                sos = signal.butter(2, 300.0, btype="highpass", fs=sample_rate, output="sos")
                processed_signal = _apply_sos_filter(processed_signal, sos)

        elif slot_name == "Modulation":
            # 트레몰로 / 코러스 / 페이저 LFO 파형 변조
            speed_val = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.5
            depth_val = reg_target[slot_idx, 1].item() if reg_mask[slot_idx, 1] > 0 else 0.5
            lfo_freq = 0.5 + speed_val * 7.0
            lfo = 1.0 - depth_val * 0.4 * (1.0 + np.sin(2 * np.pi * lfo_freq * t))
            processed_signal = processed_signal * lfo

        elif slot_name == "Delay":
            # 에코 디레이 잔향
            time_val = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.3
            mix_val = reg_target[slot_idx, 2].item() if reg_mask[slot_idx, 2] > 0 else 0.3
            delay_samples = int((0.05 + time_val * 0.4) * sample_rate)
            if delay_samples < num_samples:
                delayed = np.zeros_like(processed_signal)
                delayed[delay_samples:] = processed_signal[:-delay_samples]
                processed_signal = processed_signal + delayed * mix_val * 0.5

        elif slot_name == "Reverb":
            # 리버브 감쇠 잔향 믹스
            decay_val = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.3
            mix_val = reg_target[slot_idx, 2].item() if reg_mask[slot_idx, 2] > 0 else 0.3
            reverb_tail = np.random.randn(num_samples) * np.exp(-t * (10.0 - decay_val * 7.0))
            processed_signal = (1.0 - mix_val * 0.3) * processed_signal + (mix_val * 0.3) * reverb_tail

        elif slot_name == "Volume_Pan":
            vol_val = reg_target[slot_idx, 0].item() if reg_mask[slot_idx, 0] > 0 else 0.8
            processed_signal = processed_signal * vol_val

        dsp0_blocks[block_key] = block_data
        block_counter += 1

    # 오디오 신호 Peak Normalization
    max_amp = np.max(np.abs(processed_signal))
    if max_amp > 1e-5:
        processed_signal = (processed_signal / max_amp) * 0.85

    # Helix Native .hlx 표준 구조
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
        signal, hlx_dict, _, _, _ = generate_single_synthetic_audio_and_preset(sample_idx=i, seed=i * 100)

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

        print(f"[메모리 데이터셋] {num_samples}개의 인메모리 데이터 세트를 다양하게 생성 중...")
        for i in range(num_samples):
            signal, _, cls_target, reg_target, reg_mask = generate_single_synthetic_audio_and_preset(
                sample_idx=i, seed=i + 42
            )
            waveform_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, NUM_SAMPLES)
            with torch.no_grad():
                mel_spec = self.feature_extractor(waveform_tensor.unsqueeze(0)).squeeze(0)  # (1, N_MELS, T)

            self.samples.append((mel_spec, cls_target, reg_target, reg_mask))

        print(f"[메모리 데이터셋] {num_samples}개 다양화 샘플 인메모리 생성 완료!")

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
