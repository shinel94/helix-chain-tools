"""
Audio Processing Utility Module
================================
이 모듈은 .wav 오디오 파일의 로딩, 채널 통합, 길이 규격화(Padding/Cropping) 및
Mel-Spectrogram 변환을 담당하는 파이토치 기반 오디오 파이프라인 모듈입니다.
"""

import os
from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

import config


class AudioFeatureExtractor(nn.Module):
    """
    Audio Feature Extractor Module
    ------------------------------
    원형 Waveform 오디오 신호를 2D Log-Mel-Spectrogram 텐서로 변환합니다.
    (PyTorch 딥러닝 백본 모델의 입력 데이터 형태로 변환)
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        n_mels: int = config.N_MELS,
        f_min: float = config.F_MIN,
        f_max: float = config.F_MAX,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform shape (Batch, 1, Num_Samples)
        Output: log-mel spectrogram shape (Batch, 1, n_mels, time_frames)
        """
        # Mel-Spectrogram 계산
        mel_spec = self.mel_spectrogram(waveform)
        # 데시벨 (Log scale) 변환
        log_mel_spec = self.amplitude_to_db(mel_spec)
        # 표준화 (Instance Normalization / Standard Scaling)
        mean = log_mel_spec.mean(dim=(-2, -1), keepdim=True)
        std = log_mel_spec.std(dim=(-2, -1), keepdim=True) + 1e-6
        normalized_spec = (log_mel_spec - mean) / std

        return normalized_spec


def load_and_preprocess_audio(
    file_path: str,
    target_sample_rate: int = config.SAMPLE_RATE,
    target_num_samples: int = config.NUM_SAMPLES,
) -> torch.Tensor:
    """
    단일 오디오 파일(.wav)을 읽어 정제된 Mono Waveform 텐서로 변환합니다.

    Args:
        file_path (str): .wav 오디오 파일 경로
        target_sample_rate (int): 목표 샘플링 레이트
        target_num_samples (int): 목표 오디오 샘플 길이 (고정 길이)

    Returns:
        torch.Tensor: (1, target_num_samples) 형태의 Mono Waveform 텐서
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")

    # torchaudio로 파이프라인 로드
    waveform, sr = torchaudio.load(file_path)

    # 1. 멀티채널(Stereo)인 경우 Mono(단일 채널)로 평균화
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 2. 샘플링 레이트 변정 (Resampling)
    if sr != target_sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # 3. 고정 길이로 Padding 또는 Crop
    num_samples = waveform.shape[1]
    if num_samples < target_num_samples:
        # 길이가 모자란 경우 Zero-Padding
        padding_length = target_num_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))
    elif num_samples > target_num_samples:
        # 길이가 길면 앞쪽 중심 구간 Crop
        waveform = waveform[:, :target_num_samples]

    return waveform
