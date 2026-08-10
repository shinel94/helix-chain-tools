"""
Helix Tone AI Inference & .hlx Reconstruction Module
=====================================================
학습된 PyTorch 딥러닝 모델에 임의의 순수 기타 오디오 파일(.wav)을 입력하여,
예측된 이펙터/앰프 모델 및 노브 설정값을 재구성(Reconstruction)하여
Helix Native가 정상 인식하는 .hlx (JSON) 파일로 보관/저장하는 추론 스크립트입니다.
"""

import os
import json
import argparse
import torch

import config
from models.helix_net import HelixToneNet
from utils.audio_utils import load_and_preprocess_audio, AudioFeatureExtractor
from utils.hlx_parser import HLXParser


def predict_helix_preset(
    audio_path: str,
    model_checkpoint_path: str,
    output_hlx_path: str = "output_preset.hlx",
    preset_name: str = "AI_Generated_Tone",
):
    """
    단일 .wav 오디오 파일로부터 Helix Native 프리셋(.hlx)을 추론 및 생성합니다.

    Args:
        audio_path (str): 입력 기타 .wav 오디오 경로
        model_checkpoint_path (str): 학습 완료된 PyTorch 체크포인트 (.pth) 경로
        output_hlx_path (str): 복원하여 저장할 .hlx 결과 파일 경로
        preset_name (str): .hlx JSON 메타데이터에 등록될 프리셋 이름
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[추론] 오디오 파일: {audio_path}")
    print(f"[추론] 디바이스: {device}")

    # 1. 오디오 로드 및 Feature 추출 파이프라인
    waveform = load_and_preprocess_audio(audio_path)  # (1, NUM_SAMPLES)
    feature_extractor = AudioFeatureExtractor()

    with torch.no_grad():
        mel_spec = feature_extractor(waveform.unsqueeze(0)).to(device)  # (1, 1, N_MELS, T)

    # 2. PyTorch 모델 구축 및 체크포인트 로드
    model = HelixToneNet().to(device)
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[성공] 모델 체크포인트 로드 완료: {model_checkpoint_path}")
    else:
        print(
            f"[주의] 체크포인트({model_checkpoint_path})를 찾을 수 없어 임의 초기화된 모델로 추론을 진행합니다."
        )

    model.eval()

    # 3. Model Forward Pass 추론
    with torch.no_grad():
        cls_logits_list, reg_preds = model(mel_spec)

    # Classification Argmax 및 Continuous Knob 값 추출
    cls_preds_list = []
    for logits in cls_logits_list:
        pred_class = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        cls_preds_list.append(pred_class)

    cls_preds_array = torch.tensor(cls_preds_list).numpy()
    reg_preds_array = reg_preds.squeeze(0).cpu().numpy()  # (NUM_SLOTS, MAX_KNOBS)

    # 4. JSON .hlx 데이터 구조 복원 (Reconstruction)
    parser = HLXParser()
    reconstructed_hlx_dict = parser.reconstruct_hlx_dict(
        cls_preds=cls_preds_array,
        reg_preds=reg_preds_array,
        preset_name=preset_name,
    )

    # 5. .hlx 파일 저장
    os.makedirs(os.path.dirname(os.path.abspath(output_hlx_path)), exist_ok=True)
    with open(output_hlx_path, "w", encoding="utf-8") as f:
        json.dump(reconstructed_hlx_dict, f, indent=2, ensure_ascii=False)

    print("\n==================================================")
    print(f" [성공] Helix Native 프리셋 복원 및 저장 완료!")
    print(f"  -> 출력 파일: {output_hlx_path}")
    print("==================================================")

    # 추론 결과 요약 출력
    print("\n[추론된 이펙터 체인 요약]")
    for slot_idx, slot_name in enumerate(config.SLOT_NAMES):
        cls_id = cls_preds_array[slot_idx]
        model_name = config.MODEL_CATALOG[slot_name][cls_id]
        print(f" - Slot {slot_idx+1} ({slot_name}): {model_name}")
        if model_name != "None":
            if config.KNOB_SCHEMA.get(model_name):
                knobs = list(config.KNOB_SCHEMA[model_name].keys())[
                    : config.MAX_KNOBS_PER_SLOT
                ]
                knob_str = ", ".join(
                    [
                        f"{k}: {reg_preds_array[slot_idx, k_i]:.2f}"
                        for k_i, k in enumerate(knobs)
                    ]
                )
                print(f"    └ 노브 (Normalized [0,1]): {knob_str}")
            else:
                print(f"    └ (고정 온/오프/프리셋 모델: 노브 파라미터 없음)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helix Tone AI Audio to .hlx Inference Script"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="dataset/test_guitar.wav",
        help="추론할 입력 기타 .wav 오디오 파일 경로",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_helix_net.pth",
        help="학습된 PyTorch 체크포인트 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reconstructed_preset.hlx",
        help="저장할 .hlx 프리셋 파일 경로",
    )
    parser.add_argument(
        "--preset_name",
        type=str,
        default="GarageBand_Tone_AI",
        help="복원할 프리셋 메타데이터 이름",
    )

    args = parser.parse_args()
    predict_helix_preset(
        audio_path=args.audio,
        model_checkpoint_path=args.checkpoint,
        output_hlx_path=args.output,
        preset_name=args.preset_name,
    )
