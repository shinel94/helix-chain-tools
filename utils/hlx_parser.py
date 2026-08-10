"""
Helix Native (.hlx) Parser & Reconstructor Utility Module
==========================================================
이 모듈은 .hlx (JSON) 파일 파싱, 다중 태스크 학습용 라벨 벡터 변환(Vectorization),
및 모델 예측 텐서를 정상적인 Line 6 Helix Native .hlx 파일 구조로 재구성(Reconstruction)하는 역 파싱 기능을 구현합니다.
"""

import json
import os
from typing import Dict, Any, Tuple
import torch
import numpy as np

import config


class HLXParser:
    """
    Helix Native (.hlx) JSON 파서 및 벡터 변환기
    """

    def __init__(self):
        self.model_catalog = config.MODEL_CATALOG
        self.knob_schema = config.KNOB_SCHEMA
        self.slot_names = config.SLOT_NAMES
        self.num_slots = config.NUM_SLOTS
        self.max_knobs = config.MAX_KNOBS_PER_SLOT

    def parse_hlx_file(
        self, file_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        .hlx (JSON) 파일을 읽어 Multi-task 학습용 라벨 텐서로 변환합니다.

        Returns:
            cls_target: (NUM_SLOTS,) - 각 슬롯별 이펙터/앰프 모델 Class Index 텐서 (LongTensor)
            reg_target: (NUM_SLOTS, MAX_KNOBS) - [0, 1] 범위로 규격화된 노브 파라미터 값 (FloatTensor)
            reg_mask: (NUM_SLOTS, MAX_KNOBS) - 마스크 텐서 (유효 노브인 경우 1.0, 비활성은 0.0) (FloatTensor)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HLX 프리셋 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            preset_data = json.load(f)

        cls_target = torch.zeros(self.num_slots, dtype=torch.long)
        reg_target = torch.zeros((self.num_slots, self.max_knobs), dtype=torch.float32)
        reg_mask = torch.zeros((self.num_slots, self.max_knobs), dtype=torch.float32)

        # Helix JSON 내부의 tone/dsp0 블록 파싱
        dsp0_blocks = (
            preset_data.get("data", {})
            .get("tone", {})
            .get("dsp0", {})
        )

        # 슬롯별 탐색 및 라벨링
        for slot_idx, slot_name in enumerate(self.slot_names):
            candidate_models = self.model_catalog[slot_name]
            found_model = "None"
            found_block_dict = {}

            # dsp0 내 블록들 중 현재 슬롯에 해당하는 모델 찾기
            for block_key, block_val in dsp0_blocks.items():
                if isinstance(block_val, dict) and "@model" in block_val:
                    model_name = block_val["@model"]
                    if model_name in candidate_models:
                        found_model = model_name
                        found_block_dict = block_val
                        break

            # 1. Classification Target
            if found_model in candidate_models:
                cls_target[slot_idx] = candidate_models.index(found_model)
            else:
                cls_target[slot_idx] = 0  # None / Bypass

            # 2. Regression Target & Masking
            if found_model != "None" and self.knob_schema.get(found_model):
                schema_knobs = self.knob_schema[found_model]
                knob_keys = list(schema_knobs.keys())[: self.max_knobs]

                for knob_idx, knob_name in enumerate(knob_keys):
                    knob_info = schema_knobs[knob_name]
                    min_val = knob_info["min"]
                    max_val = knob_info["max"]
                    default_val = knob_info["default"]

                    # 프리셋 데이터 내 노브 값 추출 (없으면 기본값 사용)
                    raw_val = found_block_dict.get(knob_name, default_val)
                    if isinstance(raw_val, dict):  # 컨트롤러 할당 등으로 dict인 경우
                        raw_val = raw_val.get("@value", default_val)

                    # [0, 1] 범위로 Min-Max Normalize
                    if max_val > min_val:
                        norm_val = (raw_val - min_val) / (max_val - min_val)
                    else:
                        norm_val = 0.0

                    norm_val = max(0.0, min(1.0, float(norm_val)))

                    reg_target[slot_idx, knob_idx] = norm_val
                    reg_mask[slot_idx, knob_idx] = 1.0  # 유효 노브 마스크 활성화

        return cls_target, reg_target, reg_mask

    def reconstruct_hlx_dict(
        self,
        cls_preds: np.ndarray,
        reg_preds: np.ndarray,
        preset_name: str = "Tone_AI_Predicted",
    ) -> Dict[str, Any]:
        """
        모델의 예측 라벨(Class ID 및 Continuous Regressions)을 Helix Native가
        읽을 수 있는 완전한 형태의 .hlx (JSON) dict 객체로 역변환/복원합니다.

        Args:
            cls_preds (np.ndarray): (NUM_SLOTS,) 정수 클래스 예측값
            reg_preds (np.ndarray): (NUM_SLOTS, MAX_KNOBS) [0, 1] 범위 예측 노브값
            preset_name (str): 복원할 프리셋 이름

        Returns:
            Dict[str, Any]: Helix Native 완벽 호환 JSON Dict
        """
        dsp0_dict = {}
        block_counter = 0

        for slot_idx, slot_name in enumerate(self.slot_names):
            cls_id = int(cls_preds[slot_idx])
            candidate_models = self.model_catalog[slot_name]

            if cls_id <= 0 or cls_id >= len(candidate_models):
                continue  # "None" / Bypass 슬롯은 블록 생성 제외

            predicted_model = candidate_models[cls_id]
            block_key = f"block{block_counter}"
            block_type = config.BLOCK_TYPE_MAP.get(slot_name, 0)

            block_data: Dict[str, Any] = {
                "@type": block_type,
                "@model": predicted_model,
                "@enabled": True,
                "@position": slot_idx,
                "@path": 0,
            }

            # 노브 값 De-normalization ([0, 1] -> 실제 물리 파라미터 스케일)
            if self.knob_schema.get(predicted_model):
                schema_knobs = self.knob_schema[predicted_model]
                knob_keys = list(schema_knobs.keys())[: self.max_knobs]

                for knob_idx, knob_name in enumerate(knob_keys):
                    knob_info = schema_knobs[knob_name]
                    min_val = knob_info["min"]
                    max_val = knob_info["max"]

                    pred_norm_val = float(reg_preds[slot_idx, knob_idx])
                    pred_norm_val = max(0.0, min(1.0, pred_norm_val))

                    # De-normalize
                    real_val = pred_norm_val * (max_val - min_val) + min_val
                    # 정수형 파라미터인 경우 반올림 (예: Mic index)
                    if min_val.is_integer() and max_val.is_integer() and max_val - min_val > 5:
                        real_val = round(real_val)
                    else:
                        real_val = round(real_val, 4)

                    block_data[knob_name] = real_val

            dsp0_dict[block_key] = block_data
            block_counter += 1

        # Helix Native 표준 템플릿 JSON 생성
        hlx_dict: Dict[str, Any] = {
            "schema": "helix",
            "version": 67108864,
            "meta": {
                "name": preset_name,
                "application": "Helix Native",
                "app_version": "3.50.0",
            },
            "data": {
                "device": 2162694,
                "meta": {
                    "application": "Helix Native",
                    "app_version": "3.50.0",
                    "name": preset_name,
                },
                "tone": {
                    "dsp0": dsp0_dict,
                    "global": {
                        "@topology0": "A",
                    },
                },
            },
        }

        return hlx_dict
