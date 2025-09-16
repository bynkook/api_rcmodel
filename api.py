"""
api.py — RC(철근콘크리트) 단면 예측 API (FastAPI)

요약
- 목적: 브라우저(또는 다른 클라이언트)에서 입력한 단면/재료/하중 정보를 받아
        사전에 학습·저장된 scikit-learn 파이프라인으로
        d, As_required 등의 타깃을 예측해 반환한다.
- 모델 번들 형식(joblib):
    {
      "models": { target_name: fitted_estimator, ... },
      "features_by_target": { target_name: [feat1, feat2, ...], ... },
      "targets": [target_name1, target_name2, ...]
    }
- 파생/별칭 규칙:
    * UI 입력의 Mu → 모델 입력의 phi_mn 로 매핑
    * 출력 시 as_provided → As_required, phi_mn → phi_Mn 로 키명 변경(표시용)
    * f_idx 는 문자열로 fck+fy 입력값에서 생성
- 엔드포인트:
    GET  /metadata        : UI가 필요로 하는 입력/출력 필드, 적재된 모델 타깃 목록 조회
    POST /predict_multi   : 여러 타깃을 한 번에 예측. 누락된 입력 피처 목록도 함께 반환

주의
- 이 API는 CORS 허용으로 프론트엔드 정적 파일(예: file:// 또는 다른 포트)에서 호출 가능.
- Pydantic v2 기준(BaseModel.model_dump 사용).
- 모델 훈련 및 예측값은 Sm (Ig 는 출력시 계산하여 표시면)
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# === 모델 번들 경로 설정 ===
# BUNDLE_PATH = os.getenv("XGB_BUNDLE", "xgb_bundle.joblib")
# BUNDLE_PATH = os.getenv("SVR_BUNDLE", "svr_bundle.joblib")
BUNDLE_PATH = os.getenv("STACK_BUNDLE", "stack_bundle_1.joblib")

# === FastAPI 앱 및 CORS 설정 ===
app = FastAPI(title="RC API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 개발/테스트 편의를 위해 전체 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 모델 번들 적재 ===
# bundle: {"models":{tgt:est}, "features_by_target":{tgt:[feats]}, "targets":[...]}
bundle = joblib.load(BUNDLE_PATH)
MODELS: Dict[str, Any] = bundle["models"]
FEATS_BY_TGT: Dict[str, List[str]] = bundle["features_by_target"]
TARGETS: List[str] = bundle["targets"]

# === 입력/출력 키 매핑 규칙 ===
# UI→모델: Mu 라는 UI 입력명을 모델 입력명 phi_mn 으로 치환
INPUT_ALIAS = {"Mu": "phi_mn"}
# 모델→UI: 앱 의도에 맞는 출력을 위한 키명 변환
OUTPUT_RENAME = {"as_provided": "As_required", "phi_mn": "phi_Mn"}

# === FLOW CONTROL ===
TOL_MU = 0.05   # ±5%
MAX_ITERS = 5   # 반복 횟수

# === 요청 바디 스키마 ===
class PredictReq(BaseModel):
    # 설계기준 압축강도(MPa)
    fck: float = Field(..., description="MPa")
    # 철근 항복강도(MPa)
    fy: float = Field(..., description="MPa")
    # 단면 타입(현재 무시: 'r' 등 UI 입력만 받고 모델 피처에는 사용하지 않음)
    type: str  = Field(..., description="")
    # 단면 폭(mm)
    width: float = Field(..., description="mm")
    # 단면 높이(mm)
    height: float = Field(..., description="mm")
    # 외력 모멘트(kN·m) — 모델 입력명 phi_mn 으로 매핑됨
    Mu: float = Field(..., description="kN·m")

# === 메타데이터: UI에서 참조할 입력/출력 필드와 타깃 목록 제공 ===
@app.get("/metadata")
def metadata():
    """
    UI가 폼을 동적으로 구성할 때 필요한 정보 제공.
    - ui_inputs : 브라우저 폼의 입력 필드 목록
    - ui_outputs: 브라우저에 표시할 출력 필드 목록
    - targets_loaded: 서버에 적재된 타깃 모델 목록
    """
    return {
        "ui_inputs": ["fck", "fy", "type", "width", "height", "Mu"],
        "ui_outputs": ["d", "Ig", "As_required", "phi_Mn"],
        "targets_loaded": list(MODELS.keys())
    }

# === 다중 타깃 예측 ===
@app.post("/predict_multi")
def predict_multi(req: PredictReq):
    """
    요청 바디(PredictReq)를 받아:
      1) UI→모델 키 치환(Mu→phi_mn)
      2) 파생 피처 계산
      3) 각 타깃별로 필요한 피처 서브셋을 구성해 예측
      4) 누락 피처 집계 + UI 표기용 키로 결과 반환
    예측 결과 포맷:
    {
      "predictions": { "Sm", "bd", "rho", "phi_mn" },
      "missing_inputs": {타깃별 요구 피처 중 None/미존재},
      "echo": {사용자 원본 입력 에코백(검증/디버깅용)}
    }
    """
    # Pydantic v2: dict 변환
    payload_ui = req.model_dump()   # {fck, fy, type, width, height, Mu}
    payload = {k: v for k, v in payload_ui.items() if k not in ['type']} # type 은 미사용으로 제거
    
    # 2025.08.27 수정
    vwidth = float(payload.get("width", 0) or 0)
    vheight = float(payload.get("height", 0) or 0)
    vfck = float(payload.get("fck", 0) or 0)
    vfy = float(payload.get("fy", 0) or 0)
    payload['f_idx'] = int(str(int(vfck))+str(int(vfy))) / 1e3

    # UI 입력 'Mu' → 모델 입력 'phi_mn' 으로 치환
    payload[INPUT_ALIAS["Mu"]] = payload.pop("Mu")

    # payload 미사용 feature 삭제
    payload = {k: v for k, v in payload.items() if k not in ['fck', 'fy']}
    
    # ===============================
    # 반복 예측(동시 갱신), 입력은 고정
    # feat_iter: 고정 입력 + 최신 예측값
    # ===============================
    feat_iter: Dict[str, Any] = dict(payload)  # copy payload

    last_preds: Dict[str, float] = {}
    for _ in range(MAX_ITERS):
        preds_k: Dict[str, float] = {}
        for tgt in TARGETS:
            feats = FEATS_BY_TGT.get(tgt, [])
            mdl = MODELS.get(tgt)
            # row 는 첫회차에 payload 와 동일, 2회차부터 선택적으로 추가됨
            row = {f: feat_iter.get(f, None) for f in feats}
            X = pd.DataFrame([row])
            try:
                preds_k[tgt] = float(mdl.predict(X)[0])
            except Exception:
                preds_k[tgt] = np.nan

        # 동시 갱신: bd, Sm, rho 만 갱신
        if np.isfinite(preds_k.get("bd", np.nan)) and np.isfinite(preds_k.get("Sm", np.nan)) and np.isfinite(preds_k.get("rho", np.nan)):
            feat_iter["bd"] = preds_k["bd"]     # cm2
            feat_iter["Sm"] = preds_k["Sm"]     # cm2
            feat_iter['rho'] = preds_k['rho']
        last_preds = preds_k

    # 최종 결과 맵핑
    results: Dict[str, Any] = {}
    for k, v in last_preds.items():
        key = OUTPUT_RENAME.get(k, k)
        results[key] = None if (v is None or not np.isfinite(v)) else float(v)

    return {
        "predictions": results,
        "echo": payload_ui,
    }