"""Carrega sign_classifier.joblib e reconhece uma foto (API do hand-talk)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_VECTOR_SIZE, extract_landmarks

DIR_MODELO = Path(__file__).resolve().parent.parent / "modelos" / "handtalk"
ARQ_MODELO = DIR_MODELO / "sign_classifier.joblib"
ARQ_META = DIR_MODELO / "sign_classifier.json"


@dataclass
class ResultadoHT:
    letra: str | None
    confianca: float
    motivo: str = ""
    top: list = field(default_factory=list)

    @property
    def aceita(self) -> bool:
        return self.letra is not None


def carregar_classificador(pasta: Path = DIR_MODELO):
    arq = pasta / "sign_classifier.joblib"
    if not arq.exists():
        raise RuntimeError(
            f"Modelo hand-talk não encontrado: {arq}\n"
            "Copie sign_classifier.joblib de hand-talk/models/ para modelos/handtalk/."
        )
    modelo = joblib.load(arq)
    meta = {}
    meta_path = pasta / "sign_classifier.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    expected = meta.get("feature_vector_size", FEATURE_VECTOR_SIZE)
    if expected != FEATURE_VECTOR_SIZE:
        raise RuntimeError(
            f"Modelo espera vetor {expected}, features usam {FEATURE_VECTOR_SIZE}."
        )
    return modelo, meta


def reconhecer_foto(modelo, img_bgr: np.ndarray) -> ResultadoHT:
    vec = extract_landmarks(img_bgr)
    if vec is None:
        return ResultadoHT(None, 0.0, "nenhuma mão encontrada na imagem")
    probs = modelo.predict_proba(vec.reshape(1, -1))[0]
    classes = [str(c) for c in modelo.classes_]
    ordem = np.argsort(probs)[::-1]
    top = [(classes[i], float(probs[i])) for i in ordem[:3]]
    letra, conf = top[0]
    return ResultadoHT(letra, conf, "", top)
