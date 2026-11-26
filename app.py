"""App Streamlit — reconhecimento de letras estáticas do alfabeto manual em LIBRAS.

Upload de fotos (ordem = palavra). Sem API: escolhe entre o modelo libras_datilologia
ou o classificador hand-talk (já treinados; sem retreino aqui).
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from handtalk import infer as ht_infer
from libras import config
from libras.detector import DetectorMao
from libras.reconhecedor import carregar_modelos, reconhecer_foto

BACKEND_LIBRAS = "libras_datilologia"
BACKEND_HANDTALK = "hand-talk"

st.set_page_config(
    page_title="LIBRAS — Datilologia",
    page_icon=":raised_hand:",
    layout="centered",
)


@st.cache_resource(show_spinner="Carregando libras_datilologia…")
def _recursos_libras():
    est, din = carregar_modelos()
    det = DetectorMao(video=False)
    return est, din, det


@st.cache_resource(show_spinner="Carregando hand-talk…")
def _recursos_handtalk():
    return ht_infer.carregar_classificador()


def _bytes_para_bgr(dados: bytes) -> np.ndarray | None:
    arr = np.frombuffer(dados, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _formatar_hipoteses(top: list) -> str:
    if not top:
        return "—"
    partes = []
    for letra, prob in top:
        nome = "∅" if letra == "_TRANSICAO" else letra
        partes.append(f"{nome} {prob:.0%}")
    return ", ".join(partes)


def _inferir_libras(img, mod_est, det) -> dict:
    r = det.detectar(img)
    d = reconhecer_foto(mod_est, r[0] if r else None, r[1] if r else "Right")
    aviso = bool(d.top and d.top[0][0] in config.LETRAS_DINAMICAS)
    if d.aceita:
        return {
            "tipo": "LETRA",
            "letra": d.letra,
            "confianca": d.confianca,
            "motivo": "",
            "hipoteses": _formatar_hipoteses(d.top),
            "aviso_dinamica": False,
        }
    motivo = d.motivo
    if aviso:
        motivo = (motivo + " — " if motivo else "") + (
            "possível letra com movimento (H/J/K/X/Y/Z)"
        )
    return {
        "tipo": "REPITA",
        "letra": None,
        "confianca": d.confianca,
        "motivo": motivo,
        "hipoteses": _formatar_hipoteses(d.top),
        "aviso_dinamica": aviso,
    }


def _inferir_handtalk(img, modelo) -> dict:
    d = ht_infer.reconhecer_foto(modelo, img)
    if d.aceita:
        return {
            "tipo": "LETRA",
            "letra": d.letra,
            "confianca": d.confianca,
            "motivo": "",
            "hipoteses": _formatar_hipoteses(d.top),
            "aviso_dinamica": False,
        }
    return {
        "tipo": "REPITA",
        "letra": None,
        "confianca": d.confianca,
        "motivo": d.motivo,
        "hipoteses": _formatar_hipoteses(d.top),
        "aviso_dinamica": False,
    }


def main():
    st.title("Tradutor do alfabeto em LIBRAS")
    st.caption(
        "Envie fotos da mão (uma letra por imagem). "
        "Compare os dois modelos já treinados — sem retreino neste app."
    )

    backend = st.radio(
        "Modelo",
        options=[BACKEND_LIBRAS, BACKEND_HANDTALK],
        horizontal=True,
        help=(
            "libras_datilologia: MLP + regra LETRA/REPITA (equipe do tema). "
            "hand-talk: classificador sklearn do seu projeto CCT (22 letras estáticas)."
        ),
    )

    if backend == BACKEND_LIBRAS:
        st.info(
            "Letras com movimento (H, J, K, X, Y, Z) **não** são reconhecidas só com foto "
            "neste modelo."
        )
    else:
        st.info(
            "Vocabulário estático: A B C D E F G I L M N O P Q R S T U V W Y "
            "(sem H, J, K, X, Z). Sem regra REPITA por margem — usa a 1ª hipótese se houver mão."
        )

    if "backend_atual" not in st.session_state:
        st.session_state.backend_atual = backend
    if backend != st.session_state.backend_atual:
        st.session_state.backend_atual = backend
        st.session_state.palavra = ""
        st.session_state.resultados = []

    try:
        if backend == BACKEND_LIBRAS:
            mod_est, _mod_din, det = _recursos_libras()
            ht_modelo = None
        else:
            ht_modelo, _meta = _recursos_handtalk()
            mod_est, det = None, None
    except Exception as e:
        st.error(f"Não foi possível carregar o modelo **{backend}**: {e}")
        st.stop()

    if "palavra" not in st.session_state:
        st.session_state.palavra = ""
    if "resultados" not in st.session_state:
        st.session_state.resultados = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    arquivos = st.file_uploader(
        "Fotos das letras (na ordem da palavra)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Limpar palavra", use_container_width=True):
            st.session_state.palavra = ""
            st.session_state.resultados = []
            st.session_state.uploader_key += 1  # remonta o file_uploader vazio
            st.rerun()
    with col_b:
        if st.button("Apagar última letra", use_container_width=True):
            st.session_state.palavra = st.session_state.palavra[:-1]
            if st.session_state.resultados:
                st.session_state.resultados.pop()
            st.rerun()
    with col_c:
        reconhecer = st.button(
            "Reconhecer",
            type="primary",
            use_container_width=True,
            disabled=not arquivos,
        )

    st.markdown(
        f"### Palavra: `{st.session_state.palavra or '—'}`  \n"
        f"<small>backend: **{backend}**</small>",
        unsafe_allow_html=True,
    )

    if reconhecer and arquivos:
        st.session_state.palavra = ""
        st.session_state.resultados = []
        progresso = st.progress(0, text="Processando…")
        total = len(arquivos)

        for i, arq in enumerate(arquivos):
            progresso.progress((i + 1) / total, text=f"{arq.name} ({i + 1}/{total})")
            dados = arq.getvalue()
            img = _bytes_para_bgr(dados)
            if img is None:
                st.session_state.resultados.append({
                    "nome": arq.name,
                    "preview": None,
                    "tipo": "REPITA",
                    "letra": None,
                    "confianca": 0.0,
                    "motivo": "imagem inválida",
                    "hipoteses": "—",
                    "aviso_dinamica": False,
                    "backend": backend,
                })
                continue

            if backend == BACKEND_LIBRAS:
                out = _inferir_libras(img, mod_est, det)
            else:
                out = _inferir_handtalk(img, ht_modelo)

            if out["tipo"] == "LETRA" and out["letra"]:
                st.session_state.palavra += out["letra"]

            st.session_state.resultados.append({
                "nome": arq.name,
                "preview": dados,
                "backend": backend,
                **out,
            })

        progresso.empty()
        st.rerun()

    if st.session_state.resultados:
        st.divider()
        st.subheader("Resultados por imagem")
        for item in st.session_state.resultados:
            c1, c2 = st.columns([1, 2])
            with c1:
                if item["preview"] is not None:
                    st.image(
                        Image.open(io.BytesIO(item["preview"])),
                        caption=item["nome"],
                        use_container_width=True,
                    )
                else:
                    st.write(item["nome"])
            with c2:
                if item["tipo"] == "LETRA":
                    st.success(
                        f"LETRA RECONHECIDA: **{item['letra']}** "
                        f"({item['confianca']:.0%})"
                    )
                else:
                    st.warning(f"REPITA O SINAL ({item['motivo']})")
                st.caption(f"Hipóteses: {item['hipoteses']}")


main()
