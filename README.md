# LIBRAS — app Streamlit (datilologia)

App único em Streamlit que reconhece letras **estáticas** do alfabeto manual em LIBRAS a partir de fotos e monta a palavra soletrada.

Não há API separada: o Streamlit carrega MediaPipe + os modelos `.joblib` no mesmo processo.

## O que faz (v1)

- Upload de uma ou várias fotos (ordem = ordem da palavra)
- Seletor de modelo: **libras_datilologia** (equipe do tema) ou **hand-talk** (classificador CCT)
- Saída por imagem: `LETRA RECONHECIDA` ou `REPITA O SINAL`
- Palavra em construção (limpar / apagar última)

No backend `libras_datilologia`, letras com movimento (**H, J, K, X, Y, Z**) não são cobertas por foto única. No `hand-talk`, o vocabulário estático inclui **Y** e exclui H/J/K/X/Z.

## Rodar local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

O arquivo `hand_landmarker.task` (pesos do detector de mão) já vai no repositório. Se ele não estiver na raiz, a primeira execução baixa do MediaPipe.

## Deploy (Streamlit Cloud)

1. Suba este repositório no Git (GitHub / Git vinculado ao Streamlit).
2. Em [share.streamlit.io](https://share.streamlit.io): New app → escolha o repo.
3. **Main file path:** `app.py`
4. Python 3.12 se disponível.

O `packages.txt` instala as libs de sistema do OpenCV e do MediaPipe no Linux do Cloud (`libgl1`, `libegl1`, `libgles2`). Sem `libegl1`, o carregamento do modelo falha com `libEGL.so.1: cannot open shared object file`.

> O stack inclui `torch` e MediaPipe — no free tier o build pode demorar ou estourar memória. Se falhar no Cloud, use a demo local.

## Origem dos modelos

Os arquivos em `modelos/` e o pacote `libras/` vêm do projeto de treino da equipe (`libras_datilologia`): landmarks MediaPipe + classificador (MLP) para letras estáticas.
