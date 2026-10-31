[README_github.md](https://github.com/user-attachments/files/34210793/README_github.md)
# Agentic Multimodal ECG Diagnosis (Signal + Scalogram Fusion)

A deep-learning micro-project that classifies ECG heartbeats from two views of the same beat: the filtered **1D waveform** and its **2D wavelet scalogram**. A **LangGraph** orchestrator coordinates four agents (preprocessing, classification, retrieval, explanation), so each prediction comes with a confidence score, similar past cases and a **Grad-CAM** explanation.

> Academic prototype. Not a medical device and not clinically validated.

## Features

- **Dual-branch model:** 1D-CNN (waveform) + 2D-CNN (CWT scalogram), fused at feature level.
- **Ablation study:** trains signal-only, scalogram-only and fused models and compares accuracy, macro-F1 and confusion matrices.
- **Agentic workflow (LangGraph):** preprocessing, classification, retrieval and explanation agents with routing rules (retry on low confidence, flag when the two branches disagree).
- **Case retrieval:** FAISS index of fused embeddings returns the top-3 similar training beats.
- **Explainability:** Grad-CAM heat-maps on both the waveform and the scalogram, plus a generated text report.
- **Streamlit demo** showing the prediction, heat-maps, similar cases and the agent trace.

## Architecture

```
ECG beat -> Preprocessing Agent (filter, normalise, CWT scalogram)
         -> Classification Agent (1D-CNN + 2D-CNN -> fusion -> softmax)
              |-- low confidence? -> retry preprocessing (once)
         -> Retrieval Agent (FAISS top-k similar beats)
         -> Explanation Agent (Grad-CAM + report)
         -> Diagnosis output (class, confidence, similar cases, report)
```

## Project structure

| File | Purpose |
|------|---------|
| `config.py` | Settings: dataset records, class map, image size, training and agent thresholds |
| `data.py` | MIT-BIH loading, band-pass filter, beat segmentation, CWT scalogram, synthetic test data |
| `model.py` | 1D-CNN branch, 2D-CNN branch, fusion network, Grad-CAM |
| `train.py` | Trains the three variants, prints metrics, saves the model and FAISS index |
| `agents.py` | The four agents and the LangGraph orchestrator |
| `app.py` | Streamlit demo |
| `requirements.txt` | Python dependencies |

## Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet), 360 Hz. Beats are cut around the annotated R-peaks (180 samples) and grouped into three classes:

| Class | MIT-BIH symbols |
|-------|-----------------|
| Normal | N, L, R, e, j |
| Ventricular | V, E |
| Supraventricular | A, a, J, S |

A balanced subset is used (up to 3000 beats per class) and the loss is class-weighted.

## Installation

```bash
git clone https://github.com/Niyah18/<Agentic-Multimodal-ECG-Analysis>.git
cd <Agentic-Multimodal-ECG-Analysis>
pip install -r requirements.txt
```

## Usage

```bash
# 1. Quick test with synthetic beats (no internet needed)
python train.py --synthetic

# 2. Train on real MIT-BIH data (downloads from physionet.org via wfdb)
python train.py

# 3. Launch the demo
streamlit run app.py
```

`train.py` saves `ecg_fusion.pt` (model) and `ecg_faiss.npz` (retrieval index), which `agents.py` and `app.py` load. A GPU (for example Google Colab) speeds up training; change `RECORDS`, `EPOCHS` and `IMG_SIZE` in `config.py` to adjust the size of the run.

## Tech stack

Python, PyTorch, PyWavelets, scikit-learn, wfdb, FAISS, LangGraph, Streamlit, Matplotlib.

## Limitations and future work

- Trained on a subset of MIT-BIH only; no patient-wise split or clinical validation.
- Three simplified beat classes.
- Future work: EEG or PTB-XL as an extra modality, an LLM-written report, and evaluation on more datasets.

## Author

Niyah Meriyam Mathew, B.Tech, APJ Abdul Kalam Technological University.
