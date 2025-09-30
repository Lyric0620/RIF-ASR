# 🎙️ RIF-ASR

> Implements the **LIR-ASR** correction paradigm proposed in:
> 📄 *“Listening, Imagining & Refining: A Heuristic Optimized ASR Correction Framework with LLMs”* ([arXiv:2509.15095](https://arxiv.org/abs/2509.15095))  

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)  
![License](https://img.shields.io/badge/license-MIT-green.svg)  
![ASR](https://img.shields.io/badge/benchmark-ASR-red.svg)  

---

## 📑 Table of Contents  

- [📂 Datasets](#-datasets)  
- [📊 Evaluation Metrics](#-evaluation-metrics)  
- [⚙️ Supported ASR Engines](#%EF%B8%8F-supported-asr-engines)  
- [🧩 LIR-ASR Correction Framework](#-lir-asr-correction-framework)  
- [🚀 Usage](#-usage) 
- [📚 References](#-references)  

---

## 📂 Datasets  

The framework supports the following datasets (and can be extended easily):  

- **LibriSpeech** (test-clean / test-other)  
- **TED-LIUM**  
- **CommonVoice**  
- **Multilingual LibriSpeech (MLS)**  
- **VoxPopuli**  
- **Fleurs** (multi-language, with helper script `scripts/download_fleurs.sh`)  

---

## 📊 Evaluation Metrics  

We evaluate ASR and correction performance using:  

- 📝 **Word Error Rate (WER)** — word-level edit distance / reference count  
- ✒️ **Punctuation Error Rate (PER)** — errors in `.`, `,`, `?`, etc.  
- ⏱️ **Core-Hour** — CPU hours per 1h audio (for local models)  
- 💾 **Model Size** — size (MB) of acoustic + language models  

> ⚠️ For **cloud ASR services**, Core-Hour and Model Size are **not reported**.  

---

## ⚙️ Supported ASR Engines  

| Cloud Services         | Local / Open Models      |
|------------------------|--------------------------|
| Amazon Transcribe      | OpenAI Whisper (tiny → large) |
| Azure Speech-to-Text   | whisper.cpp              |
| Google Speech-to-Text  | Coqui STT                |
| IBM Watson             | Custom ASR models        |
| Picovoice Cheetah      |                          |
| Picovoice Leopard      |                          |

---

## 🧩 LIR-ASR Correction Framework  

### 🔍 Overview  

**LIR-ASR (Listening → Imagining → Refining)** is an **iterative correction framework** inspired by how humans “rehear” ambiguous speech.  

**Pipeline:**  
1. 👂 **Listening** — detect uncertain/misrecognized words  
2. 💭 **Imagining** — generate candidate variants (phonetic substitutions, G2P)  
3. ✨ **Refining** — use LLMs / scoring models to pick the most consistent  

**Key Features:**  
- 📌 **FSM Controller** — 3 states: *NoSearch → Search → Search++*  
- 🧭 **Heuristic Optimization** — rule-based semantic constraints  
- 🔄 **Iterative Refinement** — until convergence, ensuring monotonic score improvements  

> ✅ Achieves **~1.5% CER/WER improvements** on average over uncorrected baselines.  

---

## 🚀 Usage  

### 🔧 Setup  

```bash
# Install dependencies
pip3 install -r requirements.txt

# Prepare datasets
sh scripts/download_fleurs.sh  # example for Fleurs
```

### 📝 Example: RIF-ASR Optimization
```
from optim import prompt_optimization, evolutionary_prompt_optimization, nbest_optimization, RIF_ASR
from normalizer import Normalizer
from languages import Languages

# Init normalizer
norm = Normalizer.create(language=Languages.ZH, keep_punctuation=True, punctuation_set=".?")

asr_text = "由于分离和重组便宜在每一代的两个库之间来回变动"

# 1. Prompt optimization
corrected = prompt_optimization(asr_text, llm="Qwen3-235B")

# 2. Evolutionary optimization
refined = evolutionary_prompt_optimization(asr_text, llm="Qwen3-235B")

# 3. RIF-ASR with multiple candidates
result = RIF_ASR(asr_text, llm="Qwen3-235B", language="ZH", normalizer=norm)
```

### 🏃 Running Benchmarks
```
sh scripts/evaluation_whisper_qwen.sh
```

## 📚 References
```
@misc{liu2025listeningimaginingrefining,
      title={Listening, Imagining & Refining: A Heuristic Optimized ASR Correction Framework with LLMs}, 
      author={Yutong Liu and Ziyue Zhang and Cheng Huang and Yongbin Yu and Xiangxiang Wang and Yuqing Cai and Nyima Tashi},
      year={2025},
      eprint={2509.15095},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2509.15095}, 
}
```
