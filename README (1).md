# 🤖 MiniGPT Instructivo en Español

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Proyecto Final de Doctorado en IA/NLP**

Construcción, Entrenamiento y Evaluación de un Modelo GPT Instrucional End-to-End

---

## 📋 Descripción

Este proyecto implementa un modelo de lenguaje tipo GPT (MiniGPT) de **110 millones de parámetros**, entrenado desde cero en español para seguir instrucciones. El modelo fue desarrollado como parte del proyecto final de doctorado en Inteligencia Artificial y Procesamiento del Lenguaje Natural.

### Características Principales

- 🔤 **Tokenizador BPE personalizado** con 32,000 tokens optimizado para español
- 📚 **Dataset híbrido** de 57,471 instrucciones en español
- 🧠 **Arquitectura GPT-2** (decoder-only) de 110M parámetros
- 🎯 **Fine-tuning con LoRA** para especialización en ciencia y programación
- 📊 **Evaluación exhaustiva** con métricas cuantitativas y cualitativas
- 💬 **Interfaz interactiva** con Gradio

---

## 🏗️ Estructura del Proyecto

```
MiniGPT_Doctoral/
│
├── 📁 data/
│   └── processed/
│       ├── train.json              # Dataset de entrenamiento (54,597 ejemplos)
│       └── validation.json         # Dataset de validación (2,874 ejemplos)
│
├── 📁 tokenizer/
│   ├── bpe_tokenizer.json          # Tokenizador BPE entrenado
│   └── hf_tokenizer/               # Formato HuggingFace
│       ├── tokenizer.json
│       ├── special_tokens_map.json
│       └── tokenizer_config.json
│
├── 📁 miniGPT_final/               # Modelo entrenado
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── generation_config.json
│
├── 📁 miniGPT_lora_ciencia_prog/   # Adaptadores LoRA
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
├── 📁 checkpoints/                 # Checkpoints de entrenamiento
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   ├── checkpoint-3000/
│   ├── checkpoint-4000/
│   └── checkpoint-5000/
│
├── 📁 analysis/                    # Resultados y visualizaciones
│   ├── training_curves.png
│   ├── benchmark_results.png
│   ├── coherence_score.png
│   ├── error_analysis.png
│   ├── tokenizer_comparison.png
│   └── evaluation_report.txt
│
├── 📁 notebooks/                   # Notebooks del pipeline
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_tokenizer_bpe.ipynb
│   ├── 03_encoder_vs_decoder.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_chat_interface.ipynb
│   └── 07_lora_finetuning.ipynb
│
├── chat_interface.py               # Interfaz Gradio standalone
├── requirements.txt                # Dependencias
└── README.md                       # Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.10 o superior
- CUDA 11.8+ (para entrenamiento con GPU)
- 16GB+ RAM
- GPU con 16GB+ VRAM (recomendado: A100, L4, T4)

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/[tu-usuario]/MiniGPT-Doctoral.git
cd MiniGPT-Doctoral

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
tokenizers>=0.15.0
accelerate>=0.27.0
peft>=0.10.0
trl>=0.8.0
gradio>=4.0.0
evaluate>=0.4.0
rouge-score>=0.1.2
nltk>=3.8.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 📊 Pipeline del Proyecto

### 1️⃣ Preparación del Dataset

```bash
# Ejecutar notebook
jupyter notebook notebooks/01_dataset_preparation.ipynb
```

**Fuentes del dataset:**
- Alpaca Español: 51,942 ejemplos
- OpenAssistant ES: 14,038 ejemplos
- Instrucciones originales: 5 ejemplos

**Resultado:** 57,471 instrucciones únicas en formato Alpaca

### 2️⃣ Entrenamiento del Tokenizador BPE

```bash
jupyter notebook notebooks/02_tokenizer_bpe.ipynb
```

**Configuración:**
- Vocabulario: 32,000 tokens
- Algoritmo: Byte Pair Encoding
- Eficiencia: 5.10 caracteres/token (75% mejor que GPT-2)

### 3️⃣ Comparación Encoder vs Decoder

```bash
jupyter notebook notebooks/03_encoder_vs_decoder.ipynb
```

Análisis comparativo entre arquitecturas BERT (encoder-only) y GPT (decoder-only).

### 4️⃣ Entrenamiento del Modelo

```bash
jupyter notebook notebooks/04_training.ipynb
```

**Configuración del modelo:**
| Parámetro | Valor |
|-----------|-------|
| Arquitectura | GPT-2 |
| Parámetros | 110M |
| Capas | 12 |
| Heads | 12 |
| Embedding | 768 |
| Contexto | 512 tokens |

**Resultados:**
- Steps: 5,121
- Tiempo: 88.6 minutos
- Loss inicial: 9.0654
- Loss final: 3.6201
- Reducción: 60.1%

### 5️⃣ Evaluación

```bash
jupyter notebook notebooks/05_evaluation.ipynb
```

**Métricas obtenidas:**
| Métrica | Valor |
|---------|-------|
| Perplejidad | 80.38 |
| BLEU | 1.44 |
| ROUGE-1 | 0.1710 |
| ROUGE-L | 0.1096 |
| Coherence Score | 0.6931 |

### 6️⃣ Interfaz Interactiva

```bash
# Opción 1: Notebook
jupyter notebook notebooks/06_chat_interface.ipynb

# Opción 2: Script standalone
python chat_interface.py
```

### 7️⃣ Fine-tuning con LoRA (Tarea Avanzada)

```bash
jupyter notebook notebooks/07_lora_finetuning.ipynb
```

**Resultados LoRA:**
- Parámetros entrenables: 2.10%
- Tiempo: 18.6 minutos
- Mejora en perplejidad: 99.1%
- Tamaño adaptadores: 11.8 MB

---

## 💬 Uso del Modelo

### Carga Básica

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar modelo y tokenizador
model = AutoModelForCausalLM.from_pretrained("./miniGPT_final")
tokenizer = AutoTokenizer.from_pretrained("./miniGPT_final")

# Generar respuesta
def generate_response(instruction, max_tokens=200):
    prompt = f"### Instrucción:\n{instruction}\n\n### Respuesta:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ejemplo
response = generate_response("¿Qué es la inteligencia artificial?")
print(response)
```

### Con Adaptadores LoRA

```python
from peft import PeftModel

# Cargar modelo base
base_model = AutoModelForCausalLM.from_pretrained("./miniGPT_final")

# Cargar adaptadores LoRA
model = PeftModel.from_pretrained(base_model, "./miniGPT_lora_ciencia_prog")

# Usar igual que antes
response = generate_response("Escribe una función en Python que calcule el factorial.")
```

### Interfaz Gradio

```bash
python chat_interface.py
# Abre http://localhost:7860 en tu navegador
```

---

## 📈 Resultados

### Curvas de Entrenamiento

![Training Curves](analysis/training_curves.png)

### Benchmark por Categoría

![Benchmark Results](analysis/benchmark_results.png)

### Distribución de Coherence Score

![Coherence Score](analysis/coherence_score.png)

---

## 📁 Entregables del Proyecto

| Entregable | Ubicación | Estado |
|------------|-----------|--------|
| Memoria técnica (20 págs) | `docs/Informe_Tecnico_MiniGPT.pdf` | ✅ |
| Tokenizador BPE | `tokenizer/` | ✅ |
| Dataset curado | `data/processed/` | ✅ |
| Modelo entrenado | `miniGPT_final/` | ✅ |
| Checkpoints | `checkpoints/` | ✅ |
| Evaluación | `analysis/` | ✅ |
| Interfaz | `chat_interface.py` | ✅ |
| LoRA (Tarea Avanzada) | `miniGPT_lora_ciencia_prog/` | ✅ |

---

## 🔬 Tareas Completadas

### Obligatorias

- [x] **Tarea 1:** Construcción de Tokenizador BPE (32k tokens)
- [x] **Tarea 2:** Comparación Encoder-Only vs Decoder-Only
- [x] **Tarea 3:** Entrenamiento de MiniGPT Instructivo
- [x] **Tarea 4:** Evaluación Exhaustiva del Modelo
- [x] **Tarea 5:** Interfaz Interactiva (Gradio)

### Avanzadas (Súper Distinción)

- [x] **Tarea B:** Implementación de LoRA/QLoRA

---

## 📚 Referencias

1. Vaswani, A., et al. (2017). "Attention is All You Need"
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners"
4. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
5. Taori, R., et al. (2023). "Stanford Alpaca"

---

## 👤 Autor

**[Tu Nombre]**
- Doctorado en Inteligencia Artificial
- [Tu Universidad]
- Email: [tu@email.com]

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- Anthropic por Claude (asistencia en desarrollo)
- HuggingFace por las herramientas de NLP
- Google Colab por recursos de cómputo
- Comunidad de Alpaca Español por el dataset base

---

<p align="center">
  <b>Proyecto Final de Doctorado en IA/NLP</b><br>
  Diciembre 2024
</p>
