# Kazakh Vision-Language Models Research

This repository contains a research-oriented implementation of Kazakh Vision-Language Models (KazVLM).  
The main goal of the project is to explore how visual representations from pretrained vision encoders can be aligned with Kazakh language models for image captioning and vision-language understanding in the Kazakh language.

## Project Goal

Most modern Vision-Language Models are focused on high-resource languages such as English.  
This project investigates how to build and evaluate a Kazakh Vision-Language pipeline using:

- a pretrained vision encoder,
- projection layers for vision-to-language alignment,
- a Kazakh language model,
- translated image-caption datasets.

The project is designed as a small-scale research prototype for experimenting with different multimodal alignment strategies.

## Dataset

The dataset is based on image-caption pairs.  
Each sample contains an image path, English captions, and Kazakh translations of those captions.

Example sample:

```python
{
    "image_path": "val2014/COCO_val2014_000000522418.jpg",
    "split": "restval",
    "captions_en": [
        "A woman wearing a net on her head cutting a cake.",
        "A woman cutting a large white sheet cake.",
        "A woman wearing a hair net cutting a large sheet cake.",
        "There is a woman that is cutting a white cake.",
        "A woman marking a cake with the back of a chef's knife."
    ],
    "captions_kk": [
        "Басына тор киген әйел торт кесуде.",
        "Үлкен ақ табақ тортты кескен әйел.",
        "Шаш торап киген әйел үлкен табақ тортты кесіп жатыр.",
        "Ақ тортты кескен әйел бар.",
        "Тортты ашпақ пышағының арқасымен таңбалап отырған әйел."
    ]
}
## Dataset Fields

| Field | Description |
|---|---|
| `image_path` | Relative path to the image file |
| `split` | Dataset split, for example `train`, `validation`, or `restval` |
| `captions_en` | Original English image captions |
| `captions_kk` | Kazakh translations of the English captions |

The dataset is used to train and evaluate models that generate Kazakh textual descriptions from images.

## Model Architecture

The current pipeline follows a modular Vision-Language Model structure:

```text
Image
  ↓
Vision Encoder
  ↓
Projection Layer / Abstractor
  ↓
Kazakh Language Model
  ↓
Kazakh Caption / Answer

The project explores several alignment strategies:

### 1. Linear Projection Layer

A simple trainable linear layer maps visual embeddings into the language model embedding space.

### 2. Pooling + MLP Abstractor

A more expressive projection module pools visual features and maps them through a small MLP network.

### 3. Pooling + MLP Abstractor + LoRA / QLoRA

A parameter-efficient fine-tuning setup where the language model is adapted using LoRA-based methods.

## Repository Structure

```text
kazakh-vlm-research/
│
├── configs/
│   ├── vision_linear_llm.yaml
│   ├── vision_pooling_llm.yaml
│   └── vision_pooling_lorallm.yaml
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess_data.py
│   │
│   ├── models/
│   │   ├── linear.py
│   │   ├── llm.py
│   │   ├── mlp.py
│   │   ├── pooling.py
│   │   └── vision_encoder.py
│   │
│   └── training/
│       └── inference.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Experiments

The project currently includes the following experimental variants:

| Experiment | Vision Encoder | Projection Module | Language Model Adaptation |
|---|---|---|---|
| Linear VLM | Pretrained vision encoder | Linear layer | Frozen LLM |
| Pooling VLM | Pretrained vision encoder | Pooling + MLP | Frozen LLM |
| Pooling + LoRA VLM | Pretrained vision encoder | Pooling + MLP | LoRA / QLoRA |

## Installation

Clone the repository:

```bash
git clone https://github.com/morikonon/kazakh-vlm-research.git
cd kazakh-vlm-research
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data preprocessing

```bash
python src/data/preprocess_data.py
```

### Run inference

```bash
python src/training/inference.py
```

## Configuration

Model settings are stored in the `configs/` directory.

Example:

```text
configs/vision_linear_llm.yaml
configs/vision_pooling_llm.yaml
configs/vision_pooling_lorallm.yaml
```

Each config file defines the model components and experiment setup.

## Research Motivation

Kazakh is a low-resource language in the multimodal learning domain.  
Building Kazakh Vision-Language Models can help improve accessibility of image understanding systems for Kazakh-speaking users and contribute to research in low-resource multimodal AI.

## Author

**Mukhamedali Daniyaruly**  
KBTU  
Machine Learning / Vision-Language Models