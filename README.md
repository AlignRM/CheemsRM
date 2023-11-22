# Cheems: A Practical Framework for Chinese Reward Models

![GitHub stars](https://img.shields.io/github/stars/AlignRM/CheemsRM?style=social)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/github/license/AlignRM/CheemsRM)

This repository contains the official implementation for our paper [Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](https://arxiv.org/abs/2502.17173).

Cheems is designed to facilitate the development and evaluation of Chinese reward models, which are crucial for aligning large language models with human preferences. Our framework offers practical guidance, tools, and resources for researchers and practitioners working on Chinese LLM alignment.

## Features

- Complete training pipeline for Chinese reward models
- Carefully curated preference datasets for training
- Benchmark datasets (CheemsBench) for systematic evaluation
- Support for various model architectures and evaluation methods
- Comprehensive evaluation metrics and analysis tools
- Easy-to-use interface for integrating new models

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA compatible GPU (recommended)

### Setup

```bash
git clone https://github.com/AlignRM/CheemsRM.git
cd cheems
pip install -e .
```

## Data

We provide high-quality training and evaluation datasets:

### Training Data
- `data/cheems_preference.jsonl`: Contains paired responses with human preference annotations

### Evaluation Data (CheemsBench)
- `data/cheems_bench/human.jsonl`: Human-authored prompt subset.
- `data/cheems_bench/open.jsonl`: Open-source prompt subset.

## Training

To train your reward model:

```bash
bash scripts/train_rm.sh
```

You can customize training parameters by modifying the script or passing environment variable.

## Evaluation

Evaluate your reward model or LLM-as-judge on our benchmark:

```bash
# Evaluate a specific reward model
export MODEL_NAME=Skywork/Skywork-Reward-Gemma-2-27B-v0.2
# or
# export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

bash scripts/eval_rm.sh
```

To **evaluate new models**:
1. Implement a new Predictor in `cheems/eval/rm_predictor.py` or `cheems/eval/gen_predictor.py`
2. Add it to the `PREDICTOR_MAP` in the appropriate file

## Results

Our paper presents extensive analyses and benchmarks of various reward models. For detailed results and methodology, please refer to the [paper](https://arxiv.org/abs/2502.17173).

## Citation

If you find Cheems useful for your research or applications, please consider citing:

```bibtex
@misc{wen2025cheemspracticalguidancebuilding,
      title={Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch}, 
      author={Xueru Wen and Jie Lou and Zichao Li and Yaojie Lu and Xing Yu and Yuqiu Ji and Guohai Xu and Hongyu Lin and Ben He and Xianpei Han and Le Sun and Debing Zhang},
      year={2025},
      eprint={2502.17173},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.17173}, 
}
```

## Contact

For questions related to the code, paper, or collaboration opportunities, please contact:

- Email: `wenxueru2022@iscas.ac.cn`
- GitHub Issues: Feel free to open an issue in this repository