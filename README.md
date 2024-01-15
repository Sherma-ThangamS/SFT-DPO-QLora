# SFT-DPO-QLora Trainer Package

## Overview

Welcome to the SFT-DPO-QLora Trainer package! This package is designed to streamline the training process for large language models (LLMs) using the SFT (Scaling-Free Training) and QLora techniques, specifically tailored for Dialogue Preference Optimization (DPO) scenarios. This trainer handles the entire pipeline, from dataset processing to model fine-tuning, providing users with a convenient solution for their custom use cases.

## Installation

To install the SFT-DPO-QLora Trainer package, follow these steps:

```bash
pip install sft-dpo-qlora
```

## Usage

### 1. Import the Trainer and Config classes

```python
from sft_dpo_qlora import Trainer, Config
```

### 2. Create a Config object

```python
config = Config(
    MODEL_ID="Model/quantized-model",
    DATA=["YourHuggingFaceDataset/dataset-name","Questions","Answers]
    BITS=4,
    LORA_R=8,
    LORA_ALPHA=8,
    LORA_DROPOUT=0.1,
    TARGET_MODULES=["q_proj", "v_proj"],
    BIAS="none",
    TASK_TYPE="CAUSAL_LM",
    BATCH_SIZE=8,
    OPTIMIZER="paged_adamw_32bit",
    LR=2e-4,
    NUM_TRAIN_EPOCHS=1,
    MAX_STEPS=250,
    FP16=True,
    DATASET_SPLIT="test_prefs",
    MAX_LENGTH=512,
    MAX_TARGET_LENGTH=256,
    MAX_PROMPT_LENGTH=256,
    INFERENCE_MODE=False,
    LOGGING_FIRST_STEP=True,
    LOGGING_STEPS=10,
    OUTPUT_DIR="your_output_directory",
    PUSH_TO_HUB=True
)
```

### 3. Initialize the Trainer with the Config object

```python
trainer = Trainer(config)
```

### 4. Train the model

```python
trainer.train()
```

## Configuring the Trainer

The `Config` class allows you to customize various parameters for the training process. Key parameters include:

- `MODEL_ID`: The identifier of the base model to use.
- `DATA`: The Hugging Face dataset name , Instruction , Target
- `BITS`: Number of bits for quantization.
- `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`: LoRA Adapter configuration.
- `TARGET_MODULES`: Target modules for the LoRA Adapter.
- `BIAS`: Bias for LoRA Adapter.
- `TASK_TYPE`: Type of the task (e.g., "CAUSAL_LM").
- `BATCH_SIZE`: Training batch size.
- `OPTIMIZER`: Optimizer for training.
- `LR`: Learning rate.
- `NUM_TRAIN_EPOCHS`: Number of training epochs.
- `MAX_STEPS`: Maximum number of training steps.
- `FP16`: Enable mixed-precision training.
- `DATASET_SPLIT`: Split of the dataset to use.
- `MAX_LENGTH`, `MAX_TARGET_LENGTH`, `MAX_PROMPT_LENGTH`: Maximum sequence lengths.
- `INFERENCE_MODE`: Enable inference mode for LoRA Adapter.
- `LOGGING_FIRST_STEP`, `LOGGING_STEPS`: Logging configuration.
- `OUTPUT_DIR`: Output directory for training results.
- `PUSH_TO_HUB`: Push results to the Hugging Face Hub.

Adjust these parameters based on your specific requirements and the characteristics of your dialogue preference optimization task.

## Dataset Processing

The `Trainer` class provides methods to download and process the training dataset:

- `_dpo_data`: Downloads and processes the DPO dataset.

## Model Preparation

The `train` method initiates the training loop using the specified dataset and model:

- Downloads and processes the DPO dataset.
- Prepares the model.
- Sets training arguments.
- Initializes the DPOTrainer from the trl library.
- Trains the model.

## Conclusion

With the SFT-DPO-QLora Trainer package, you can easily fine-tune large language models for dialogue preference optimization in your custom use case. Experiment with different configurations, datasets, and models to achieve optimal results. If you encounter any issues or have questions, please refer to the documentation or reach out to our support community. Happy training!
