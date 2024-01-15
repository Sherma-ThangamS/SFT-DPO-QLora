# SFT-QLora Trainer Package

## Overview

Welcome to the SFT-QLora Trainer package! This package is designed to facilitate the training of the llms like Zephyr 7B model for custom use cases. The trainer handles the entire process from dataset processing to model fine-tuning.

## Installation

To install the SFT-QLora Trainer package, follow these steps:

```bash
pip install git+https://github.com/Sherma-ThangamS/SFT-QLora.git
```

## Usage

### 1. Import the Trainer and Config classes

```python
from zephyr_trainer import Trainer, Config
```

### 2. Create a Config object

```python
config = Config(
    MODEL_ID="TheBloke/zephyr-7B-alpha-GPTQ",
    DATA=["bitext/Bitext-customer-support-llm-chatbot-training-dataset", "instruction", "response"],
    CONTEXT_FIELD="",
    BITS=4,
    USE_EXLLAMA=False,
    # ... (configure other parameters as needed)
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

The `Config` class allows you to customize various parameters for the training process. Here are some key parameters:

- `MODEL_ID`: The identifier of the base model to use.
- `DATA`: List containing dataset information (ID, instruction field, target field).
- `BITS`: Number of bits for quantization.
- `USE_EXLLAMA`: Whether to use the ExLLAMA module.
- `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`: LoRA Adapter configuration.
- `BATCH_SIZE`: Training batch size.
- `OPTIMIZER`: Optimizer for training.
- `LR`: Learning rate.
- `NUM_TRAIN_EPOCHS`: Number of training epochs.
- `MAX_STEPS`: Maximum number of training steps.
- `FP16`: Enable mixed-precision training.
- `DATASET_TEXT_FIELD`: Name of the dataset text field.
- `MAX_SEQ_LENGTH`: Maximum sequence length for training.

Adjust these parameters based on your specific requirements.

## Dataset Processing

The `Trainer` class provides methods to download and process the training dataset:

- `create_dataset`: Downloads and processes the dataset.
- `process_data_sample`: Helper function to process individual dataset samples.

## Model Preparation

The `prepare_model` method prepares the model for fine-tuning:

- Quantizes the model.
- Attaches LoRA Adapter modules.

## Training

The `train` method initiates the training loop using the specified dataset and model:

- Downloads and processes the dataset.
- Prepares the model.
- Sets training arguments.
- Initializes the SFTTrainer from the trl library.
- Trains the model.

## Conclusion

With the SFT-QLora Trainer package, you can easily fine-tune the model for your custom use case. Experiment with different configurations to achieve optimal results for your specific chatbot application. If you encounter any issues or have questions, please refer to the documentation or reach out to our support community. Happy training!
