import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, GPTQConfig
from trl import DPOTrainer
from huggingface_hub import login
login()


class dpoTrainer:
    def __init__(self, config=None):
        if config is None:
            raise ValueError("Please provide a DPOConfig object when creating an instance of DPOTrainer.")

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=GPTQConfig(bits=self.config.BITS, disable_exllama=True)
        )

        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=GPTQConfig(bits=self.config.BITS, disable_exllama=True)
        )

        self.train_dataset = self._dpo_data()

    def _dpo_data(self):
        dataset = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized",
            split="test_prefs",
            use_auth_token=True
        )

        original_columns = dataset.column_names

        def return_prompt_and_responses(samples):
            return {
                "prompt": [prompt for prompt in samples["prompt"]],
                "chosen": samples["chosen"],
                "rejected": samples["rejected"],
            }

        return dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=original_columns,
        )

    def train(self):
        train_df = self.train_dataset.to_pandas()
        train_df["chosen"] = train_df["chosen"].apply(lambda x: x[1]["content"])
        train_df["rejected"] = train_df["rejected"].apply(lambda x: x[1]["content"])
        train_df = train_df.dropna()
        val_df = train_df.sample(10)
        train_data = Dataset.from_pandas(train_df)
        val_data = Dataset.from_pandas(val_df)

        peft_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.TARGET_MODULES,
            bias=self.config.BIAS,
            task_type=self.config.TASK_TYPE,
        )
        peft_config.inference_mode = self.config.INFERENCE_MODE

        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.config.pretraining_tp = 1
        self.model = get_peft_model(self.model, peft_config)

        self.model_ref = prepare_model_for_kbit_training(self.model_ref)
        self.model_ref.config.use_cache = False
        self.model_ref.gradient_checkpointing_enable()
        self.model_ref.config.pretraining_tp = 1
        self.model_ref = get_peft_model(self.model_ref, peft_config)

        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.BATCH_SIZE,
            max_steps=self.config.MAX_STEPS,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=self.config.LR,
            evaluation_strategy="steps",
            logging_first_step=self.config.LOGGING_FIRST_STEP,
            logging_steps=self.config.LOGGING_STEPS,
            output_dir=self.config.OUTPUT_DIR,
            optim=self.config.OPTIMIZER,
            warmup_steps=self.config.WARMUP_STEPS,
            fp16=self.config.FP16,
            push_to_hub=self.config.PUSH_TO_HUB
        )

        dpo_trainer = DPOTrainer(
            self.model,
            self.model_ref,
            args=training_args,
            beta=0.1,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_LENGTH,
            max_target_length=self.config.MAX_TARGET_LENGTH,
            max_prompt_length=self.config.MAX_PROMPT_LENGTH
        )

        dpo_trainer.train()

        dpo_trainer.push_to_hub()
