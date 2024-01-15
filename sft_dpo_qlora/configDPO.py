class DPOConfig:
    def __init__(
        self,
        MODEL_ID="TheBloke/Mistral-7B-v0.1-GPTQ",
        BITS=4,
        LORA_R=8,
        LORA_ALPHA=8,
        LORA_DROPOUT=0.1,
        TARGET_MODULES=["q_proj", "v_proj"],
        BIAS="none",
        TASK_TYPE="CAUSAL_LM",
        INFERENCE_MODE=False,
        BATCH_SIZE=1,
        MAX_STEPS=50,
        LOGGING_FIRST_STEP=True,
        LOGGING_STEPS=10,
        OUTPUT_DIR="openhermes-mistral-dpo-gptq",
        OPTIMIZER="paged_adamw_32bit",
        LR=2e-4,
        WARMUP_STEPS=2,
        FP16=True,
        PUSH_TO_HUB=True,
        MAX_LENGTH=512,
        MAX_TARGET_LENGTH=256,
        MAX_PROMPT_LENGTH=256,
    ):
        self.MODEL_ID = MODEL_ID
        self.BITS = BITS
        self.LORA_R = LORA_R
        self.LORA_ALPHA = LORA_ALPHA
        self.LORA_DROPOUT = LORA_DROPOUT
        self.TARGET_MODULES = TARGET_MODULES
        self.BIAS = BIAS
        self.TASK_TYPE = TASK_TYPE
        self.INFERENCE_MODE = INFERENCE_MODE
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_STEPS = MAX_STEPS
        self.LOGGING_FIRST_STEP = LOGGING_FIRST_STEP
        self.LOGGING_STEPS = LOGGING_STEPS
        self.OUTPUT_DIR = OUTPUT_DIR
        self.OPTIMIZER = OPTIMIZER
        self.LR = LR
        self.WARMUP_STEPS = WARMUP_STEPS
        self.FP16 = FP16
        self.PUSH_TO_HUB = PUSH_TO_HUB
        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_TARGET_LENGTH = MAX_TARGET_LENGTH
        self.MAX_PROMPT_LENGTH = MAX_PROMPT_LENGTH
