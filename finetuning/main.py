from finetuning import dataset

import torch
import trl
import transformers
import peft

_BASE_MODEL = "mistralai/Mistral-7B-v0.1"
_FINETUNED_MODEL_NAME = "mistral_finetuned"


def _finetune_and_store_model(
    base_model=_BASE_MODEL, finetuned_model_name=_FINETUNED_MODEL_NAME
):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        load_in_4bit=True,
        use_flash_attention_2=True,
        use_cache=False,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, pad_token=tokenizer.eos_token, padding_side="left"
    )

    peft_config = peft.LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
        ],
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(model, peft_config)

    args = transformers.TrainingArguments(
        output_dir=finetuned_model_name,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=6,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        logging_steps=1,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        lr_scheduler_type="constant",
    )
    trainer = trl.SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset.get_dataset(split="train"),
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(finetuned_model_name)


if __name__ == "__main__":
    _finetune_and_store_model()
