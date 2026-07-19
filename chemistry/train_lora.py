"""train_lora — LoRA-дообучение Qwen3.5-4B на датасете станка (CHEM-ds01).

Тот же чекпойнт, что у baseline-экзамена (Qwen/Qwen3.5-4B): сравнение чистое.
bf16 LoRA r=16, alpha=r, на все линейные слои; полный FT в 40GB не влезает,
QLoRA для Qwen3.5 не рекомендован (потери квантования).

Формат — чат-шаблон с enable_thinking=False: тот же язык, что на экзамене
(llama.cpp --jinja, думание заглушено). Лосс — только на ход ассистента
(train_on_responses_only): вопрос модель читает, отвечать — учится.

    HF_HOME=/workspace/.hf_home python3 train_lora.py \
        [--data /workspace/chem/dataset] [--out /workspace/chem/out] \
        [--epochs 2] [--lr 1e-4] [--bs 16] [--accum 2]
"""

import json
import sys

from unsloth import FastLanguageModel                     # до transformers
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
from datasets import Dataset                               # noqa: E402
from trl import SFTConfig, SFTTrainer                      # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
SEED = 7


def arg(flag, default):
    a = sys.argv[1:]
    return a[a.index(flag) + 1] if flag in a else default


def load_texts(path, tokenizer):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    texts = [tokenizer.apply_chat_template(
        r["messages"], tokenize=False, enable_thinking=False) for r in rows]
    return Dataset.from_dict({"text": texts})


def main():
    data = arg("--data", "/workspace/chem/dataset")
    out = arg("--out", "/workspace/chem/out")
    epochs = float(arg("--epochs", "2"))
    lr = float(arg("--lr", "1e-4"))
    bs = int(arg("--bs", "16"))
    accum = int(arg("--accum", "2"))

    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=1024, load_in_4bit=False, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)

    train = load_texts(f"{data}/train.jsonl", tokenizer)
    test = load_texts(f"{data}/test.jsonl", tokenizer)
    evals = test.shuffle(seed=SEED).select(range(512))
    print(f"train {len(train)}, eval {len(evals)} (из test {len(test)})")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train, eval_dataset=evals,
        args=SFTConfig(
            dataset_text_field="text", max_length=1024,
            per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
            num_train_epochs=epochs, learning_rate=lr,
            lr_scheduler_type="linear", warmup_ratio=0.03,
            optim="adamw_8bit", weight_decay=0.01, bf16=True,
            logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
            seed=SEED, output_dir=f"{out}/ckpt", report_to="none"))
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")

    trainer.train()

    model.save_pretrained(f"{out}/lora")
    tokenizer.save_pretrained(f"{out}/lora")
    model.save_pretrained_merged(f"{out}/merged", tokenizer,
                                 save_method="merged_16bit")
    print(f"готово: {out}/lora (адаптер), {out}/merged (bf16 для GGUF)")


if __name__ == "__main__":
    main()
