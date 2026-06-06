"""
Türkçe nedensel dil modeli (CLM) ince ayarı.

Dosya biçimi: her satır bir metin örneği (UTF-8).

Önerilen taban model: savasy/gpt2-turkish-medium (Türkçe GPT-2 tabanı).
CPU'da yavaş olabilir; mümkünse GPU kullanın.
"""

from __future__ import annotations

import argparse
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def make_group_fn(block_size: int):
    def group_texts(examples):
        keys = [k for k in examples if k != "special_tokens_mask"]
        concatenated = {k: list(chain(*examples[k])) for k in keys}
        total_length = len(concatenated["input_ids"])
        if total_length < block_size:
            return {k: [] for k in keys}
        total_length = (total_length // block_size) * block_size
        return {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}

    return group_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="UTF-8 satır bazlı metin dosyası.")
    parser.add_argument("--model_name", type=str, default="savasy/gpt2-turkish-medium")
    parser.add_argument("--output_dir", type=str, default="output/turkish-lm-ft")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 = tüm satırlar.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("text", data_files={"train": args.corpus}, split="train")
    if args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=True)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        desc="Tokenize",
    )

    group_fn = make_group_fn(args.block_size)
    grouped = tokenized.map(
        group_fn,
        batched=True,
        num_proc=1,
        desc="Blokla",
    )

    if len(grouped) == 0:
        def tok_fixed(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.block_size, padding="max_length")

        grouped = dataset.map(
            tok_fixed,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            desc="Kisa corpus: sabit uzunluk",
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Kaydedildi: {args.output_dir}")


if __name__ == "__main__":
    main()
