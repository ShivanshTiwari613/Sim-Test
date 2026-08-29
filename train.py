"""LoRA SFT of a small base LLM as a next-state predictor.

Loss is computed on the completion (next_state + EOS) only; prompt tokens are
masked with -100. Works on CUDA (Colab T4), Apple MPS, or CPU (slowly).

  python train.py                       # full run
  python train.py --smoke               # 200 examples, 1 epoch, sanity check
"""

import argparse
import json

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from env import PROMPT_TEMPLATE


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class TransitionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=512):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = PROMPT_TEMPLATE.format(state=r["state"], action=r["action"])
        completion = r["next_state"] + self.tok.eos_token
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self.tok(completion, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attention.append([1] * (max_len - pad) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--output-dir", default="checkpoints/qwen05b-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    # Sequences are ~120 tokens and the model is 0.5B — small batches leave a
    # data-center GPU idle on launch overhead. 64 keeps a 4090 busy; drop to 8
    # (with --grad-accum 2) on a T4 or MPS.
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    rows = load_jsonl(args.train_file)
    if args.smoke:
        rows, args.epochs = rows[:200], 1

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=25,
        save_strategy="epoch",
        fp16=use_cuda,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TransitionDataset(rows, tokenizer),
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
