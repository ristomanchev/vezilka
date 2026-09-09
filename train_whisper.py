import os
import re
import csv
from pathlib import Path

import evaluate
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset_v2_new"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "whisper-mk-finetuned"))
RESUME = os.environ.get("RESUME") == "1"
BASE_MODEL = os.environ.get("BASE_MODEL", "openai/whisper-small")
LANGUAGE = "macedonian"
TASK = "transcribe"
INCLUDE_SILVER = os.environ.get("INCLUDE_SILVER") == "1"
EXTRA_DATA = [s.strip().lower() for s in os.environ.get("EXTRA_DATA", "fleurs").split(",") if s.strip()]
MAX_STEPS = int(os.environ.get("MAX_STEPS", 1000))
DROPOUT = float(os.environ.get("DROPOUT", 0.1))
LR = float(os.environ.get("LR", 1e-5))


_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def norm_text(s):
    s = (s or "").lower().strip()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_our_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = DATA_DIR / r["file_name"]
            if p.exists() and r["transcription"].strip():
                rows.append({"audio": str(p),
                             "sentence": r["transcription"].strip(),
                             "split": r.get("split", "train")})
    return rows


def our_datasets():
    rows = load_our_rows(DATA_DIR / "metadata.csv")
    if INCLUDE_SILVER and (DATA_DIR / "metadata_silver.csv").exists():
        rows += load_our_rows(DATA_DIR / "metadata_silver.csv")
    train = [{"audio": r["audio"], "sentence": r["sentence"]} for r in rows if r["split"] != "test"]
    test = [{"audio": r["audio"], "sentence": r["sentence"]} for r in rows if r["split"] == "test"]
    mk = lambda lst: Dataset.from_list(lst).cast_column("audio", Audio(sampling_rate=16000))
    return mk(train), mk(test)


def load_extra_train():
    """Extra Macedonian training data. Returns a list of Datasets with
    columns {audio (16 kHz), sentence}. Failures are warned, not fatal."""
    out = []
    for src in EXTRA_DATA:
        if src == "none":
            continue
        try:
            if src == "fleurs":
                d = load_dataset("google/fleurs", "mk_mk", split="train")
                d = d.select_columns(["audio", "raw_transcription"]).rename_column("raw_transcription", "sentence")
            elif src == "commonvoice":
                d = load_dataset("mozilla-foundation/common_voice_17_0", "mk", split="train")
                d = d.select_columns(["audio", "sentence"])
            else:
                print(f"[extra] unknown source '{src}', skipping")
                continue
            d = d.cast_column("audio", Audio(sampling_rate=16000))
            d = d.filter(lambda s: bool(s and s.strip()), input_columns=["sentence"])
            print(f"[extra] {src}: {len(d)} clips")
            out.append(d)
        except Exception as e:
            print(f"[extra] could not load '{src}': {e}\n"
                  f"        (commonvoice needs `huggingface-cli login` + accepting its terms)")
    return out


def auto_batch(model_name):
    n = model_name.lower()
    if "large" in n:
        return 2, 8
    if "medium" in n:
        return 4, 4
    return 8, 2


def main():
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    ds_train_ours, ds_test = our_datasets()
    extra = load_extra_train()
    print(f"our train={len(ds_train_ours)}  test={len(ds_test)}  extra sources={len(extra)}")

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    proc = lambda d: d.map(prepare, remove_columns=d.column_names,
                           num_proc=os.cpu_count() if len(d) > 500 else 1)

    train_parts = [proc(ds_train_ours)] + [proc(d) for d in extra]
    ds_train = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
    ds_train = ds_train.shuffle(seed=42)
    ds_test = proc(ds_test)
    print(f"TOTAL train examples: {len(ds_train)}")

    import dataclasses
    from typing import Any

    @dataclasses.dataclass
    class Collator:
        processor: Any

        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_raw = 100 * metric.compute(predictions=pred_str, references=label_str)

        pairs = [(norm_text(p), norm_text(l)) for p, l in zip(pred_str, label_str)]
        pairs = [(p, l) for p, l in pairs if l]
        wer_norm = 100 * metric.compute(
            predictions=[p for p, _ in pairs], references=[l for _, l in pairs]
        )
        return {"wer": round(wer_norm, 2), "wer_raw": round(wer_raw, 2)}

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, dropout=DROPOUT, attention_dropout=DROPOUT
    )
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    ckpt_steps = max(5, min(250, MAX_STEPS // 4))
    bs, ga = auto_batch(BASE_MODEL)
    bs = int(os.environ.get("BATCH", bs))

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        per_device_eval_batch_size=bs,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=min(80, MAX_STEPS // 4),
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=ckpt_steps,
        eval_steps=ckpt_steps,
        logging_steps=max(1, ckpt_steps // 10),
        save_total_limit=2,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer_kwargs = dict(
        args=args,
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=Collator(processor=processor),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    import inspect as _inspect
    if "processing_class" in _inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    resume = RESUME or (OUTPUT_DIR.exists() and any(OUTPUT_DIR.glob("checkpoint-*")))
    if resume:
        print(f"\n=== resuming from last checkpoint in {OUTPUT_DIR} ===")
    else:
        print("\n=== baseline eval (before fine-tuning) ===")
        print(trainer.evaluate())

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("\n=== final eval (best checkpoint) ===")
    print(trainer.evaluate())
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
