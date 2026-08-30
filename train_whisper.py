"""
Fine-tune Whisper on the Macedonian dataset produced by build_stt_dataset_v2_new.py.

Install first:
    .venv/bin/pip install "transformers>=4.44" "datasets>=2.20" accelerate \
        evaluate jiwer torch soundfile librosa tensorboard

Then:
    .venv/bin/python train_whisper.py

Notes
-----
* Uses dataset_v2_new/metadata.csv (the GOLD split) by default. Set INCLUDE_SILVER=1
  in the environment to also train on the silver clips.
* Base checkpoint defaults to openai/whisper-small. For real quality on
  Macedonian use whisper-large-v3, but that needs a GPU.
* The `split` column decides train vs. test.
"""

import os
from pathlib import Path

import evaluate
import torch
from datasets import Audio, Dataset, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset_v2_new"
BASE_MODEL = os.environ.get("BASE_MODEL", "openai/whisper-small")
OUTPUT_DIR = ROOT / "whisper-mk-finetuned"
LANGUAGE = "macedonian"
TASK = "transcribe"
INCLUDE_SILVER = os.environ.get("INCLUDE_SILVER") == "1"

import csv


def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = DATA_DIR / r["file_name"]
            if p.exists() and r["transcription"].strip():
                rows.append({"audio": str(p),
                             "sentence": r["transcription"].strip(),
                             "split": r.get("split", "train")})
    return rows


def build_dataset():
    rows = load_rows(DATA_DIR / "metadata.csv")
    if INCLUDE_SILVER and (DATA_DIR / "metadata_silver.csv").exists():
        rows += load_rows(DATA_DIR / "metadata_silver.csv")

    train = [r for r in rows if r["split"] != "test"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"train={len(train)}  test={len(test)}")

    ds_train = Dataset.from_list(train).cast_column("audio", Audio(sampling_rate=16000))
    ds_test = Dataset.from_list(test).cast_column("audio", Audio(sampling_rate=16000))
    return ds_train, ds_test


def main():
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    ds_train, ds_test = build_dataset()

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    ds_train = ds_train.map(prepare, remove_columns=ds_train.column_names)
    ds_test = ds_test.map(prepare, remove_columns=ds_test.column_names)

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
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=1000,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=Collator(processor=processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
