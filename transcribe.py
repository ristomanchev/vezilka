import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dataset_label(path):
    full = ROOT / "dataset_v2_new" / "metadata_full.csv"
    if not full.exists():
        return None
    name = Path(path).name
    with open(full, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["file_name"].endswith(name):
                return r["text"], r["source"]
    return None


def run_faster_whisper(path, model_size):
    from faster_whisper import WhisperModel
    print(f"[faster-whisper {model_size}, cpu int8]")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), language="mk", beam_size=5,
                                      vad_filter=True)
    print(f"detected language prob: {info.language_probability:.2f}\n")
    out = []
    for s in segments:
        print(f"[{s.start:6.2f} -> {s.end:6.2f}]  {s.text.strip()}")
        out.append(s.text.strip())
    return " ".join(out)


def run_hf(path, model_dir):
    import torch
    from transformers import pipeline
    print(f"[HF model: {model_dir}]")
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=model_dir, device=device,
                   chunk_length_s=30, return_timestamps=False,
                   generate_kwargs={"language": "macedonian", "task": "transcribe"})
    text = asr(str(path))["text"].strip()
    print(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper size: tiny/base/small/medium/large-v3")
    ap.add_argument("--hf", default="", help="path to a fine-tuned HF model dir")
    ap.add_argument("--label", action="store_true",
                    help="also print the dataset label for this clip")
    args = ap.parse_args()

    if not Path(args.file).exists():
        sys.exit(f"no such file: {args.file}")

    if args.hf:
        hyp = run_hf(args.file, args.hf)
    else:
        hyp = run_faster_whisper(args.file, args.model)

    if args.label:
        lbl = dataset_label(args.file)
        print("\n" + "-" * 50)
        if lbl:
            print(f"dataset label ({lbl[1]}): {lbl[0]}")
            print(f"model output        : {hyp}")
        else:
            print("no dataset label found for this file")


if __name__ == "__main__":
    main()
