#!/usr/bin/env python3
import argparse
import datetime
import json
import shutil
from pathlib import Path

SCHEME_MAP = {
    "fp8": "FP8_DYNAMIC",
    "nvfp4": "NVFP4",
}

IGNORE_MODULES = ["lm_head", "re:.*vision.*", "re:.*audio.*"]

BUILTIN_CALIB_TEXTS = [
    "Explain how coherent unified memory on Grace Blackwell affects dense transformer token generation.",
    "Schreiben Sie eine kurze Zusammenfassung ueber die Schweizerische Eidgenossenschaft und ihre vier Landessprachen.",
    "Redigez une fonction Python robuste qui parse un flux JSON Lines sans charger tout le fichier en memoire.",
    "Scrivi una query SQL che calcola la latenza mediana e il percentile p99 per ciascun nodo del cluster.",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "Step 1: Measure single-stream decode bandwidth. Step 2: Compare BF16 weight size against FP8 and NVFP4. Step 3: Verify speculative acceptance rate.",
    "#!/usr/bin/env bash\nset -euo pipefail\nip -4 -o addr show dev enp1s0f1np1 | awk '{print $4}'",
    "In distributed tensor parallelism of degree 2, each linear layer shards either column-wise or row-wise followed by an all-reduce across the 200GbE link.",
]


def is_already_quantized(out_dir: Path, model_id: str, resolved_scheme: str) -> bool:
    meta_file = out_dir / "apertux_quant_meta.json"
    if not meta_file.is_file():
        return False
    if not any(out_dir.glob("*.safetensors")):
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except ValueError:
        return False
    return meta.get("source_model") == model_id and meta.get("scheme") == resolved_scheme


def load_calib_texts(tokenizer, calib_file: str | None, count: int) -> list[str]:
    texts: list[str] = []
    if calib_file:
        for raw in Path(calib_file).read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if "messages" in item:
                texts.append(tokenizer.apply_chat_template(item["messages"], tokenize=False))
            elif "text" in item:
                texts.append(str(item["text"]))
    if not texts:
        texts = list(BUILTIN_CALIB_TEXTS)
    expanded: list[str] = []
    while len(expanded) < count:
        for idx, base in enumerate(texts):
            expanded.append(f"[sample {len(expanded) + 1}/{count}] {base}")
            if len(expanded) >= count:
                break
    return expanded


def build_calib_dataset(tokenizer, calib_file: str | None, count: int, max_seq_len: int):
    from datasets import Dataset

    texts = load_calib_texts(tokenizer, calib_file, count)
    rows = [
        tokenizer(
            t,
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )
        for t in texts
    ]
    return Dataset.from_list(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quantize Apertus checkpoints offline for vLLM.")
    parser.add_argument("--model", required=True, help="Cached Hugging Face repo ID or local directory")
    parser.add_argument("--output-dir", required=True, help="Target directory for compressed checkpoint")
    parser.add_argument("--scheme", choices=["fp8", "nvfp4"], default="nvfp4")
    parser.add_argument("--calib-file", default=None, help="Optional local JSONL calibration file")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--gpu-gib", default="28GiB")
    parser.add_argument("--cpu-gib", default="64GiB")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    resolved_scheme = SCHEME_MAP[args.scheme]
    out_dir = Path(args.output_dir)

    if not args.force and is_already_quantized(out_dir, args.model, resolved_scheme):
        print(f"Reusing existing {resolved_scheme} checkpoint at {out_dir}")
        return 0

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    offload_dir = out_dir / ".offload_tmp"
    offload_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={0: args.gpu_gib, "cpu": args.cpu_gib},
            offload_folder=str(offload_dir),
            local_files_only=True,
            trust_remote_code=True,
        )

        recipe = QuantizationModifier(
            targets="Linear",
            scheme=resolved_scheme,
            ignore=list(IGNORE_MODULES),
        )

        calib_ds = None
        if resolved_scheme == "NVFP4":
            calib_ds = build_calib_dataset(tokenizer, args.calib_file, args.calib_samples, args.max_seq_len)

        oneshot(
            model=model,
            recipe=recipe,
            dataset=calib_ds,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.calib_samples if calib_ds is not None else 0,
        )

        model.save_pretrained(str(out_dir), save_compressed=True)
        tokenizer.save_pretrained(str(out_dir))

        meta = {
            "source_model": args.model,
            "scheme": resolved_scheme,
            "ignore": IGNORE_MODULES,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (out_dir / "apertux_quant_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    finally:
        shutil.rmtree(offload_dir, ignore_errors=True)

    print(f"Saved {resolved_scheme} checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
