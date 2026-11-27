import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class FakeModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, out_dir, save_compressed=True):
        self.saved_to = Path(out_dir)
        self.saved_to.mkdir(parents=True, exist_ok=True)
        (self.saved_to / "model-00001-of-00008.safetensors").write_bytes(b"quantized")


class FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def apply_chat_template(self, messages, tokenize=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, padding=False, max_length=2048, truncation=True, add_special_tokens=False):
        tokens = [1, 2, 3, 4][:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def save_pretrained(self, out_dir):
        self.saved_to = Path(out_dir)
        (self.saved_to / "tokenizer.json").write_text("{}")


class QuantizeApertusTests(unittest.TestCase):
    def setUp(self):
        self.calls = {"from_pretrained": [], "modifiers": [], "oneshot": []}
        self.fake_model = FakeModel()
        self.fake_tok = FakeTokenizer()

        transformers_mod = types.ModuleType("transformers")

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                self.calls["from_pretrained"].append((model_id, kwargs))
                if "offload_folder" in kwargs:
                    Path(kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
                    (Path(kwargs["offload_folder"]) / "tmp.bin").write_bytes(b"x")
                return self.fake_model

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                return self.fake_tok

        transformers_mod.AutoModelForCausalLM = AutoModelForCausalLM
        transformers_mod.AutoTokenizer = AutoTokenizer

        llmcomp_mod = types.ModuleType("llmcompressor")
        modifiers_pkg = types.ModuleType("llmcompressor.modifiers")
        quant_mod = types.ModuleType("llmcompressor.modifiers.quantization")

        def QuantizationModifier(**kwargs):
            self.calls["modifiers"].append(kwargs)
            return kwargs

        def oneshot(**kwargs):
            self.calls["oneshot"].append(kwargs)

        llmcomp_mod.oneshot = oneshot
        quant_mod.QuantizationModifier = QuantizationModifier

        datasets_mod = types.ModuleType("datasets")

        class Dataset:
            @staticmethod
            def from_list(rows):
                return rows

        datasets_mod.Dataset = Dataset

        self._orig_modules = {
            k: sys.modules.get(k)
            for k in [
                "transformers",
                "llmcompressor",
                "llmcompressor.modifiers",
                "llmcompressor.modifiers.quantization",
                "datasets",
            ]
        }
        sys.modules["transformers"] = transformers_mod
        sys.modules["llmcompressor"] = llmcomp_mod
        sys.modules["llmcompressor.modifiers"] = modifiers_pkg
        sys.modules["llmcompressor.modifiers.quantization"] = quant_mod
        sys.modules["datasets"] = datasets_mod

    def tearDown(self):
        for k, v in self._orig_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_fp8_dynamic_memory_safe_and_cleanup(self):
        import quantize_apertus

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "Apertus-70B-FP8"
            rc = quantize_apertus.main([
                "--model", "swiss-ai/Apertus-v1.5-70B",
                "--output-dir", str(out_dir),
                "--scheme", "fp8",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(len(self.calls["from_pretrained"]), 1)
            _, load_kw = self.calls["from_pretrained"][0]
            self.assertTrue(load_kw.get("local_files_only"))
            self.assertTrue(load_kw.get("low_cpu_mem_usage"))
            self.assertEqual(load_kw.get("max_memory"), {0: "28GiB", "cpu": "64GiB"})
            self.assertFalse((out_dir / ".offload_tmp").exists())

            mod_kw = self.calls["modifiers"][0]
            self.assertEqual(mod_kw["scheme"], "FP8_DYNAMIC")
            self.assertIn("lm_head", mod_kw["ignore"])
            self.assertIsNone(self.calls["oneshot"][0].get("dataset"))

            meta = json.loads((out_dir / "apertux_quant_meta.json").read_text())
            self.assertEqual(meta["scheme"], "FP8_DYNAMIC")
            self.assertEqual(meta["source_model"], "swiss-ai/Apertus-v1.5-70B")

            # Idempotent second call skips reload unless --force
            rc2 = quantize_apertus.main([
                "--model", "swiss-ai/Apertus-v1.5-70B",
                "--output-dir", str(out_dir),
                "--scheme", "fp8",
            ])
            self.assertEqual(rc2, 0)
            self.assertEqual(len(self.calls["from_pretrained"]), 1)

    def test_nvfp4_uses_offline_calibration_and_supports_custom_jsonl(self):
        import quantize_apertus

        with tempfile.TemporaryDirectory() as tmp:
            calib_jsonl = Path(tmp) / "calib.jsonl"
            calib_jsonl.write_text(
                json.dumps({"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Gruezi"}]}) + "\n"
                + json.dumps({"text": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"}) + "\n"
            )
            out_dir = Path(tmp) / "Apertus-70B-NVFP4"
            rc = quantize_apertus.main([
                "--model", "swiss-ai/Apertus-v1.5-70B",
                "--output-dir", str(out_dir),
                "--scheme", "nvfp4",
                "--calib-file", str(calib_jsonl),
                "--calib-samples", "16",
            ])
            self.assertEqual(rc, 0)
            mod_kw = self.calls["modifiers"][0]
            self.assertEqual(mod_kw["scheme"], "NVFP4")
            oneshot_kw = self.calls["oneshot"][0]
            self.assertIsNotNone(oneshot_kw.get("dataset"))
            self.assertEqual(len(oneshot_kw["dataset"]), 16)
            self.assertTrue((out_dir / "model-00001-of-00008.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
