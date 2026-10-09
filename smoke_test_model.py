#!/usr/bin/env python3
"""One-shot smoke test: load the reconstructed Qwen checkpoint and generate one reply.

This deliberately runs a single short prompt. It is not a benchmark or a quality
suite; success means the checkpoint loads and the model emits non-empty text.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Directory containing reconstructed Safetensors and config")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="float16",
                        help="Runtime dtype. float16 is usually more compatible with consumer GPUs.")
    parser.add_argument("--device-map", default="auto", help="Transformers device map; default auto can offload to CPU")
    parser.add_argument("--prompt", default="Olá! Responda em português, em uma frase curta: o modelo reconstruído está funcionando?")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"[ERRO] Diretório do modelo não encontrado: {model_dir}", file=sys.stderr)
        return 2
    if not (model_dir / "config.json").is_file():
        print(f"[ERRO] config.json não encontrado em {model_dir}", file=sys.stderr)
        return 2

    try:
        import torch
        import transformers
    except Exception as error:
        print("[ERRO] Falha ao importar PyTorch ou Transformers.", file=sys.stderr)
        print(f"Detalhe: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exception(error)
        return 3

    def package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "não instalado"

    # AutoProcessor is a lazy import in Transformers. Its short error message
    # can hide the dependency failure (often an incompatible torchvision build).
    try:
        AutoProcessor = getattr(transformers, "AutoProcessor")
    except Exception as error:
        print("[ERRO] A importação de AutoProcessor falhou; abaixo está a causa original.", file=sys.stderr)
        print(f"Python: {sys.version.split()[0]}", file=sys.stderr)
        for package in ("torch", "torchvision", "torchaudio", "transformers", "accelerate", "huggingface-hub"):
            print(f"{package}: {package_version(package)}", file=sys.stderr)
        print(f"CUDA visível pelo PyTorch: {getattr(torch.version, 'cuda', None)}", file=sys.stderr)
        traceback.print_exception(error)
        return 3

    # Try the current Qwen3.5 API, then the compatible image-text-to-text alias.
    model_loader = None
    loader_errors = []
    for loader_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        try:
            model_loader = getattr(transformers, loader_name)
            print(f"Classe de carregamento: {loader_name}", flush=True)
            break
        except Exception as error:
            loader_errors.append((loader_name, error))

    if model_loader is None:
        print("[ERRO] Nenhuma classe de carregamento multimodal compatível pôde ser importada.", file=sys.stderr)
        print(f"Versão do Transformers: {getattr(transformers, '__version__', 'desconhecida')}", file=sys.stderr)
        print("Execute: python -m pip install -U transformers accelerate", file=sys.stderr)
        for loader_name, error in loader_errors:
            print(f"  {loader_name}: {type(error).__name__}: {error}", file=sys.stderr)
        return 3

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    started = time.perf_counter()
    print(f"[1/3] Carregando processador de: {model_dir}", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

        print(f"[2/3] Carregando modelo (dtype={args.dtype}, device_map={args.device_map})...", flush=True)
        model = model_loader.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map=args.device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        print("[3/3] Gerando uma única resposta...", flush=True)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
        }]

        # Prefer the model's own chat template. Some reconstructed folders do
        # not contain a tokenizer_config.json with chat_template, so fall back
        # to the official Qwen message tokens for this text-only smoke test.
        if getattr(processor, "chat_template", None):
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        elif getattr(tokenizer, "chat_template", None):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            print("[AVISO] O tokenizer local não possui chat_template; usando os tokens de conversa Qwen.", flush=True)
            formatted_prompt = (
                "<|im_start|>user\\n"
                + args.prompt
                + "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"
            )
            inputs = tokenizer(formatted_prompt, return_tensors="pt")

        # With device_map='auto', some parameters may be offloaded and appear
        # on the meta device. Place inputs according to the input embedding,
        # not the first arbitrary parameter in the model.
        input_device = model.get_input_embeddings().weight.device
        if input_device.type == "meta":
            device_map = getattr(model, "hf_device_map", {})
            embed_devices = [
                device for name, device in device_map.items()
                if "embed_tokens" in name
            ]
            if embed_devices:
                mapped_device = embed_devices[0]
                if mapped_device == "disk":
                    input_device = torch.device("cpu")
                elif isinstance(mapped_device, int):
                    input_device = torch.device(f"cuda:{mapped_device}" if torch.cuda.is_available() else "cpu")
                else:
                    input_device = torch.device(mapped_device)
            else:
                input_device = next(
                    (parameter.device for parameter in model.parameters() if parameter.device.type != "meta"),
                    torch.device("cpu"),
                )
        inputs = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max(1, args.max_new_tokens),
                do_sample=False,
                use_cache=True,
            )
        input_length = inputs["input_ids"].shape[-1]
        answer = tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        elapsed = time.perf_counter() - started
        if not answer:
            print("\n[FAIL] O modelo carregou, mas não gerou texto visível.")
            print(f"Tempo total: {elapsed:.1f}s")
            return 1
        print("\n=== RESPOSTA DO MODELO ===")
        print(answer)
        print("\n[PASS] O checkpoint foi carregado e gerou uma resposta.")
        print(f"Tempo total (carregamento + geração): {elapsed:.1f}s")
        print("Este é apenas um teste básico de funcionamento, não uma validação de qualidade.")
        return 0
    except Exception as error:
        print("\n[FAIL] O teste de resposta falhou.", file=sys.stderr)
        print(f"Diretório: {model_dir}", file=sys.stderr)
        print(f"Erro: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nCopie o traceback completo acima para investigar a causa.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
