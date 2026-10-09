#!/usr/bin/env python3
"""One-shot smoke test: load the reconstructed Qwen checkpoint and generate one reply.

This deliberately runs a single short prompt. It is not a benchmark or a quality
suite; success means the checkpoint loads and the model emits non-empty text.
"""
from __future__ import annotations

import argparse
import sys
import time
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
        from transformers import AutoProcessor
    except Exception as error:
        print("[ERRO] Falha ao importar PyTorch ou Transformers.", file=sys.stderr)
        print("Execute: python -m pip install -U torch transformers accelerate", file=sys.stderr)
        print(f"Detalhe: {type(error).__name__}: {error}", file=sys.stderr)
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
        print(f"[2/3] Carregando modelo (dtype={args.dtype}, device_map={args.device_map})...", flush=True)
        model = model_loader.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
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
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        first_device = next(model.parameters()).device
        inputs = {key: value.to(first_device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max(1, args.max_new_tokens),
                do_sample=False,
                use_cache=True,
            )
        input_length = inputs["input_ids"].shape[-1]
        answer = processor.decode(
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
