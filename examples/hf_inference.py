"""Hugging Face Transformer LLM Inference on a single AWS Trainium chip.

Prerequisites:
    pip install transformers accelerate
    # Or install example dependencies from repository:
    # pip install -e ".[examples]"

Usage:
    python examples/hf_inference.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

# IMPORTANT: torch_tpu_neuron MUST be imported before torch or transformers.
# This automatically configures:
#   1. AWS Trainium PJRT backend and device registration (torch.accelerator / torch.neuron).
#   2. torch.backends.cuda SDP disabling so Hugging Face PyTorch SDPA lowers to pure StableHLO.
import torch_tpu_neuron

import argparse
import sys
import torch


def parse_args():
  parser = argparse.ArgumentParser(
      description="Run Hugging Face Transformer LLM inference on AWS Trainium"
  )
  parser.add_argument(
      "--model",
      type=str,
      default="Qwen/Qwen2.5-0.5B-Instruct",
      help=(
          "Hugging Face model ID or path. Recommended small ungated public models: "
          "'Qwen/Qwen2.5-0.5B-Instruct' or 'Qwen/Qwen3-0.6B' (no HF account/token required)."
      ),
  )
  parser.add_argument(
      "--prompt",
      type=str,
      default="Explain what AWS Trainium is in two concise sentences.",
      help="Prompt text for generation",
  )
  parser.add_argument(
      "--max-new-tokens",
      type=int,
      default=64,
      help="Maximum number of new tokens to generate (default: 64)",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
  except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please install transformers: pip install transformers accelerate")
    sys.exit(1)

  device = torch.accelerator.current_accelerator()
  props = torch.tpu.get_device_properties()
  print(f"[torch_tpu_neuron] Device: {props.name} ({device})")
  print(f"[torch_tpu_neuron] Loading '{args.model}' (bfloat16)...")

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForCausalLM.from_pretrained(
      args.model,
      torch_dtype=torch.bfloat16,
  ).to(device)
  model.eval()

  if getattr(tokenizer, "chat_template", None):
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
  else:
    formatted_prompt = args.prompt

  inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
  input_len = inputs["input_ids"].shape[1]

  print(f"\n[Prompt]:\n{args.prompt}\n")
  print("[Generating response on AWS Trainium...]")

  with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

  # Decode only the generated response tokens
  generated_tokens = output_ids[0, input_len:]
  response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
  print(f"[Generated Response]:\n{response.strip()}\n")


if __name__ == "__main__":
  main()
