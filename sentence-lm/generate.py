"""
Generate text from a trained HierarchicalLM checkpoint.

The model generates segment by segment:
  1. Encode all prompt segments with the online encoder to get CLS vectors
  2. Optionally predict the next CLS via the JEPA predictor
  3. Decode the next segment token-by-token conditioned on the CLS prefix

Usage:
    python sentence-lm/generate.py --checkpoint sentence-lm/checkpoints/ckpt_XXXXXX.ckpt
    python sentence-lm/generate.py --checkpoint ... --prompt "The economy is struggling."
    python sentence-lm/generate.py --checkpoint ... --segments 4 --tokens 64 --temp 0.8
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2Tokenizer

sys.path.insert(0, "sentence-lm")
from model import HierarchicalLM
from embeddings import CausalAttentionMask
from train import BERT_DIR, GPT2_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPTS = [
    "The researchers discovered a new species of deep-sea fish.",
    "Economic growth slowed in the third quarter as inflation remained high.",
    "She opened the letter and began to read.",
    "The algorithm failed to converge after one thousand iterations.",
]


@torch.no_grad()
def encode_segments(model, bert_tok, segments, device):
    """Encode a list of text segments into CLS vectors (1, N, 768)."""
    enc = bert_tok(
        segments,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)          # (N, L)
    seg_ids = enc["token_type_ids"].to(device) # (N, L)
    cls = model.encoder(ids, seg_ids)[:, 0]    # (N, 768)
    return cls.unsqueeze(0)                    # (1, N, 768)


@torch.no_grad()
def decode_segment(model, gpt_tok, cls_prefix, max_tokens, temperature, top_k, stop_at_eos=True):
    """
    Autoregressively decode one segment conditioned on cls_prefix (1, k, 768).
    Returns decoded string.
    """
    device = cls_prefix.device
    k = cls_prefix.shape[1]

    # start with BOS token
    token_ids = torch.tensor([[gpt_tok.bos_token_id]], device=device)  # (1, 1)

    for _ in range(max_tokens):
        T = token_ids.shape[1]
        decoder_input, _, _ = model.embed(cls_prefix, token_ids)
        mask = CausalAttentionMask.build(k, T, device=device)

        logits, _ = model.decoder(inputs_embeds=decoder_input, attn_mask=mask)
        next_logits = logits[0, -1, :] / temperature  # (vocab,)

        if top_k is not None:
            topk_vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < topk_vals[-1]] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1, 1)

        if stop_at_eos and next_token.item() == gpt_tok.eos_token_id:
            break

        token_ids = torch.cat([token_ids, next_token], dim=1)

    # strip BOS
    generated = token_ids[0, 1:].tolist()
    return gpt_tok.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate(model, bert_tok, gpt_tok, prompt_text, n_segments, max_tokens,
             temperature, top_k, use_jepa, use_no_eos=False):
    """
    Given a prompt string (one or more sentences), generate n_segments continuations.
    """
    # split prompt into segments on sentence boundaries
    import re
    _split = re.compile(r'(?<=[.?!])\s+')
    prompt_segs = [s.strip() for s in _split.split(prompt_text.strip()) if s.strip()]
    if not prompt_segs:
        prompt_segs = [prompt_text.strip()]

    print(f"\nPrompt segments ({len(prompt_segs)}):")
    for i, s in enumerate(prompt_segs):
        print(f"  [{i+1}] {s}")
    print()

    # encode prompt
    cls_vectors = encode_segments(model, bert_tok, prompt_segs, DEVICE)  # (1, N, 768)

    generated_segs = []

    for step in range(n_segments):
        k = cls_vectors.shape[1]  # number of CLS vectors so far

        if use_jepa:
            # predict next CLS via JEPA predictor, append to prefix
            pred_cls = model.jepa_predictor(cls_vectors[:, -1, :])  # (1, 768)
            decode_cls = torch.cat([cls_vectors, pred_cls.unsqueeze(1)], dim=1)
        else:
            decode_cls = cls_vectors

        seg_text = decode_segment(
            model, gpt_tok, decode_cls, max_tokens, temperature, top_k,
            stop_at_eos=not use_no_eos,
        )
        generated_segs.append(seg_text)
        print(f"  [{k+1}] {seg_text}")

        # encode the generated segment and append its CLS to the prefix
        new_cls = encode_segments(model, bert_tok, [seg_text], DEVICE)  # (1, 1, 768)
        cls_vectors = torch.cat([cls_vectors, new_cls], dim=1)

    return generated_segs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt",    type=str, default=None,
                        help="Prompt text. If omitted, runs DEFAULT_PROMPTS.")
    parser.add_argument("--segments",  type=int, default=4,
                        help="Number of segments to generate after the prompt.")
    parser.add_argument("--tokens",    type=int, default=64,
                        help="Max tokens per segment.")
    parser.add_argument("--temp",      type=float, default=0.8)
    parser.add_argument("--top_k",     type=int, default=50)
    parser.add_argument("--jepa",      action="store_true",
                        help="Use JEPA predictor to guide next-segment CLS.")
    parser.add_argument("--no_eos",   action="store_true",
                        help="Disable early stopping at EOS (useful for debugging undertrained models).")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {args.checkpoint}")

    from train import HierarchicalLMLit
    lit = HierarchicalLMLit.load_from_checkpoint(args.checkpoint)
    model = lit.model.to(DEVICE).eval()

    bert_tok = BertTokenizer.from_pretrained(BERT_DIR)
    gpt_tok  = GPT2Tokenizer.from_pretrained(GPT2_DIR)
    gpt_tok.pad_token = gpt_tok.eos_token

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for prompt in prompts:
        print("\n" + "=" * 70)
        generate(
            model, bert_tok, gpt_tok,
            prompt_text=prompt,
            n_segments=args.segments,
            max_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            use_jepa=args.jepa,
            use_no_eos=args.no_eos,
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
