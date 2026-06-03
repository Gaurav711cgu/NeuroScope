"""ReAct-style multi-step agent loop using GPT-2 Small itself.
Greedy decode keeps things deterministic for research reproducibility."""
from __future__ import annotations

import torch

SYSTEM_PROMPT = (
    "You are a reasoning agent. At each step write:\n"
    "Thought: <your reasoning>\n"
    "Action: <search|lookup|calc|answer>\n"
    "Input: <the input>\n"
)


def build_prompt(task: str, step_n: int, history: list[str]) -> str:
    hist = "\n".join(history)
    return f"{SYSTEM_PROMPT}\nTask: {task}\n{hist}\nStep {step_n}:\nThought:"


def greedy_decode(model, prompt: str, max_new: int = 30, stop_at_double_newline: bool = True) -> str:
    tokens = model.to_tokens(prompt)
    out_ids: list[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(tokens, return_type="logits")
            nxt = logits[0, -1].argmax().item()
            out_ids.append(nxt)
            if nxt == model.tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)
            txt = model.tokenizer.decode(out_ids)
            if stop_at_double_newline and "\n\n" in txt:
                break
            if len(txt) > 200:
                break
    return model.tokenizer.decode(out_ids).strip()


def tool_for_output(text: str) -> str | None:
    low = text.lower()
    for kw in ("search", "lookup", "calc", "answer"):
        if f"action: {kw}" in low or low.strip().startswith(f"action: {kw}"):
            return kw
    return None


def steer_decode(
    model,
    prompt: str,
    layer: int,
    feature_id: int,
    alpha: float,
    max_new: int = 40,
    stop_at_double_newline: bool = True
) -> str:
    """Greedy decode with SAE feature steering active in the residual stream."""
    from .loader import get_sae
    sae, _ = get_sae(layer=layer)
    
    device = next(model.parameters()).device
    # W_dec direction shape [d_model]
    feature_dir = sae.W_dec[feature_id].to(device=device, dtype=model.cfg.dtype)
    
    def steer_hook(value, hook):
        # value: [batch, seq, d_model]
        return value + alpha * feature_dir.view(1, 1, -1)
        
    hook_name = f"blocks.{layer}.hook_resid_post"
    tokens = model.to_tokens(prompt).to(device)
    out_ids: list[int] = []
    
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
            for _ in range(max_new):
                logits = model(tokens, return_type="logits")
                nxt = logits[0, -1].argmax().item()
                out_ids.append(nxt)
                if nxt == model.tokenizer.eos_token_id:
                    break
                tokens = torch.cat([tokens, torch.tensor([[nxt]], device=device)], dim=1)
                txt = model.tokenizer.decode(out_ids)
                if stop_at_double_newline and "\n\n" in txt:
                    break
                if len(txt) > 200:
                    break
                    
    return model.tokenizer.decode(out_ids).strip()

