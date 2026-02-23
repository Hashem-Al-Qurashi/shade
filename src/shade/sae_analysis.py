"""SAE-based decomposition of refusal directions.

Decomposes abliteration refusal directions into interpretable sparse
autoencoder features using SAELens. Shows what the refusal direction
overlaps with (math, code, facts, etc.).
"""

from __future__ import annotations

import torch


def decompose_direction_into_features(
    direction: torch.Tensor,
    sae: object,
    top_k: int = 20,
) -> list[dict]:
    """Decompose a direction vector into SAE features.

    Args:
        direction: The refusal direction vector, shape (d_model,).
        sae: An SAELens SAE instance with .encode() method.
        top_k: Number of top-activating features to return.

    Returns:
        List of dicts with 'index' and 'activation' keys, sorted by activation.
    """
    # SAE expects (batch, seq, d_model)
    acts = sae.encode(direction.unsqueeze(0).unsqueeze(0))
    acts_flat = acts.squeeze()

    topk = acts_flat.abs().topk(top_k)
    return [
        {"index": idx.item(), "activation": acts_flat[idx].item()}
        for idx, val in zip(topk.indices, topk.values)
    ]


def compute_refusal_direction(
    model: object,
    tokenizer: object,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer_index: int,
) -> torch.Tensor:
    """Compute the mean-difference refusal direction at a specific layer.

    Args:
        model: HuggingFace model.
        tokenizer: Model tokenizer.
        harmful_prompts: Prompts that elicit refusal.
        harmless_prompts: Prompts that elicit compliance.
        layer_index: Which transformer layer to extract from.

    Returns:
        Normalized refusal direction vector, shape (d_model,).
    """
    def get_activations(prompts: list[str]) -> torch.Tensor:
        acts = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Take last token's hidden state at target layer
            hidden = outputs.hidden_states[layer_index + 1]  # +1 for embedding layer
            acts.append(hidden[0, -1, :])  # last token
        return torch.stack(acts)

    harmful_acts = get_activations(harmful_prompts)
    harmless_acts = get_activations(harmless_prompts)

    direction = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
    return direction / direction.norm()


def load_sae_for_model(
    model_name: str,
    layer_index: int,
    width: str = "16k",
) -> object:
    """Load a pre-trained SAE from SAELens registry.

    Supported models:
    - gemma-2-2b: gemma-scope-2b-pt-res-canonical
    - gpt2: gpt2-small-res-jb

    Args:
        model_name: HuggingFace model name.
        layer_index: Target layer.
        width: SAE width (e.g., "16k", "65k").

    Returns:
        SAELens SAE instance.
    """
    from sae_lens import SAE

    # Map model names to SAELens release IDs
    sae_registry = {
        "google/gemma-2-2b": {
            "release": "gemma-scope-2b-pt-res-canonical",
            "sae_id": f"layer_{layer_index}/width_{width}/canonical",
        },
        "openai-community/gpt2": {
            "release": "gpt2-small-res-jb",
            "sae_id": f"blocks.{layer_index}.hook_resid_post",
        },
    }

    if model_name not in sae_registry:
        raise ValueError(
            f"No pre-trained SAE available for {model_name}. "
            f"Supported: {list(sae_registry.keys())}"
        )

    config = sae_registry[model_name]
    return SAE.from_pretrained(
        release=config["release"],
        sae_id=config["sae_id"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def run_sae_analysis(
    model: object,
    tokenizer: object,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    model_name: str,
    layer_index: int,
    top_k: int = 20,
) -> dict:
    """Full SAE analysis pipeline.

    1. Compute refusal direction at target layer
    2. Load pre-trained SAE
    3. Decompose into features
    4. Return results dict

    Returns:
        Dict with direction, features, and metadata.
    """
    direction = compute_refusal_direction(
        model, tokenizer, harmful_prompts, harmless_prompts, layer_index,
    )

    sae = load_sae_for_model(model_name, layer_index)
    features = decompose_direction_into_features(direction, sae, top_k=top_k)

    return {
        "model_name": model_name,
        "layer_index": layer_index,
        "direction_norm": direction.norm().item(),
        "top_features": features,
        "n_harmful_prompts": len(harmful_prompts),
        "n_harmless_prompts": len(harmless_prompts),
    }
