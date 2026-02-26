"""GPT-2 architecture compatibility for the experiment runner.

The shade Model class targets LLaMA-style architectures (self_attn.o_proj,
mlp.down_proj, model.model.layers). GPT-2 uses a different structure
(attn.c_proj, mlp.c_proj, model.transformer.h) with Conv1D layers
instead of Linear.

These helpers monkey-patch Model so the abliteration pipeline works
with GPT-2 for pipeline validation purposes.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, suppress

import torch
import torch.nn.functional as F
from torch.nn import Module


def _gpt2_get_layers(self):
    """Return GPT-2's transformer blocks (model.transformer.h)."""
    from peft import PeftModel

    model = self.model
    if isinstance(model, PeftModel):
        model = model.base_model.model
    return model.transformer.h


def _gpt2_get_layer_modules(self, layer_index):
    """Return GPT-2's abliterable modules per layer."""
    layer = self.get_layers()[layer_index]
    modules: dict[str, list[Module]] = {}

    def try_add(component: str, module):
        if isinstance(module, Module):
            modules.setdefault(component, []).append(module)

    with suppress(AttributeError):
        try_add("attn.c_proj", layer.attn.c_proj)
    with suppress(AttributeError):
        try_add("mlp.c_proj", layer.mlp.c_proj)

    total = sum(len(mods) for mods in modules.values())
    assert total > 0, "No abliterable modules found in layer"
    return modules


def _gpt2_abliterate(self, refusal_directions, direction_index, parameters):
    """Abliterate GPT-2 LoRA adapters (handles Conv1D weight layout)."""
    if direction_index is None:
        refusal_direction = None
    else:
        weight_frac, index = math.modf(direction_index + 1)
        refusal_direction = F.normalize(
            refusal_directions[int(index)].lerp(
                refusal_directions[int(index) + 1], weight_frac
            ),
            p=2,
            dim=0,
        )

    for layer_index in range(len(self.get_layers())):
        for component, modules in self.get_layer_modules(layer_index).items():
            params = parameters[component]
            distance = float(abs(layer_index - params.max_weight_position))
            if distance > params.min_weight_distance:
                continue
            w = params.max_weight + (distance / params.min_weight_distance) * (
                params.min_weight - params.max_weight
            )
            if refusal_direction is None:
                layer_dir = refusal_directions[layer_index + 1]
            else:
                layer_dir = refusal_direction

            for module in modules:
                v = layer_dir.to(next(module.parameters()).device)
                W = module.base_layer.weight.to(torch.float32)
                # Conv1D stores weights as (in_features, out_features);
                # transpose to (out_features, in_features) if needed.
                if W.shape[0] != v.shape[0]:
                    W = W.T
                W = W.reshape(W.shape[0], -1)

                lora_A = (v @ W).view(1, -1)
                lora_B = (-w * v).view(-1, 1)

                wa = module.lora_A["default"].weight
                wb = module.lora_B["default"].weight
                wa.data = lora_A.to(wa.dtype)
                wb.data = lora_B.to(wb.dtype)


@contextmanager
def gpt2_model_patches():
    """Context manager that patches Model methods for GPT-2 compatibility."""
    from unittest.mock import patch

    from shade.model import Model

    with (
        patch.object(Model, "get_layers", _gpt2_get_layers),
        patch.object(Model, "get_layer_modules", _gpt2_get_layer_modules),
        patch.object(Model, "abliterate", _gpt2_abliterate),
    ):
        yield
