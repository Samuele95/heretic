# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
from transformers.models.qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
)

from heretic.model import _get_last_position


class LastPositionTests(unittest.TestCase):
    def test_unflattened_tensor(self) -> None:
        tensor = torch.randn(3, 5, 4)

        self.assertTrue(torch.equal(_get_last_position(tensor, 3), tensor[:, -1, :]))

    def test_flattened_tensor(self) -> None:
        tensor = torch.randn(3, 5, 4)
        flattened = tensor.reshape(-1, 4)

        self.assertTrue(torch.equal(_get_last_position(flattened, 3), tensor[:, -1, :]))

    def test_qwen3_5_moe_shared_expert(self) -> None:
        # The Qwen3.5 MoE block flattens the hidden states before calling the
        # shared expert, so its down_proj receives (prompt * position, component).
        torch.manual_seed(0)
        config = Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            pad_token_id=0,
        )
        model = Qwen3_5MoeForCausalLM(config).eval()
        module = model.model.layers[1].mlp.shared_expert.down_proj

        captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        module.register_forward_hook(
            lambda _module, inputs, outputs: captured.append((inputs[0], outputs))
        )

        def run(input_ids: list[list[int]], attention_mask: list[list[int]]) -> None:
            with torch.no_grad():
                model(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(attention_mask),
                )

        # Left-padded batch, as used by Heretic.
        run([[5, 6, 7, 8, 9], [0, 0, 11, 12, 13]], [[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
        run([[5, 6, 7, 8, 9]], [[1, 1, 1, 1, 1]])
        run([[11, 12, 13]], [[1, 1, 1]])

        self.assertEqual(captured[0][0].dim(), 2)

        for index in range(2):
            batched = _get_last_position(captured[0][index], 2)
            first = _get_last_position(captured[1][index], 1)
            second = _get_last_position(captured[2][index], 1)

            self.assertTrue(torch.allclose(batched[0], first[0], atol=1e-5))
            self.assertTrue(torch.allclose(batched[1], second[0], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
