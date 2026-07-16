"""Inject / swap a streamed LoRA adapter into a live policy.

Shared by ``LoRAPolicyServer`` and the standalone example subscriber so the two can
never drift. The first adapter injects the LoRA layers into the policy in place; every
adapter after that is a cheap in-place tensor overwrite.
"""

import logging

logger = logging.getLogger("lora_adapter_applier")


class AdapterApplier:
    """Holds the ``PeftModel`` handle and applies adapters from a local dir."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.peft_model = None

    def apply(self, policy, adapter_path: str, version: int | None = None) -> str:
        """Apply the adapter at ``adapter_path`` to ``policy``. Returns "injected"/"swapped"."""
        from peft import PeftModel, load_peft_weights, set_peft_model_state_dict

        if self.peft_model is None:
            # First adapter: PEFT rewrites the targeted submodules in place, so
            # ``policy`` keeps working and now routes through the adapters.
            self.peft_model = PeftModel.from_pretrained(policy, adapter_path, is_trainable=False)
            self.peft_model.eval()
            policy.eval()
            return "injected"

        # Steady state: overwrite adapter tensors, touching nothing else.
        state_dict = load_peft_weights(adapter_path, device=str(self.device))
        result = set_peft_model_state_dict(self.peft_model, state_dict)
        unexpected = getattr(result, "unexpected_keys", None)
        if unexpected:
            logger.warning(f"Adapter v{version} had unexpected keys: {unexpected}")
        return "swapped"
