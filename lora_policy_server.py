"""Variant of lerobot's policy_server.py that hot-reloads LoRA adapters.

Identical to the stock ``PolicyServer`` except that it watches ``adapter_dir`` for
adapters published by ``learner.py`` and swaps them into the running policy
without a restart. The base policy is loaded once, at handshake, exactly as
before; only the adapter weights (a few MB) are ever reloaded.

Two things make the swap safe:

* It happens at an action-chunk boundary, on the inference thread, before the
  observation is preprocessed -- never in the middle of a forward pass.
* The learner publishes atomically (rename-into-place), so a partially written
  adapter is never visible here.

Set ``reload_on=handshake`` to restrict swaps to client (re)connections, i.e.
between episodes, if you don't want the policy changing mid-rollout.

Usage:
    python -m lora_finetuning.lora_policy_server \
        --host=0.0.0.0 --port=8080 --fps=30 \
        --adapter_dir=/path/to/adapters \
        --reload_on=chunk
"""

import logging
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import grpc

from lerobot.async_inference.helpers import TimedAction, TimedObservation, get_logger
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc  # type: ignore
from lerobot.utils.import_utils import register_third_party_plugins

from .adapter_store import AdapterVersion, AdapterWatcher
from .configs import LoRAPolicyServerConfig


class LoRAPolicyServer(PolicyServer):
    prefix = "lora_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: LoRAPolicyServerConfig):
        super().__init__(config)
        self.config: LoRAPolicyServerConfig = config

        self.adapter_root = Path(config.adapter_dir).expanduser() if config.adapter_dir else None
        self._watcher = AdapterWatcher(self.adapter_root) if self.adapter_root else None
        self._peft_model = None
        self._adapter_lock = threading.Lock()
        self.adapter_version: AdapterVersion | None = None

        if self._watcher is None:
            self.logger.info("No adapter_dir set: running as a stock PolicyServer (no hot-reload)")
        else:
            self.logger.info(
                f"Watching {self.adapter_root} for LoRA adapters (reload_on={config.reload_on})"
            )

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Load the base policy (stock behavior), then attach the current adapter."""
        response = super().SendPolicyInstructions(request, context)
        self._sync_adapter()
        return response

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Chunk boundary: the one place it is safe to change the policy's weights."""
        if self.config.reload_on == "chunk":
            self._sync_adapter()
        return super()._predict_action_chunk(observation_t)

    def _sync_adapter(self) -> None:
        """Load a newer adapter if the learner has published one. Never raises."""
        if self._watcher is None or self.policy is None:
            return

        meta = self._watcher.poll()
        if meta is None:
            return

        with self._adapter_lock:
            try:
                self._apply_adapter(meta)
            except Exception as e:
                # A bad adapter must not take the robot down: keep serving the
                # weights we already have. mark_loaded is skipped, so a later
                # (fixed) publish will be retried.
                self.logger.error(f"Failed to load adapter v{meta.version} from {meta.path(self.adapter_root)}: {e}")
                return

            self._watcher.mark_loaded(meta)
            self.adapter_version = meta

    def _apply_adapter(self, meta: AdapterVersion) -> None:
        from peft import PeftModel, load_peft_weights, set_peft_model_state_dict

        adapter_path = str(meta.path(self.adapter_root))
        start = time.perf_counter()

        if self._peft_model is None:
            # First adapter: inject the LoRA layers into the live policy. PEFT
            # rewrites the targeted submodules in place, so self.policy keeps
            # working and now routes through the adapters.
            self._peft_model = PeftModel.from_pretrained(self.policy, adapter_path, is_trainable=False)
            self._peft_model.eval()
            self.policy.eval()
            action = "injected"
        else:
            # Steady state: overwrite adapter tensors, touching nothing else.
            state_dict = load_peft_weights(adapter_path, device=str(self.device))
            result = set_peft_model_state_dict(self._peft_model, state_dict)
            unexpected = getattr(result, "unexpected_keys", None)
            if unexpected:
                self.logger.warning(f"Adapter v{meta.version} had unexpected keys: {unexpected}")
            action = "swapped"

        elapsed = (time.perf_counter() - start) * 1000
        self.logger.info(
            f"LoRA adapter v{meta.version} {action} (learner step {meta.step}, "
            f"loss {meta.loss if meta.loss is not None else float('nan'):.4f}) in {elapsed:.0f}ms"
        )


@draccus.wrap()
def serve(cfg: LoRAPolicyServerConfig):
    logging.info(pformat(asdict(cfg)))

    policy_server = LoRAPolicyServer(cfg)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"LoRAPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    register_third_party_plugins()
    serve()
