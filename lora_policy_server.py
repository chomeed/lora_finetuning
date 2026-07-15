"""Variant of lerobot's policy_server.py that hot-reloads LoRA adapters.

Identical to the stock ``PolicyServer`` except that it pulls LoRA adapters from a
remote trainer's gRPC ``AdapterService`` (see ``transport/client.py``) and swaps each
one into the running policy without a restart. The base policy is loaded once, at
handshake, exactly as before; only the adapter weights (a few MB) are ever fetched
and reloaded.

Two things make the swap safe:

* The pull-and-swap happens at an action-chunk boundary, on the inference thread,
  before the observation is preprocessed -- never in the middle of a forward pass.
* The client reassembles the whole adapter into a local dir before returning it, so
  a partial adapter is never applied.

Set ``reload_on=handshake`` to restrict swaps to client (re)connections, i.e.
between episodes, if you don't want the policy changing mid-rollout.

Usage:
    python -m lora_finetuning.lora_policy_server \
        --host=0.0.0.0 --port=8080 --fps=30 \
        --adapter_addr=trainer-host:8090 \
        --reload_on=chunk
"""

import logging
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat

import draccus
import grpc

from lerobot.async_inference.helpers import TimedAction, TimedObservation, get_logger
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc  # type: ignore
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import LoRAPolicyServerConfig
from .transport import AdapterApplier, AdapterClient, AdapterVersion


class LoRAPolicyServer(PolicyServer):
    prefix = "lora_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: LoRAPolicyServerConfig):
        super().__init__(config)
        self.config: LoRAPolicyServerConfig = config

        self._applier = AdapterApplier(device=str(self.device))
        self._adapter_lock = threading.Lock()
        self.adapter_version: AdapterVersion | None = None

        if config.adapter_addr:
            self._client = AdapterClient(config.adapter_addr, root=config.adapter_cache_dir)
            self.logger.info(
                f"Pulling LoRA adapters from {config.adapter_addr} (reload_on={config.reload_on})"
            )
        else:
            self._client = None
            self.logger.info("No adapter_addr set: running as a stock PolicyServer (no hot-reload)")

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
        """Pull the latest adapter and apply it if the trainer has a newer one. Never raises.

        This is where the client chooses *when* to ask: it runs at every action-chunk
        boundary (reload_on=chunk) or only on (re)connect (reload_on=handshake).
        """
        if self._client is None or self.policy is None:
            return

        meta = self._client.fetch()
        if meta is None:
            return

        with self._adapter_lock:
            try:
                self._apply_adapter(meta)
            except Exception as e:
                # A bad adapter must not take the robot down: keep serving the
                # weights we already have. mark_loaded is skipped, so a later
                # (fixed) publish will be retried.
                self.logger.error(f"Failed to load adapter v{meta.version} from {meta.local_dir}: {e}")
                return

            self._client.mark_loaded(meta)
            self.adapter_version = meta

    def _apply_adapter(self, meta: AdapterVersion) -> None:
        start = time.perf_counter()
        action = self._applier.apply(self.policy, meta.local_dir, version=meta.version)
        elapsed = (time.perf_counter() - start) * 1000
        self.logger.info(
            f"LoRA adapter v{meta.version} {action} (trainer step {meta.step}, "
            f"loss {meta.loss if meta.loss is not None else float('nan'):.4f}) in {elapsed:.0f}ms"
        )


@draccus.wrap()
def serve(cfg: LoRAPolicyServerConfig):
    # Silence PI05's per-image resize_with_pad_torch warning (fires every forward).
    logging.getLogger("lerobot.policies.pi05.modeling_pi05").setLevel(logging.ERROR)
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
