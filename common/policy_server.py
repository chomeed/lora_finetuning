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

Shortcut (installed by `pip install -e .`): ``ws-serve-policy``.

    ws-serve-policy \
        --host=0.0.0.0 --port=8080 --fps=30 \
        --adapter_addr=trainer-host:8090 --reload_on=chunk

``--version_status_file`` (default ``/tmp/lora_policy_version.json``, shared with
the converter's ``--policy_status_file``) publishes the currently-serving adapter
version so the converter can tag each dagger episode's manifest row with the
policy version it was collected under. Written at startup as version 0 (= base
policy) and rewritten on every swap.

The console shows lifecycle events only (startup, adapter swaps, errors); pass
``--verbose=true`` to restore lerobot's per-observation inference logging.

Equivalent module form: ``python -m lora_finetuning.common.policy_server``.
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
from .manifest import write_policy_status
from .transport import AdapterApplier, AdapterClient, AdapterVersion


class LoRAPolicyServer(PolicyServer):
    prefix = "lora_policy_server"
    # Inherited PolicyServer code logs its per-observation chatter through this;
    # quiet mode (the default) demotes it to WARNING in serve().
    logger = get_logger(prefix)
    # Our own lifecycle events (startup, adapter swaps) use a separate logger so
    # they stay visible when the stock chatter above is demoted.
    lora_logger = logging.getLogger("lora_adapter")

    def __init__(self, config: LoRAPolicyServerConfig):
        super().__init__(config)
        self.config: LoRAPolicyServerConfig = config

        self._applier = AdapterApplier(device=str(self.device))
        self._adapter_lock = threading.Lock()
        self.adapter_version: AdapterVersion | None = None

        if config.adapter_addr:
            self._client = AdapterClient(config.adapter_addr, root=config.adapter_cache_dir)
            self.lora_logger.info(
                f"Pulling LoRA adapters from {config.adapter_addr} (reload_on={config.reload_on})"
            )
        else:
            self._client = None
            self.lora_logger.info("No adapter_addr set: running as a stock PolicyServer (no hot-reload)")

        # Start the status file at version 0 (= base policy, no adapter applied
        # yet) so a stale file left by a previous run can never mislabel the
        # episodes collected before this run's first swap.
        if config.version_status_file:
            try:
                write_policy_status(config.version_status_file, version=0)
            except OSError as e:
                self.lora_logger.warning(f"could not write version status file: {e}")

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
                self.lora_logger.error(f"Failed to load adapter v{meta.version} from {meta.local_dir}: {e}")
                return

            self._client.mark_loaded(meta)
            self.adapter_version = meta

            # Publish the live version so a co-located converter can tag which
            # policy version each dagger episode was collected under.
            if self.config.version_status_file:
                try:
                    write_policy_status(
                        self.config.version_status_file,
                        version=meta.version,
                        trainer_step=meta.step,
                        loss=meta.loss,
                    )
                except OSError as e:
                    self.lora_logger.warning(f"could not write version status file: {e}")

    def _apply_adapter(self, meta: AdapterVersion) -> None:
        start = time.perf_counter()
        action = self._applier.apply(self.policy, meta.local_dir, version=meta.version)
        elapsed = (time.perf_counter() - start) * 1000
        self.lora_logger.info(
            f"LoRA adapter v{meta.version} {action} (trainer step {meta.step}, "
            f"loss {meta.loss if meta.loss is not None else float('nan'):.4f}) in {elapsed:.0f}ms"
        )


class _DropTransportChatter(logging.Filter):
    """Drop lerobot transport/utils chatter ("Starting receiver", byte counts),
    which the library logs on the ROOT logger for every gRPC stream."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.pathname.endswith("transport/utils.py")


@draccus.wrap()
def serve(cfg: LoRAPolicyServerConfig):
    # Silence PI05's per-image resize_with_pad_torch warning (fires every forward).
    logging.getLogger("lerobot.policies.pi05.modeling_pi05").setLevel(logging.ERROR)
    if not cfg.verbose:
        # Keep the console to lifecycle events (startup, adapter swaps, errors):
        # demote the stock PolicyServer's per-observation INFO chatter -- WARNING
        # and up still surface -- and drop the transport layer's per-stream noise.
        logging.getLogger(LoRAPolicyServer.prefix).setLevel(logging.WARNING)
        logging.getLogger().addFilter(_DropTransportChatter())
    logging.info(pformat(asdict(cfg)))

    policy_server = LoRAPolicyServer(cfg)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.lora_logger.info(f"LoRAPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()

    policy_server.lora_logger.info("Server terminated")


if __name__ == "__main__":
    register_third_party_plugins()
    serve()
