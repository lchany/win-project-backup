#!/usr/bin/env python3
"""Fail-closed template for the STEP-215-E real SOAP adapter.

Copy this file outside the business repository and implement it only after the
authoritative repo/config/checkpoint are available in the target container.
The gate rejects this template because every readiness flag is False.
"""

READINESS = {
    "real_soap_optimizer": False,
    "loads_requested_checkpoint": False,
    "uses_requested_config": False,
    "deterministic_replay_gradient": False,
    "state_view_includes_parameters": False,
    "state_view_includes_optimizer_state": False,
    "state_view_includes_q": False,
    "state_view_includes_gg_exp_avg_exp_avg_sq_step": False,
    "checkpoint_roundtrip": False,
    "sort_observable_through_torch_argsort": False,
}


class MissingRealSOAPAdapter:
    """Required interface; every method deliberately fails closed."""

    def _missing(self, name):
        raise NotImplementedError(f"authoritative real SOAP adapter field is not implemented: {name}")

    def build_trial(self, context):
        return self._missing("build_trial")

    def make_gradient(self, context, logical_step):
        return self._missing("make_gradient")

    def apply_gradient(self, trial, gradient, logical_step):
        return self._missing("apply_gradient")

    def state_view(self, trial):
        # Must return exactly {"parameters": ..., "optimizer_state": ...}.
        return self._missing("state_view")

    def save_trial(self, trial, checkpoint_path):
        return self._missing("save_trial")

    def load_trial(self, context, checkpoint_path):
        return self._missing("load_trial")

    def destroy_trial(self, trial):
        return self._missing("destroy_trial")


def create_adapter(context):
    return MissingRealSOAPAdapter()
