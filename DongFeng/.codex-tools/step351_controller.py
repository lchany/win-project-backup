#!/usr/bin/env python3
"""STEP-351 adapter for the proven back-8 controller."""

import step343_world8_controller as _base


def _shadow_gate(row):
    return row.get("shadow_gate") is True


_base.validate_ready_opp_transition = _shadow_gate
atomic_json = _base.atomic_json
postflight = _base.postflight
preflight = _base.preflight
process_starttime = _base.process_starttime
supervise = _base.supervise

