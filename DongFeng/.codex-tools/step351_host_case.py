#!/usr/bin/env python3
"""Use the existing host supervisor with the STEP-351 controller adapter."""

import step343_host_case as _host
import step351_controller as _controller

_host.atomic_json = _controller.atomic_json
_host.postflight = _controller.postflight
_host.preflight = _controller.preflight
_host.process_starttime = _controller.process_starttime
_host.supervise = _controller.supervise

if __name__ == "__main__":
    raise SystemExit(_host.main())

