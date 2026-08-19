#!/usr/bin/env python3
"""STEP-221 Stage B candidate: stale-Q asynchronous SOAP cycle QR.

Reference artifact for the change to the authoritative `soap.py`
(19,169 bytes, SHA 0e49429dbca9d9a2546c29f54e79639265f7468703ba4b36fa3b3796861a1077).
This file is the reviewable definition of the patch; it is not imported by training.

Design decisions that differ from the v2 draft, forced by reading the source:

1. `get_orthogonal_matrix_QR` is left BYTE-UNTOUCHED. The new `_qr_plan` /
   `_qr_finish` / `_qr_install` trio is used only by the stale path, so with
   `SOAP_STALE_Q_K=0` the executed baseline path is unchanged rather than
   merely "believed equivalent". The gate proves trio == original positively.
2. Only `torch.linalg.qr(power_iter)` goes to the side stream. Everything that
   reads `GG` runs on the default stream first, because `GG.lerp_()` mutates it
   in place every step and a cross-step side-stream read would race.
3. The `exp_avg_sq` permutation is decided at submit time but replayed at
   install time, together with the new Q, so the two are never mismatched.
4. `soap.py` has mixed line endings (259 CRLF lines, rest LF). Anchors below are
   byte-exact; the patcher must read/write in binary.
"""

# Appended verbatim at the end of the SOAP class body. CRLF, matching the
# surrounding region of the file.
STALE_Q_METHODS = '''
    # --- STEP-221 stale-Q asynchronous cycle QR -------------------------------
    _STALE_Q_SHAPES = frozenset({
        1, 3, 4, 7, 8, 11, 22, 32, 40, 64, 96, 120, 128, 160, 192,
        220, 256, 352, 440, 512, 768, 1024, 2560,
    })

    def _stale_q_k(self):
        cached = getattr(self, "_stale_q_k_cached", None)
        if cached is None:
            try:
                cached = int(os.environ.get("SOAP_STALE_Q_K", "0"))
            except ValueError:
                cached = 0
            if cached < 0:
                cached = 0
            self._stale_q_k_cached = cached
        return cached

    def _stale_q_side_stream(self):
        stream = getattr(self, "_stale_q_stream_cached", None)
        if stream is None:
            stream = torch.npu.Stream()
            self._stale_q_stream_cached = stream
        return stream

    def _stale_q_eligible(self, state, merge_dims):
        k = self._stale_q_k()
        if k <= 0 or merge_dims:
            return False
        if k >= int(state["precondition_frequency"]):
            return False
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            return False
        if self._data_format == "channels_last" and state["exp_avg_sq"].dim() == 4:
            return False
        for m, o in zip(state["GG"], state["Q"]):
            if len(m) == 0:
                continue
            if m.dtype != torch.float32 or o.dtype != torch.float32:
                return False
            if o.shape[0] != o.shape[1] or int(o.shape[0]) not in self._STALE_Q_SHAPES:
                return False
        return True

    def _qr_plan(self, state, max_precond_dim=10000, merge_dims=False):
        """Default-stream half of the cycle basis update; contains no QR."""
        precond_list = state['GG']
        orth_list = state['Q']

        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.dtype != torch.float32:
                matrix.append(m.float())
                orth_matrix.append(o.float())
            else:
                matrix.append(m)
                orth_matrix.append(o)

        plan = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                plan.append(None)
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True, stable=True)
            o = o.index_select(1, sort_idx)
            plan.append({
                "ind": ind,
                "sort_idx": sort_idx,
                "power_iter": m @ o,
                "original_dtype": precond_list[ind].dtype,
            })
        return plan

    def _qr_finish(self, plan):
        """The AICPU-bound half: one linalg.qr per planned factor."""
        result = []
        for entry in plan:
            if entry is None:
                result.append(None)
                continue
            Q, _ = torch.linalg.qr(entry["power_iter"])
            if Q.dtype != entry["original_dtype"]:
                Q = Q.to(entry["original_dtype"])
            result.append(Q)
        return result

    def _qr_install(self, state, plan, qlist, max_precond_dim=10000, merge_dims=False):
        """Atomically apply the planned exp_avg_sq permutation and the new Q."""
        orig_shape = state['exp_avg_sq'].shape
        if self._data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']

        final = []
        for entry, Q in zip(plan, qlist):
            if entry is None:
                final.append([])
                continue
            exp_avg_sq = exp_avg_sq.index_select(entry["ind"], entry["sort_idx"])
            final.append(Q)

        if merge_dims:
            if self._data_format == 'channels_last' and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state['exp_avg_sq'] = exp_avg_sq
        state['Q'] = final
        return final

    def _stale_q_submit(self, state, max_precond_dim, merge_dims):
        if state.get("_stale_q_pending") is not None:
            raise RuntimeError("SOAP stale-Q pending overlap (k must stay < frequency)")
        plan = self._qr_plan(state, max_precond_dim, merge_dims)
        stream = self._stale_q_side_stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            qlist = self._qr_finish(plan)
        for entry in plan:
            if entry is not None:
                # record_stream lets the allocator reclaim the block only after
                # the side stream has consumed it, so the reference can go now
                # instead of being held for k steps.
                entry["power_iter"].record_stream(stream)
                entry["power_iter"] = None
        event = torch.npu.Event()
        event.record(stream)
        state["_stale_q_pending"] = {
            "plan": plan,
            "qlist": qlist,
            "event": event,
            "target_step": int(state["step"]) + self._stale_q_k(),
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
        }

    def _stale_q_install_if_due(self, state, force=False):
        pending = state.get("_stale_q_pending")
        if pending is None:
            return
        if not force and int(state["step"]) < int(pending["target_step"]):
            return
        pending["event"].synchronize()
        current = torch.npu.current_stream()
        for value in pending["qlist"]:
            if torch.is_tensor(value):
                # Allocated on the side stream, consumed here on the default one.
                value.record_stream(current)
        self._qr_install(
            state, pending["plan"], pending["qlist"],
            pending["max_precond_dim"], pending["merge_dims"],
        )
        state.pop("_stale_q_pending", None)

    def state_dict(self):
        """Flush any in-flight basis so the persisted schema stays the 7-key contract."""
        for state in self.state.values():
            self._stale_q_install_if_due(state, force=True)
        return super().state_dict()
'''


# Byte-exact edits. `soap.py` mixes CRLF and LF; each anchor below carries the
# newline actually present at that location, verified against the live file.
REPLACEMENTS = [
    {
        "id": "import_os",
        "why": "SOAP_STALE_Q_K is read from the environment.",
        "region_newline": "CRLF",
        "old": b"import torch\r\n",
        "new": b"import os\r\nimport torch\r\n",
        "count": 1,
    },
    {
        "id": "cycle_branch",
        "why": "Route the periodic basis refresh through the side stream when k>0.",
        "region_newline": "CRLF",
        "old": (
            b"        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:\r\n"
            b"            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)"
        ),
        "new": (
            b"        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:\r\n"
            b"            if self._stale_q_eligible(state, merge_dims):\r\n"
            b"                self._stale_q_submit(state, max_precond_dim, merge_dims)\r\n"
            b"            else:\r\n"
            b"                state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)"
        ),
        "count": 1,
    },
    {
        "id": "install_before_project",
        "why": (
            "A due basis must be installed before this step's project() so Q and "
            "exp_avg_sq are always consumed as a matched pair."
        ),
        "region_newline": "LF",
        "old": b'        beta1, beta2 = group["betas"]\n        grad_projected = [',
        "new": (
            b'        beta1, beta2 = group["betas"]\n'
            b"        for state in states:\n"
            b"            self._stale_q_install_if_due(state)\n"
            b"        grad_projected = ["
        ),
        "count": 1,
    },
]

# The trio is appended at the end of the class body; `get_orthogonal_matrix_QR`
# itself is not modified, so the k=0 path is unchanged by construction.
APPEND_AT_END_OF_CLASS = True

INVARIANTS = [
    "get_orthogonal_matrix_QR is byte-untouched; k=0 executes the HEAD path.",
    "Gate must positively prove _qr_plan+_qr_finish+_qr_install == get_orthogonal_matrix_QR "
    "bit-for-bit on the real 543-factor checkpoint state.",
    "0 < k < precondition_frequency, so at most one pending basis exists per state.",
    "Side stream runs only linalg.qr; every GG/Q read happens on the default stream.",
    "exp_avg_sq permutation is replayed at install time, never at submit time.",
    "state_dict() force-installs and pops the pending key, keeping exactly the 7 SOAP keys.",
    "power_iter tensors get record_stream(side) before their references are dropped.",
    "Non-FP32, merge_dims, channels_last-4D and off-whitelist shapes fall back to sync QR.",
]
