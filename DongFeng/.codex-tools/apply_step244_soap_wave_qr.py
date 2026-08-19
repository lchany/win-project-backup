#!/usr/bin/env python3
"""Apply STEP-244 wave-parallel batched cycle QR to soap.py (regex, mixed newlines)."""
import re
from pathlib import Path

SOAP = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/projects/mmdet3d_plugin/optimizers/soap.py")
data = SOAP.read_bytes()

def sub1(pattern, repl, name, count=1):
    global data
    bpat = pattern.encode("utf-8") if isinstance(pattern, str) else pattern
    brepl = repl.encode("utf-8") if isinstance(repl, str) else repl
    bpat = bpat.replace(b"\n", rb"\r?\n")
    new_data, n = re.subn(bpat, brepl, data, count=count, flags=re.DOTALL)
    assert n == count, f"{name}: matched {n}, expected {count}"
    data = new_data

# step(): flush batches at end
sub1(
    r"        flush_chunk\(\)\r?\n        return loss",
    "        flush_chunk()\n        self._flush_cycle_qr_batches()\n        return loss",
    "step flush",
)

# update_preconditioner: queue cycle QR
sub1(
    r"            if self\._stale_q_eligible\(state, merge_dims\):\r?\n"
    r"                self\._stale_q_submit\(state, max_precond_dim, merge_dims\)\r?\n"
    r"            else:\r?\n"
    r"                state\['Q'\] = self\.get_orthogonal_matrix_QR\(state, max_precond_dim, merge_dims\)\s*\r?\n",
    "            if self._stale_q_eligible(state, merge_dims):\n"
    "                self._stale_q_queue(state, max_precond_dim, merge_dims)\n"
    "            else:\n"
    "                self._sync_qr_queue(state, max_precond_dim, merge_dims)\n",
    "update_preconditioner",
)

# get_orthogonal_matrix_QR: batch within state
sub1(
    r"        final = \[\]\r?\n"
    r"        for ind, \(m,o\) in enumerate\(zip\(matrix, orth_matrix\)\):\r?\n"
    r"            if len\(m\)==0:\r?\n"
    r"                final\.append\(\[\]\)\r?\n"
    r"                continue\r?\n"
    r"            original_dtype = precond_list\[ind\]\.dtype\r?\n"
    r"            est_eig = torch\.diag\(o\.T @ m @ o\)\r?\n"
    r"            sort_idx = torch\.argsort\(est_eig, descending=True, stable=True\)\r?\n"
    r"            exp_avg_sq = exp_avg_sq\.index_select\(ind, sort_idx\)\r?\n"
    r"            o = o\.index_select\(1, sort_idx\)\r?\n"
    r"            power_iter = m @ o\r?\n"
    r"            Q = self\._sharded_qr\(power_iter\)\r?\n"
    r"            if Q\.dtype != original_dtype:\r?\n"
    r"                Q = Q\.to\(original_dtype\)\r?\n"
    r"            final\.append\(Q\)\r?\n",
    "        pending_q = []\n"
    "        final = []\n"
    "        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):\n"
    "            if len(m)==0:\n"
    "                final.append([])\n"
    "                continue\n"
    "            original_dtype = precond_list[ind].dtype\n"
    "            est_eig = torch.diag(o.T @ m @ o)\n"
    "            sort_idx = torch.argsort(est_eig, descending=True, stable=True)\n"
    "            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)\n"
    "            o = o.index_select(1, sort_idx)\n"
    "            power_iter = m @ o\n"
    "            pending_q.append((ind, sort_idx, power_iter, original_dtype))\n"
    "        q_values = self._parallel_qr_waves([t[2] for t in pending_q])\n"
    "        for (ind, sort_idx, power_iter, original_dtype), Q in zip(pending_q, q_values):\n"
    "            if Q.dtype != original_dtype:\n"
    "                Q = Q.to(original_dtype)\n"
    "            final.append(Q)\n",
    "get_orthogonal_matrix_QR loop",
)

# _qr_finish
sub1(
    r"    def _qr_finish\(self, plan\):\r?\n"
    r"        \"\"\"The AICPU-bound half: one linalg\.qr per planned factor\.\"\"\"\r?\n"
    r"        result = \[\]\r?\n"
    r"        for entry in plan:\r?\n"
    r"            if entry is None:\r?\n"
    r"                result\.append\(None\)\r?\n"
    r"                continue\r?\n"
    r"            Q = self\._sharded_qr\(entry\[\"power_iter\"\]\)\r?\n"
    r"            if Q\.dtype != entry\[\"original_dtype\"\]:\r?\n"
    r"                Q = Q\.to\(entry\[\"original_dtype\"\]\)\r?\n"
    r"            result\.append\(Q\)\r?\n"
    r"        return result",
    "    def _qr_finish(self, plan):\n"
    "        \"\"\"The AICPU-bound half: batched wave-parallel linalg.qr.\"\"\"\n"
    "        active = [(i, e) for i, e in enumerate(plan) if e is not None]\n"
    "        result = [None] * len(plan)\n"
    "        if not active:\n"
    "            return result\n"
    "        q_values = self._parallel_qr_waves([e[\"power_iter\"] for _, e in active])\n"
    "        for (i, entry), Q in zip(active, q_values):\n"
    "            if Q.dtype != entry[\"original_dtype\"]:\n"
    "                Q = Q.to(entry[\"original_dtype\"])\n"
    "            result[i] = Q\n"
    "        return result",
    "_qr_finish",
)

# install optional event
sub1(
    r"        pending\[\"event\"\]\.synchronize\(\)",
    "        if pending.get(\"event\") is not None:\n            pending[\"event\"].synchronize()",
    "install sync",
)

# replace STEP-243 block
pat243 = re.compile(
    rb"    # --- STEP-243 rank-sharded cycle QR .*?"
    rb"        dist\.broadcast\(Q, src=owner\)\r?\n"
    rb"        return Q\r?\n",
    re.DOTALL,
)
helper = (
    "    # --- STEP-244 wave-parallel batched cycle QR -----------------------------\n"
    "    def _dist_qr_enabled(self):\n"
    "        cached = getattr(self, \"_dist_qr_enabled_cached\", None)\n"
    "        if cached is None:\n"
    "            cached = os.environ.get(\"SOAP_DIST_QR\", \"1\") == \"1\"\n"
    "            if cached:\n"
    "                import torch.distributed as dist\n"
    "                cached = (dist.is_available() and dist.is_initialized()\n"
    "                          and dist.get_world_size() > 1)\n"
    "            self._dist_qr_enabled_cached = cached\n"
    "        return cached\n"
    "\n"
    "    def _parallel_qr_waves(self, power_iters):\n"
    "        \"\"\"Run linalg.qr on up to world_size inputs concurrently per wave.\"\"\"\n"
    "        if not power_iters:\n"
    "            return []\n"
    "        if not self._dist_qr_enabled():\n"
    "            return [torch.linalg.qr(pi)[0] for pi in power_iters]\n"
    "        import torch.distributed as dist\n"
    "        rank = dist.get_rank()\n"
    "        world = dist.get_world_size()\n"
    "        results = [None] * len(power_iters)\n"
    "        for wave_start in range(0, len(power_iters), world):\n"
    "            wave_len = min(world, len(power_iters) - wave_start)\n"
    "            local_Q = None\n"
    "            if rank < wave_len:\n"
    "                local_Q, _ = torch.linalg.qr(power_iters[wave_start + rank])\n"
    "                local_Q = local_Q.contiguous()\n"
    "            for r in range(wave_len):\n"
    "                pi = power_iters[wave_start + r]\n"
    "                if rank == r:\n"
    "                    Q = local_Q\n"
    "                else:\n"
    "                    Q = torch.empty_like(pi)\n"
    "                dist.broadcast(Q, src=r)\n"
    "                results[wave_start + r] = Q\n"
    "        return results\n"
    "\n"
    "    def _cycle_qr_batches(self):\n"
    "        if not hasattr(self, \"_cycle_qr_stale_batch\"):\n"
    "            self._cycle_qr_stale_batch = []\n"
    "            self._cycle_qr_sync_batch = []\n"
    "        return self._cycle_qr_stale_batch, self._cycle_qr_sync_batch\n"
    "\n"
    "    def _stale_q_queue(self, state, max_precond_dim, merge_dims):\n"
    "        if state.get(\"_stale_q_pending\") is not None:\n"
    "            raise RuntimeError(\"SOAP stale-Q pending overlap (k must stay < frequency)\")\n"
    "        stale_batch, _ = self._cycle_qr_batches()\n"
    "        stale_batch.append({\n"
    "            \"state\": state,\n"
    "            \"plan\": self._qr_plan(state, max_precond_dim, merge_dims),\n"
    "            \"max_precond_dim\": max_precond_dim,\n"
    "            \"merge_dims\": merge_dims,\n"
    "            \"target_step\": int(state[\"step\"]) + self._stale_q_k(),\n"
    "        })\n"
    "\n"
    "    def _sync_qr_queue(self, state, max_precond_dim, merge_dims):\n"
    "        _, sync_batch = self._cycle_qr_batches()\n"
    "        sync_batch.append({\n"
    "            \"state\": state,\n"
    "            \"plan\": self._qr_plan(state, max_precond_dim, merge_dims),\n"
    "            \"max_precond_dim\": max_precond_dim,\n"
    "            \"merge_dims\": merge_dims,\n"
    "        })\n"
    "\n"
    "    def _flush_cycle_qr_batches(self):\n"
    "        stale_batch, sync_batch = self._cycle_qr_batches()\n"
    "        if stale_batch:\n"
    "            self._flush_stale_q_batch(stale_batch)\n"
    "            stale_batch.clear()\n"
    "        if sync_batch:\n"
    "            self._flush_sync_qr_batch(sync_batch)\n"
    "            sync_batch.clear()\n"
    "\n"
    "    def _collect_plan_qrs(self, jobs):\n"
    "        mapping = []\n"
    "        tensors = []\n"
    "        for job_i, job in enumerate(jobs):\n"
    "            for entry_i, entry in enumerate(job[\"plan\"]):\n"
    "                if entry is None:\n"
    "                    continue\n"
    "                mapping.append((job_i, entry_i))\n"
    "                tensors.append(entry[\"power_iter\"])\n"
    "        return mapping, tensors\n"
    "\n"
    "    def _flush_stale_q_batch(self, jobs):\n"
    "        mapping, tensors = self._collect_plan_qrs(jobs)\n"
    "        q_values = self._parallel_qr_waves(tensors)\n"
    "        qlists = [[None] * len(job[\"plan\"]) for job in jobs]\n"
    "        for (job_i, entry_i), Q in zip(mapping, q_values):\n"
    "            entry = jobs[job_i][\"plan\"][entry_i]\n"
    "            if Q.dtype != entry[\"original_dtype\"]:\n"
    "                Q = Q.to(entry[\"original_dtype\"])\n"
    "            qlists[job_i][entry_i] = Q\n"
    "        for job, qlist in zip(jobs, qlists):\n"
    "            job[\"state\"][\"_stale_q_pending\"] = {\n"
    "                \"plan\": job[\"plan\"],\n"
    "                \"qlist\": qlist,\n"
    "                \"event\": None,\n"
    "                \"target_step\": job[\"target_step\"],\n"
    "                \"max_precond_dim\": job[\"max_precond_dim\"],\n"
    "                \"merge_dims\": job[\"merge_dims\"],\n"
    "            }\n"
    "\n"
    "    def _flush_sync_qr_batch(self, jobs):\n"
    "        mapping, tensors = self._collect_plan_qrs(jobs)\n"
    "        q_values = self._parallel_qr_waves(tensors)\n"
    "        qlists = [[None] * len(job[\"plan\"]) for job in jobs]\n"
    "        for (job_i, entry_i), Q in zip(mapping, q_values):\n"
    "            entry = jobs[job_i][\"plan\"][entry_i]\n"
    "            if Q.dtype != entry[\"original_dtype\"]:\n"
    "                Q = Q.to(entry[\"original_dtype\"])\n"
    "            qlists[job_i][entry_i] = Q\n"
    "        for job, qlist in zip(jobs, qlists):\n"
    "            self._qr_install(\n"
    "                job[\"state\"], job[\"plan\"], qlist,\n"
    "                job[\"max_precond_dim\"], job[\"merge_dims\"],\n"
    "            )\n"
)
m = pat243.search(data)
assert m, "STEP-243 block missing"
data = data[: m.start()] + helper.encode("utf-8") + data[m.end() :]

SOAP.write_bytes(data)
print("patched_ok", "parallel", data.count(b"_parallel_qr_waves"), "sharded", data.count(b"_sharded_qr"))
