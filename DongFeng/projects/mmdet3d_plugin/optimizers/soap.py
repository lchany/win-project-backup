import json
import os
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from mmcv.runner.optimizer import OPTIMIZERS
from itertools import chain

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py
@OPTIMIZERS.register_module()
class SOAP(Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
            Helps at large precondition_frequency (~100 in our experiments), 
            but hurts performance at small precondition_frequency (~10 in our experiments).
        data_format (`str`, *optional*, defaults to `channels_first`):
            Data format of the input for convolutional layers.
            Should be "channels_last" for data_format of NHWC and "channels_first" for NCHW.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.9),
        shampoo_beta: float= -1,
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        precondition_frequency: int=10,
        max_precond_dim: int=10000, # 
        one_sided_dim_threshold: int | None = None,
        merge_dims: bool = False, # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "one_sided_dim_threshold": one_sided_dim_threshold,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        
    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        
        if curr_shape > 1 or len(new_shape)==0:
            new_shape.append(curr_shape)
        
        new_grad = grad.reshape(new_shape)
        return new_grad     
    
    def _foreach_chunk_key(self, group, state, grad):
        """Return settings that must remain identical inside a chunk."""
        return (
            group["betas"], group["eps"], group["lr"],
            group["weight_decay"], group["correct_bias"],
            group["normalize_grads"], group["max_precond_dim"],
            group["merge_dims"], group["precondition_1d"],
            group["precondition_frequency"], state["step"],
            grad.device, grad.dtype,
        )

    def _step_foreach_chunk(self, chunk):
        """Update an ordered chunk without changing each tensor's math."""
        params = [item[0] for item in chunk]
        grads = [item[1] for item in chunk]
        states = [item[2] for item in chunk]
        group = chunk[0][3]
        beta1, beta2 = group["betas"]
        for state in states:
            self._stale_q_install_if_due(state)
        grad_projected = [
            self.project(grad, state, merge_dims=group["merge_dims"],
                         max_precond_dim=group["max_precond_dim"])
            for grad, state in zip(grads, states)
        ]
        exp_avg = [state["exp_avg"] for state in states]
        exp_avg_sq = [state["exp_avg_sq"] for state in states]
        for state in states:
            state["step"] += 1
        torch._foreach_mul_(exp_avg, beta1)
        torch._foreach_add_(exp_avg, grads, alpha=(1.0 - beta1))
        torch._foreach_mul_(exp_avg_sq, beta2)
        grad_projected_sq = torch._foreach_mul(grad_projected, grad_projected)
        torch._foreach_add_(exp_avg_sq, grad_projected_sq,
                            alpha=(1.0 - beta2))
        del grad_projected_sq, grad_projected
        denom = torch._foreach_sqrt(exp_avg_sq)
        torch._foreach_add_(denom, group["eps"])
        exp_avg_projected = [
            self.project(avg, state, merge_dims=group["merge_dims"],
                         max_precond_dim=group["max_precond_dim"])
            for avg, state in zip(exp_avg, states)
        ]
        preconditioned = torch._foreach_div(exp_avg_projected, denom)
        del exp_avg_projected, denom
        norm_grads = [
            self.project_back(value, state, merge_dims=group["merge_dims"],
                              max_precond_dim=group["max_precond_dim"])
            for value, state in zip(preconditioned, states)
        ]
        del preconditioned
        if group["normalize_grads"]:
            norm_grads = [
                grad / (1e-30 + torch.mean(grad ** 2) ** 0.5)
                for grad in norm_grads
            ]
        step_size = group["lr"]
        if group["correct_bias"]:
            step = states[0]["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = step_size * (bias_correction2 ** .5) / bias_correction1
        torch._foreach_add_(params, norm_grads, alpha=-step_size)
        if group["weight_decay"] > 0.0:
            torch._foreach_add_(params, params,
                                alpha=(-group["lr"] * group["weight_decay"]))
        for grad, state in zip(grads, states):
            self.update_preconditioner(
                grad, state, max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
                precondition_1d=group["precondition_1d"])

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        loss = None
        chunk = []
        chunk_key = None
        chunk_numel = 0
        chunk_numel_limit = 8_000_000

        def flush_chunk():
            nonlocal chunk, chunk_key, chunk_numel
            if chunk:
                self._step_foreach_chunk(chunk)
                chunk = []
                chunk_key = None
                chunk_numel = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.nan_to_num_()
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if "Q" not in state:
                    flush_chunk()
                    self.init_preconditioner(
                        grad, state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(group["shampoo_beta"]
                                      if group["shampoo_beta"] >= 0
                                      else group["betas"][1]),
                        max_precond_dim=group["max_precond_dim"],
                        one_sided_dim_threshold=group["one_sided_dim_threshold"],
                        merge_dims=group["merge_dims"])
                    self.update_preconditioner(
                        grad, state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"])
                    continue
                key = self._foreach_chunk_key(group, state, grad)
                if chunk and (key != chunk_key or
                              chunk_numel + grad.numel() > chunk_numel_limit):
                    flush_chunk()
                if not chunk:
                    chunk_key = key
                chunk.append((p, grad, state, group))
                chunk_numel += grad.numel()
                if chunk_numel >= chunk_numel_limit:
                    flush_chunk()
        flush_chunk()
        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10, 
                            shampoo_beta=0.95, max_precond_dim=10000,
                            one_sided_dim_threshold=None, precondition_1d=False,
                            merge_dims=False):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state['GG'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            one_sided_axis = None
            if (
                one_sided_dim_threshold is not None
                and grad.dim() == 2
                and max(grad.shape) > one_sided_dim_threshold
            ):
                one_sided_axis = min(range(2), key=lambda index: (grad.shape[index], index))

            for index, sh in enumerate(grad.shape):
                if sh > max_precond_dim or (
                    one_sided_axis is not None and index != one_sided_axis
                ):
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device))
                    
        state['Q'] = None # Will hold all the eigenbases of the preconditioner.
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta          
        
    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == 'channels_last':
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [0]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad
        
    def update_preconditioner(self, grad, state, 
                              max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1-state['shampoo_beta'])
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if len(state['GG'][idx]) > 0:
                        outer_product = torch.tensordot(
                                new_grad,
                                new_grad,
                                dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                            )
                        state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
            else:
                for idx, sh in enumerate(grad.shape):
                    if len(state['GG'][idx]) > 0:
                        outer_product = torch.tensordot(
                                grad,
                                grad,
                                # Contracts across all dimensions except for k.
                                dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                            )
                        state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
                     
        if state['Q'] is None:
            state['Q'] = self.get_identity_basis(state['GG'])
            if self._stale_q_eligible(state, merge_dims):
                # Async path: submit QR on the side stream; Q stays as the
                # identity basis until the first install step (iter k).
                # This avoids the ~208 s synchronous AICPU stall at iter 4.
                self._stale_q_submit(state, max_precond_dim, merge_dims)
            else:
                state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
        elif state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            if self._stale_q_eligible(state, merge_dims):
                self._stale_q_submit(state, max_precond_dim, merge_dims)
            else:
                state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == 'channels_last' and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [1]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
                
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad
        
    def get_identity_basis(self, mat):
        """
        Initializes the eigenbasis on-device, following Meta's QR SOAP implementation.
        """
        basis = []
        for m in mat:
            if len(m) == 0:
                basis.append([])
            else:
                basis.append(torch.eye(m.shape[0], device=m.device, dtype=m.dtype))
        return basis
        

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration 
        followed by torch.linalg.qr decomposition.

        The complete update stays on the parameter device and computes in FP32.
        """
        precond_list = state['GG']
        orth_list = state['Q']

        matrix = []
        orth_matrix = []
        for m,o in zip(precond_list, orth_list):
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
        
        orig_shape = state['exp_avg_sq'].shape
        if self._data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']
            
        final = []
        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
            if len(m)==0:
                final.append([])
                continue
            original_dtype = precond_list[ind].dtype
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True, stable=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o.index_select(1, sort_idx)
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            if Q.dtype != original_dtype:
                Q = Q.to(original_dtype)
            final.append(Q)
        
        if merge_dims:
            if self._data_format == 'channels_last' and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)
                
        state['exp_avg_sq'] = exp_avg_sq
        return final

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
        """The AICPU-bound half: one linalg.qr per planned factor.

        Each factor gets its own NPU Event so _stale_q_install_if_due can
        query per-factor readiness without a single global synchronize() that
        forces 379 QR kernels to drain in series on the host side.
        """
        stream = torch.npu.current_stream()
        result = []
        for entry in plan:
            if entry is None:
                result.append(None)
                continue
            Q, _ = torch.linalg.qr(entry["power_iter"])
            if Q.dtype != entry["original_dtype"]:
                Q = Q.to(entry["original_dtype"])
            # One event per factor: recorded immediately after the QR kernel is
            # submitted to the stream.  The install step can query each event
            # independently and skip waiting for already-completed ones.
            ev = torch.npu.Event()
            ev.record(stream)
            result.append({"Q": Q, "event": ev})
        return result

    def _install_diag_enabled(self):
        return os.environ.get("SOAP_INSTALL_DIAG", "0") == "1"

    def _install_diag_log(self, record):
        if not self._install_diag_enabled():
            return
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
        except Exception:
            pass
        base = os.environ.get("SOAP_INSTALL_DIAG_LOG")
        if not base:
            return
        if base.endswith(".jsonl"):
            path = base[:-6] + f"_rank{rank}.jsonl"
        else:
            path = f"{base}_rank{rank}.jsonl"
        record = dict(record)
        record["rank"] = rank
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _qr_install(self, state, plan, qlist, max_precond_dim=10000, merge_dims=False):
        """Atomically apply the planned exp_avg_sq permutation and the new Q.

        qlist entries are either plain tensors (legacy synchronous path) or
        dicts {"Q": tensor, "event": Event} produced by the per-factor async
        path.  The event is queried first; only those still in-flight incur an
        actual host wait.  Most events will already be signalled because k≥1
        steps have elapsed since submit, so query() typically returns True
        immediately for all 559 factors and the host never blocks.
        """
        current = torch.npu.current_stream()
        orig_shape = state['exp_avg_sq'].shape
        if self._data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']

        final = []
        n_ready = 0
        n_wait = 0
        sync_ms = 0.0
        t0 = time.perf_counter()
        for entry, q_item in zip(plan, qlist):
            if entry is None:
                final.append([])
                continue
            # Unwrap per-factor async dict or plain tensor (sync / first-init path).
            if isinstance(q_item, dict):
                ev = q_item["event"]
                if ev.query():
                    n_ready += 1
                else:
                    n_wait += 1
                    ts = time.perf_counter()
                    ev.synchronize()
                    sync_ms += (time.perf_counter() - ts) * 1000.0
                Q = q_item["Q"]
                Q.record_stream(current)
            else:
                Q = q_item
                if torch.is_tensor(Q):
                    Q.record_stream(current)
            exp_avg_sq = exp_avg_sq.index_select(entry["ind"], entry["sort_idx"])
            final.append(Q)
        install_ms = (time.perf_counter() - t0) * 1000.0
        if self._install_diag_enabled() and (n_ready + n_wait) > 0:
            self._install_diag_log({
                "event": "qr_install",
                "step": int(state.get("step", -1)),
                "n_factors": n_ready + n_wait,
                "n_query_true": n_ready,
                "n_query_false": n_wait,
                "sync_ms": round(sync_ms, 3),
                "install_ms": round(install_ms, 3),
            })

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
            # _qr_finish now returns per-factor {"Q", "event"} dicts so that
            # _qr_install can query each factor independently at install time
            # instead of issuing one global synchronize() that serialises the
            # host wait for all 559 AICPU QR completions at once.
            qlist = self._qr_finish(plan)
        for entry in plan:
            if entry is not None:
                # record_stream lets the allocator reclaim the block only after
                # the side stream has consumed it, so the reference can go now
                # instead of being held for k steps.
                entry["power_iter"].record_stream(stream)
                entry["power_iter"] = None
        state["_stale_q_pending"] = {
            "plan": plan,
            "qlist": qlist,
            # No single global event here; per-factor events live in each
            # qlist[i]["event"] and are queried lazily in _qr_install.
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
        t0 = time.perf_counter()
        self._qr_install(
            state, pending["plan"], pending["qlist"],
            pending["max_precond_dim"], pending["merge_dims"],
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if self._install_diag_enabled():
            self._install_diag_log({
                "event": "stale_q_install_if_due",
                "step": int(state.get("step", -1)),
                "target_step": int(pending["target_step"]),
                "wall_ms": round(wall_ms, 3),
            })
        state.pop("_stale_q_pending", None)

    def state_dict(self):
        """Flush any in-flight basis so the persisted schema stays the 7-key contract."""
        for state in self.state.values():
            self._stale_q_install_if_due(state, force=True)
        return super().state_dict()
