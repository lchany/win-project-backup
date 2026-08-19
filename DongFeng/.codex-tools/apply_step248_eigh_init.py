#!/usr/bin/env python3
"""STEP-248: restore CPU FP64 eigh init as a forward patch; keep hybrid QR."""
from pathlib import Path

SOAP = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/projects/mmdet3d_plugin/optimizers/soap.py")
data = SOAP.read_bytes()

old_init = (
    b"        if state['Q'] is None:\n"
    b"            state['Q'] = self.get_identity_basis(state['GG'])\n"
    b"            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)\n"
)
new_init = (
    b"        if state['Q'] is None:\n"
    b"            state['Q'] = self.get_orthogonal_matrix(state['GG'])\n"
)
if old_init not in data:
    raise SystemExit("init identity+QR block not found")
data = data.replace(old_init, new_init, 1)

old_fn = (
    b"    def get_identity_basis(self, mat):\n"
)
new_fn = (
    b"    def get_orthogonal_matrix(self, mat):\n"
    b"        \"\"\"CPU FP64 eigh init (63861df path), kept as a forward function.\"\"\"\n"
    b"        matrix = []\n"
    b"        for m in mat:\n"
    b"            if len(m) == 0:\n"
    b"                matrix.append([])\n"
    b"                continue\n"
    b"            if m.data.dtype != torch.float:\n"
    b"                float_data = False\n"
    b"                original_type = m.data.dtype\n"
    b"                original_device = m.data.device\n"
    b"                matrix.append(m.data.float())\n"
    b"            else:\n"
    b"                float_data = True\n"
    b"                matrix.append(m.data)\n"
    b"        final = []\n"
    b"        for m in matrix:\n"
    b"            if len(m) == 0:\n"
    b"                final.append([])\n"
    b"                continue\n"
    b"            if str(m.device).startswith(\"npu\"):\n"
    b"                m_cpu = m.detach().to(device=\"cpu\", dtype=torch.float64)\n"
    b"                eye_cpu = torch.eye(m.shape[0], device=\"cpu\", dtype=torch.float64)\n"
    b"                _, Q = torch.linalg.eigh(m_cpu + 1e-30 * eye_cpu)\n"
    b"                Q = Q.to(device=m.device, dtype=m.dtype)\n"
    b"            else:\n"
    b"                try:\n"
    b"                    _, Q = torch.linalg.eigh(\n"
    b"                        m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device))\n"
    b"                    Q = Q.to(m.dtype)\n"
    b"                except Exception:\n"
    b"                    _, Q = torch.linalg.eigh(m.cpu() + 1e-30 * torch.eye(m.shape[0]))\n"
    b"                    Q = Q.to(m.device)\n"
    b"            Q = torch.flip(Q, [1])\n"
    b"            if not float_data:\n"
    b"                Q = Q.to(original_device).type(original_type)\n"
    b"            final.append(Q)\n"
    b"        return final\n"
    b"\n"
    b"    def get_identity_basis(self, mat):\n"
)
if old_fn not in data:
    raise SystemExit("get_identity_basis not found")
data = data.replace(old_fn, new_fn, 1)
SOAP.write_bytes(data)
print("patched_ok eigh", data.count(b"def get_orthogonal_matrix("),
      "identity_call", data.count(b"get_identity_basis(state"),
      "hybrid", data.count(b"_hybrid_qr"))
