# STEP-214 Triton-Ascend isolated environment manifest

- Container scope: exact existing container `mapqr-leicheng` only.
- Diagnostic relative directory: `diagnostics/step214_triton_ascend_3.2.0rc4` outside the business repository.
- Installation mode: Python `venv --system-site-packages`; wheel installed with `--no-index --no-deps --no-cache-dir`.
- Package: `triton-ascend==3.2.0rc4`.
- Official source: `https://gitcode.com/Ascend/triton-ascend/tree/v3.2.0rc4`.
- Annotated tag object: `e94156eeeb8ac16e348b5aa3e23bfc3c85cec7dc`.
- Peeled source commit: `0df4da8eb40099438686864ed94540e62a04e753`.
- Wheel: `triton_ascend-3.2.0rc4-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`.
- Wheel size: `50145947` bytes.
- Wheel SHA256: `91770af4b45a27abadd607cb501e8b77d0f0f395980005b42151aff7f2484a35`.
- Dry-run report SHA256: `a9030d67aff0b3b63b4c2bdc4c352b3f490fa070b470fb419237d38bb1f77301`.
- Reused, not copied or replaced: global `torch==2.7.1`, `torch-npu==2.7.1`, CANN `8.3.RC1`.
- Preserved global Triton: default Python remains `triton==3.7.1`; default Python does not see the `triton-ascend` distribution.
- Isolated validation: venv loads Triton module version `3.2.0`, distribution `triton-ascend==3.2.0rc4`, and backend registry contains only `ascend`.
- NPU execution status: no NPU kernel, training, profiler, or mechanism test was run during installation validation.
- Rollback: after resolving and validating the diagnostic root is exactly this STEP-214 directory and is not a symlink, remove that directory only. Do not remove global site-packages or CANN paths.
