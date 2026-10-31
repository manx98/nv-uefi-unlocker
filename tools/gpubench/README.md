# GPU Benchmark (Windows x64)

`gpu-benchmark-x86_64-windows.exe` is a standalone CUDA Driver API benchmark for checking NVIDIA CUDA-core and Tensor Core execution after the UEFI unlock. It loads `nvcuda.dll` from the installed NVIDIA driver and JIT-compiles embedded PTX; it does not require CUDA Toolkit, `nvcc`, PyTorch, or a separate CUDA runtime DLL on the target machine.

## Tests

- **FP16 GEMM 512×512** — a WMMA matrix multiplication with direct global-memory tile loads. The output is checked against a CPU FP64 reference. This is a correctness/reference GEMM, not a tuned cuBLAS replacement; its absolute TFLOPS should not be compared directly with cuBLAS results.
- **FP32 FMA** — CUDA-core FMA issue-rate microbenchmark, with deterministic output checked against a CPU `f32::mul_add` reference.
- **Tensor Core issue-rate** — warp-level `mma.sync` instruction throughput. FP16 is available on Turing and newer; BF16 and TF32 on Ampere and newer; FP8 E4M3 on Ada (SM 8.9) and newer with CUDA Driver API 12.4 or later. Unsupported formats are omitted automatically.

The two issue-rate tests are microbenchmarks, not application GEMM throughput. Use the fixed-size GEMM for a second, memory-and-compute-influenced data point. For an application-level comparison, use the same external workload (for example, the project's existing PyTorch `bench_gemm.py` or `cli-stressor-cuda-rs`) before and after unlock.

## Usage

```powershell
# List CUDA-visible NVIDIA GPUs
.\gpu-benchmark-x86_64-windows.exe --list

# Run the short benchmark on GPU 0 (default); writes a JSON report
.\gpu-benchmark-x86_64-windows.exe --device 0

# Optional five-minute FP16 stability run, stop at 85 °C
.\gpu-benchmark-x86_64-windows.exe --device 0 --stress-seconds 300 --stress-precision fp16 --max-temp-c 85
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--list` | off | List CUDA-visible GPUs and exit |
| `--device INDEX` | `0` | CUDA device index |
| `--output PATH` | auto-generated | JSON report path |
| `--stress-seconds N` | `0` | Opt-in sustained load, 1–3600 seconds |
| `--stress-precision MODE` | `fp16` | `fp16`, `bf16`, `tf32`, `fp8`, or `fp32` |
| `--max-temp-c C` | `85` | Stop stress at this GPU temperature (65–95 °C) |

The JSON report records the GPU name/index, compute capability, SM count, VRAM, driver versions, sample timings, TFLOPS, correctness verdicts and (when requested) temperature, power, clock, utilization and P-state telemetry.

## Safety and interpretation

- The normal benchmark is short. Sustained stress is **off by default** and requires an explicit `--stress-seconds` value.
- Stress mode requires `nvidia-smi` temperature telemetry and stops when the selected temperature limit is reached. If temperature cannot be read, the stress run is refused.
- The program does not change clocks, voltage, power limits, fan settings, firmware, or GPU registers. It only submits compute workloads and reads telemetry.
- Compare before/after results on the same GPU, driver, power profile, cooling setup and workload. Close other GPU workloads first. Report the GEMM and issue-rate results separately; do not treat a microbenchmark multiplier as a universal application speedup.

## Build

From the repository root on an x86_64 Windows MSVC Rust host:

```powershell
cargo test --locked --manifest-path tools/gpubench/Cargo.toml --target-dir target/gpubench
cargo build --release --locked --manifest-path tools/gpubench/Cargo.toml --target-dir target/gpubench
```

The executable is `target/gpubench/release/gpu-benchmark.exe`. The root `build.ps1` builds it and copies it to the USB root as `gpu-benchmark-x86_64-windows.exe`.
