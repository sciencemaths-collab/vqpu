# RAD Compute Engine 1.1

vQPU SDK 0.5.0 is the first RAD Compute Engine implementation. vQPU remains the quantum
fabric, while the public compute boundary defines content-addressed workload planning,
bounded execution, verification, backend identity, resource estimates, cost declarations,
and explicit approval requirements.

The qualified backend is `cpu.quantum_simulator`. It accepts an allowlisted quantum circuit,
1-20 qubits, 1-100,000 shots, at most 10,000 gates, and a fixed seed. Planning reports statevector
memory, gate count, network use, cost, and a plan digest. Execution requires the exact plan digest;
verification checks the immutable result digest and shot accounting.

Version 0.6.0 adds `rad.compute.apple_gpu` through MLX on Apple Metal. Its bounded workload is
deterministically generated float32 matrix multiplication with dimensions from 1 to 256. Planning
declares memory, operation count, network, cost, approval, and no-fallback policy. Execution pins
MLX to its GPU device, compares the result with a NumPy reference, and binds the output bytes and
complete result to SHA-256 digests.

The formal Apple qualification runs 15 real-device cases across three matrix sizes and five seeds.
Every case must replay the exact output bytes, pass the numerical tolerance, identify MLX and GPU,
and declare `simulated: false`. Availability alone leaves the backend unroutable; only the exact
qualification report enables discovery as qualified.

Slurm HPC, cloud batch, and physical QPU families are present in the fail-closed inventory but are
always unroutable in this release. Each requires its own adapter, workload benchmark, cost model,
opaque credential references where applicable, explicit approval, and attestation. There is no
silent fallback between any backend classes.
