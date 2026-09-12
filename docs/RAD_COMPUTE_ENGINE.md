# RAD Compute Engine 1.0

vQPU SDK 0.5.0 is the first RAD Compute Engine implementation. vQPU remains the quantum
fabric, while the public compute boundary defines content-addressed workload planning,
bounded execution, verification, backend identity, resource estimates, cost declarations,
and explicit approval requirements.

The qualified backend is `cpu.quantum_simulator`. It accepts an allowlisted quantum circuit,
1-20 qubits, 1-100,000 shots, at most 10,000 gates, and a fixed seed. Planning reports statevector
memory, gate count, network use, cost, and a plan digest. Execution requires the exact plan digest;
verification checks the immutable result digest and shot accounting.

No GPU, cloud, HPC, or physical QPU is qualified by this release. There is no silent fallback:
the operation names the exact CPU simulator backend. Real QPU discovery and execution remain in
the vQPU fabric but are not exposed through the qualified RAD boundary. They require separate
provider qualification, opaque credentials, price estimation, explicit approval, and evidence.
