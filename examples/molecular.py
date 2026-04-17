"""
Axiomatic Quantum Intelligence (AQI) — Molecular Simulation Pipeline
=====================================================================
Stage 1: Axiom Space       — Define Hilbert space from molecular orbitals
Stage 2: Hamiltonian Build  — Construct qubit Hamiltonian from electron integrals
Stage 3: Quantum Evolution  — VQE with parameterized unitary circuits
Stage 4: Measurement        — Collapse to ground state energy + comparison
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qchem
import json
import time

# ─────────────────────────────────────────────────────────
# TEST MOLECULE: H2 (Hydrogen) — simplest molecule
# Known exact ground state energy: -1.1373 Ha at 0.735 Å
# We'll also scan bond distances to build dissociation curve
# ─────────────────────────────────────────────────────────

def write_pdb(symbols, coords_angstrom, filename):
    """Write a minimal PDB file from atom symbols and coordinates."""
    with open(filename, "w") as f:
        f.write("HEADER    AQI TEST MOLECULE\n")
        f.write("REMARK    Generated for Axiomatic Quantum Intelligence pipeline\n")
        for i, (sym, coord) in enumerate(zip(symbols, coords_angstrom)):
            f.write(
                f"ATOM  {i+1:5d}  {sym:<2s}  MOL A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           {sym:>2s}\n"
            )
        f.write("END\n")
    return filename


def read_pdb(filename):
    """Parse atom symbols and coordinates from PDB."""
    symbols = []
    coords = []
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                sym = line[76:78].strip()
                if not sym:
                    sym = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                symbols.append(sym)
                coords.append([x, y, z])
    return symbols, np.array(coords)


# ═══════════════════════════════════════════════════════════
#  STAGE 1: AXIOM SPACE
#  Define the mathematical universe for the molecule
# ═══════════════════════════════════════════════════════════

def stage1_axiom_space(symbols, coordinates, basis="sto-3g"):
    """
    Select Hilbert space H, identify symmetry group G,
    define inner product <a|b>.
    
    Returns: axiom_space dict with all structural info
    """
    print("\n" + "="*60)
    print("  STAGE 1: AXIOM SPACE")
    print("  Defining the mathematical universe")
    print("="*60)
    
    n_electrons = sum({"H": 1, "He": 2, "Li": 3, "C": 6, "N": 7, "O": 8}[s] for s in symbols)
    
    # Number of molecular orbitals (qubits needed)
    # For STO-3G: 1 basis function per H, 5 per Li/C/N/O
    n_orbitals_map = {"H": 1, "He": 1, "Li": 5, "C": 5, "N": 5, "O": 5}
    n_orbitals = sum(n_orbitals_map.get(s, 5) for s in symbols)
    n_qubits = 2 * n_orbitals  # spin orbitals
    
    # Hilbert space dimension
    hilbert_dim = 2 ** n_qubits
    
    # Identify molecular symmetry
    # For diatomic: D∞h (homonuclear) or C∞v (heteronuclear)
    if len(symbols) == 2:
        if symbols[0] == symbols[1]:
            symmetry_group = "D∞h (homonuclear diatomic)"
        else:
            symmetry_group = "C∞v (heteronuclear diatomic)"
    else:
        symmetry_group = "C1 (general)"
    
    axiom_space = {
        "symbols": symbols,
        "n_electrons": n_electrons,
        "n_orbitals": n_orbitals,
        "n_qubits": n_qubits,
        "hilbert_dim": hilbert_dim,
        "symmetry_group": symmetry_group,
        "basis_set": basis,
        "inner_product": "Fock space overlap <ψ|φ>",
    }
    
    print(f"  Molecule: {''.join(symbols)}")
    print(f"  Electrons: {n_electrons}")
    print(f"  Molecular orbitals: {n_orbitals}")
    print(f"  Qubits needed: {n_qubits}")
    print(f"  Hilbert space dim: 2^{n_qubits} = {hilbert_dim}")
    print(f"  Symmetry group G: {symmetry_group}")
    print(f"  Basis set: {basis}")
    print(f"  Inner product: Fock space <ψ|φ>")
    
    return axiom_space


# ═══════════════════════════════════════════════════════════
#  STAGE 2: HAMILTONIAN CONSTRUCTION
#  Build the energy operator from molecular integrals
# ═══════════════════════════════════════════════════════════

def stage2_hamiltonian(symbols, coordinates, axiom_space):
    """
    Construct the molecular Hamiltonian:
    1. Compute 1- and 2-electron integrals (Lagrangian)
    2. Apply Jordan-Wigner transform (least action → qubit space)
    3. Output qubit Hamiltonian H_hat
    """
    print("\n" + "="*60)
    print("  STAGE 2: HAMILTONIAN CONSTRUCTION")
    print("  Building the energy landscape from molecular data")
    print("="*60)
    
    # Convert coordinates to Bohr (atomic units)
    coords_bohr = np.array(coordinates) / 0.529177  # Angstrom to Bohr
    
    print(f"  Computing electron integrals ({axiom_space['basis_set']})...")
    t0 = time.time()
    
    # Build the molecular Hamiltonian using PennyLane qchem
    H, n_qubits = qchem.molecular_hamiltonian(
        symbols,
        coords_bohr,  # PennyLane expects Bohr
        basis=axiom_space["basis_set"],
        mapping="jordan_wigner",
    )
    
    dt = time.time() - t0
    
    # Count terms in the Hamiltonian
    coeffs = H.terms()[0]
    ops = H.terms()[1]
    n_terms = len(coeffs)
    
    # Analyze Hamiltonian structure
    identity_terms = sum(1 for op in ops if len(op.wires) == 0 or str(op) == "I")
    pauli_z_terms = sum(1 for op in ops if "Z" in str(op) and "X" not in str(op) and "Y" not in str(op))
    mixed_terms = n_terms - identity_terms - pauli_z_terms
    
    print(f"  Hamiltonian built in {dt:.2f}s")
    print(f"  Qubits: {n_qubits}")
    print(f"  Pauli terms: {n_terms}")
    print(f"  Nuclear repulsion (constant): included")
    print(f"  Mapping: Jordan-Wigner (fermion → qubit)")
    
    hamiltonian_info = {
        "H": H,
        "n_qubits": n_qubits,
        "n_terms": n_terms,
        "build_time": dt,
    }
    
    return hamiltonian_info


# ═══════════════════════════════════════════════════════════
#  STAGE 3: QUANTUM EVOLUTION
#  Parameterized unitary circuit (VQE ansatz)
# ═══════════════════════════════════════════════════════════

def stage3_quantum_evolution(hamiltonian_info, axiom_space, max_iterations=80):
    """
    Evolve quantum state through parameterized unitary gates:
    1. Prepare initial state |ψ₀⟩ (Hartree-Fock)
    2. Apply variational unitary U(θ)
    3. Optimize θ via gradient descent (least action principle)
    """
    print("\n" + "="*60)
    print("  STAGE 3: QUANTUM EVOLUTION")
    print("  Propagating through entangled layers")
    print("="*60)
    
    H = hamiltonian_info["H"]
    n_qubits = hamiltonian_info["n_qubits"]
    n_electrons = axiom_space["n_electrons"]
    
    # Initial state: Hartree-Fock (fill lowest orbitals)
    hf_state = qchem.hf_state(n_electrons, n_qubits)
    print(f"  Initial state |ψ₀⟩: Hartree-Fock {hf_state}")
    
    # Build the quantum device
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # Number of layers in the variational circuit
    # Use AllSinglesDoubles for chemically-inspired ansatz
    singles, doubles = qchem.excitations(n_electrons, n_qubits)
    print(f"  Single excitations: {len(singles)}")
    print(f"  Double excitations: {len(doubles)}")
    
    n_params = len(singles) + len(doubles)
    print(f"  Variational parameters: {n_params}")
    
    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        qml.AllSinglesDoubles(
            params, 
            wires=range(n_qubits),
            hf_state=hf_state,
            singles=singles,
            doubles=doubles,
        )
        return qml.expval(H)
    
    # Optimization: gradient descent (= least action principle)
    optimizer = qml.GradientDescentOptimizer(stepsize=0.4)
    params = pnp.zeros(n_params, requires_grad=True)
    
    print(f"\n  Running VQE optimization ({max_iterations} iterations)...")
    print(f"  {'Iter':>6s}  {'Energy (Ha)':>12s}  {'ΔE':>12s}")
    print(f"  {'-'*34}")
    
    energies = []
    prev_energy = 0.0
    convergence_history = []
    
    t0 = time.time()
    for i in range(max_iterations):
        params, energy = optimizer.step_and_cost(circuit, params)
        energy_val = float(energy)
        delta = abs(energy_val - prev_energy)
        energies.append(energy_val)
        convergence_history.append({"iter": i+1, "energy": energy_val, "delta": delta})
        
        if (i+1) % 10 == 0 or i == 0 or delta < 1e-6:
            print(f"  {i+1:6d}  {energy_val:12.6f}  {delta:12.2e}")
        
        if delta < 1e-7 and i > 5:
            print(f"  Converged at iteration {i+1}!")
            break
        prev_energy = energy_val
    
    dt = time.time() - t0
    final_energy = energies[-1]
    
    print(f"\n  Evolution complete in {dt:.2f}s")
    print(f"  Final energy: {final_energy:.6f} Ha")
    print(f"  Optimal parameters θ*: [{', '.join(f'{p:.4f}' for p in params[:4])}{'...' if len(params) > 4 else ''}]")
    
    evolution_result = {
        "final_energy": final_energy,
        "optimal_params": params.tolist() if hasattr(params, 'tolist') else list(params),
        "convergence": convergence_history,
        "n_iterations": len(energies),
        "evolution_time": dt,
        "circuit": circuit,
    }
    
    return evolution_result


# ═══════════════════════════════════════════════════════════
#  STAGE 4: MEASUREMENT COLLAPSE
#  Extract classical answer with uncertainty
# ═══════════════════════════════════════════════════════════

def stage4_measurement(evolution_result, axiom_space, known_exact=None):
    """
    Collapse quantum state to classical observable:
    1. Choose observable M_hat (energy)
    2. Compute ⟨ψ|M|ψ⟩ (expectation value)
    3. Compare with exact known value
    """
    print("\n" + "="*60)
    print("  STAGE 4: MEASUREMENT COLLAPSE")
    print("  Extracting classical answer from quantum state")
    print("="*60)
    
    final_energy = evolution_result["final_energy"]
    convergence = evolution_result["convergence"]
    
    # Estimate uncertainty from convergence tail
    if len(convergence) > 5:
        tail_energies = [c["energy"] for c in convergence[-5:]]
        uncertainty = np.std(tail_energies)
    else:
        uncertainty = abs(convergence[-1]["delta"])
    
    print(f"  Observable: Total electronic energy")
    print(f"  ⟨ψ|H|ψ⟩ = {final_energy:.6f} Ha")
    print(f"  Uncertainty: ±{uncertainty:.2e} Ha")
    print(f"  Energy in eV: {final_energy * 27.2114:.4f} eV")
    
    result = {
        "energy_hartree": final_energy,
        "energy_ev": final_energy * 27.2114,
        "uncertainty": uncertainty,
    }
    
    if known_exact is not None:
        error = abs(final_energy - known_exact)
        accuracy = (1 - error / abs(known_exact)) * 100
        result["known_exact"] = known_exact
        result["absolute_error"] = error
        result["accuracy_pct"] = accuracy
        
        print(f"\n  ── Comparison with exact solution ──")
        print(f"  Known exact:  {known_exact:.6f} Ha")
        print(f"  AQI result:   {final_energy:.6f} Ha")
        print(f"  Error:        {error:.6f} Ha ({error*27.2114*1000:.2f} meV)")
        print(f"  Accuracy:     {accuracy:.4f}%")
    
    return result


# ═══════════════════════════════════════════════════════════
#  DISSOCIATION CURVE SCAN
#  Scan bond distance to map the full energy landscape
# ═══════════════════════════════════════════════════════════

def run_dissociation_scan(symbols, axis=0, distances=None):
    """Run AQI at multiple bond distances to build dissociation curve."""
    if distances is None:
        distances = np.arange(0.4, 2.5, 0.15)
    
    print("\n" + "="*60)
    print("  DISSOCIATION CURVE SCAN")
    print(f"  Scanning {len(distances)} bond distances")
    print("="*60)
    
    scan_results = []
    for i, d in enumerate(distances):
        print(f"\n  ── Distance {d:.2f} Å ({i+1}/{len(distances)}) ──")
        coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        
        try:
            axiom = stage1_axiom_space(symbols, coords, basis="sto-3g")
            ham = stage2_hamiltonian(symbols, coords, axiom)
            evol = stage3_quantum_evolution(ham, axiom, max_iterations=60)
            
            scan_results.append({
                "distance": float(d),
                "energy": evol["final_energy"],
                "n_iterations": evol["n_iterations"],
            })
        except Exception as e:
            print(f"  ERROR at d={d:.2f}: {e}")
            scan_results.append({
                "distance": float(d),
                "energy": None,
                "error": str(e),
            })
    
    return scan_results


# ═══════════════════════════════════════════════════════════
#  MAIN: Run the full AQI pipeline
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═"*58 + "╗")
    print("║  AXIOMATIC QUANTUM INTELLIGENCE (AQI)                   ║")
    print("║  Molecular Simulation Pipeline                          ║")
    print("║  Test: H₂ molecule with known exact solution            ║")
    print("╚" + "═"*58 + "╝")
    
    # ── Create test PDB ──
    symbols = ["H", "H"]
    bond_length = 0.735  # Angstrom (equilibrium)
    coords = np.array([[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]])
    
    pdb_file = write_pdb(symbols, coords, "/home/claude/h2_test.pdb")
    print(f"\n  PDB written: {pdb_file}")
    
    # Read it back (proving the PDB pipeline works)
    sym_read, coords_read = read_pdb(pdb_file)
    print(f"  PDB read: {sym_read}, coords shape: {coords_read.shape}")
    
    # Known exact ground state energy for H2 at 0.735 Å (STO-3G)
    EXACT_H2_STO3G = -1.1373  # Hartree
    
    # ── Run 4-stage pipeline ──
    t_total = time.time()
    
    axiom_space = stage1_axiom_space(sym_read, coords_read)
    hamiltonian = stage2_hamiltonian(sym_read, coords_read, axiom_space)
    evolution = stage3_quantum_evolution(hamiltonian, axiom_space, max_iterations=80)
    measurement = stage4_measurement(evolution, axiom_space, known_exact=EXACT_H2_STO3G)
    
    total_time = time.time() - t_total
    
    # ── Dissociation curve (fewer points for speed) ──
    scan_distances = np.arange(0.5, 2.6, 0.25)
    scan_results = run_dissociation_scan(symbols, distances=scan_distances)
    
    # ── Save all results ──
    output = {
        "molecule": "H2",
        "pdb_file": pdb_file,
        "bond_length_angstrom": bond_length,
        "total_pipeline_time": total_time,
        "stage1_axiom_space": {
            "n_electrons": axiom_space["n_electrons"],
            "n_qubits": axiom_space["n_qubits"],
            "hilbert_dim": axiom_space["hilbert_dim"],
            "symmetry_group": axiom_space["symmetry_group"],
            "basis_set": axiom_space["basis_set"],
        },
        "stage2_hamiltonian": {
            "n_terms": hamiltonian["n_terms"],
            "n_qubits": hamiltonian["n_qubits"],
            "build_time": hamiltonian["build_time"],
        },
        "stage3_evolution": {
            "final_energy": evolution["final_energy"],
            "n_iterations": evolution["n_iterations"],
            "evolution_time": evolution["evolution_time"],
            "convergence": evolution["convergence"],
        },
        "stage4_measurement": measurement,
        "dissociation_scan": scan_results,
    }
    
    with open("/home/claude/aqi_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Results saved: /home/claude/aqi_results.json")
    print("="*60)
