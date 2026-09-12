import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    wheels = sorted(Path("dist").glob("vqpu_sdk-0.6.0-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("exactly one vQPU 0.6.0 wheel is required")
    with tempfile.TemporaryDirectory() as directory:
        environment = Path(directory) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / "bin/python"
        subprocess.run([str(python), "-m", "pip", "install", str(wheels[0])], check=True)
        code = """
import json
from vqpu.compute_engine import execute, plan, verify
w={'workload_type':'quantum.simulation','qubits':2,'shots':128,'seed':7,'gates':[{'name':'H','targets':[0]},{'name':'CNOT','targets':[0,1]}]}
r=execute({'workload':w,'plan_digest':plan(w)['plan_digest']})
print(json.dumps({'verified':verify(r)['verified'],'shots':sum(r['counts'].values()),'backend':r['backend_id']}))
"""
        output = subprocess.run(
            [str(python), "-c", code], check=True, text=True, capture_output=True
        )
        if json.loads(output.stdout) != {
            "verified": True,
            "shots": 128,
            "backend": "cpu.quantum_simulator",
        }:
            raise SystemExit("clean-wheel acceptance failed")
        inventory = subprocess.run(
            [str(python), "-m", "vqpu", "inventory"], check=True, text=True, capture_output=True
        )
        document = json.loads(inventory.stdout)
        if document["fallback_policy"] != "DENIED" or any(
            item["routable"]
            for item in document["backends"]
            if item["backend_id"] in {"hpc.slurm", "cloud.batch", "qpu.physical"}
        ):
            raise SystemExit("clean-wheel inventory acceptance failed")
    print("clean-wheel acceptance passed")


if __name__ == "__main__":
    main()
