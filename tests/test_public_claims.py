from pathlib import Path


def test_readme_keeps_hardware_claims_inside_qualified_boundary():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "does **not** promise universal execution" in readme
    assert "Bounded local quantum simulation" in readme
    assert "Bounded float32 matrix multiplication only" in readme
    assert "not physical-QPU evidence" in readme
    assert "Selecting a target does not qualify it" in readme
    assert "True hardware-agnostic execution" not in readme
    assert "All modules have been validated" not in readme
    assert "trapped-ion quantum hardware or simulator" not in readme
    assert "$0.01/shot" not in readme
