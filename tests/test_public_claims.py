from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_keeps_hardware_and_research_claims_bounded() -> None:
    text = README.read_text(encoding="utf-8")
    forbidden = (
        "routes circuits to the most suitable device",
        "native gate sequences for any backend",
        "remain valid on real noisy hardware",
        'api_key="your-ionq-api-key-here"',
        'api_key="your-key"',
    )
    for claim in forbidden:
        assert claim not in text
    assert "not physical-hardware or current RAD qualification" in text
    assert 'os.environ["IONQ_API_KEY"]' in text
