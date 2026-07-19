"""AppTest smoke + regression tests for the Genome Firewall Streamlit app.

Run from the repo root:  python -m pytest tests/test_app.py
Guards the text/config corrections in app.py and data/drug_properties.yaml.
"""
import os
import re

import pytest
import yaml
from streamlit.testing.v1 import AppTest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _chdir_repo_root():
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _all_markdown(at):
    return "\n".join(el.value for el in at.markdown)


def test_app_runs_without_exception():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception, f"app raised: {at.exception}"


def test_responsibility_text_is_de_overclaimed():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    md = _all_markdown(at)
    # The false "cannot leak" absolute must be gone...
    assert "cannot leak across splits" not in md
    # ...replaced with the honest ST437/ST258 disclosure.
    assert "ST437" in md and "0.0034" in md
    # Unqualified "distribution-free" claim removed; measured coverage stated.
    assert "distribution-free" not in md.lower()
    assert "exchangeable" in md
    # Gentamicin no longer claims it "reaches nominal coverage".
    assert "reach nominal coverage" not in md and "still short of 0.90" in md
    # New provenance-honesty block present.
    assert "What we could not validate" in md
    assert "upload path is documented but unmeasured" in md


def test_target_gate_not_drawn_as_active_node():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    md = _all_markdown(at)
    # The How-it-works diagram must not present the target-presence gate as a
    # live NO-CALL node on the upload path.
    assert "target-presence gate (drug target absent" not in md
    assert "fires on 0/779 test genomes" in md


def test_ceftazidime_determinants_exclude_narrow_spectrum_penicillinases():
    """blaSHV-11 / blaTEM-1 are narrow-spectrum, not ESBLs: 573.24328 must show
    ceftazidime category (i) via KPC-3 alone, with SHV-11 not a determinant."""
    import app  # noqa: E402  (imported here so AppTest fixtures set cwd first)

    with open("data/drug_properties.yaml") as fh:
        props = yaml.safe_load(fh)
    ceft_patterns = [re.compile(p, re.IGNORECASE)
                     for p in props["ceftazidime"]["resistance_determinant_patterns"]]

    import pandas as pd
    feat_cols = list(pd.read_parquet("data/results/demo_genomes.parquet").columns[7:])
    _vec, gene_names = app.parse_amrfinder_tsv(
        "examples/example_resistant_KPC_573.24328.tsv", feat_cols)

    hits = app.determinant_hits(gene_names, ceft_patterns)
    assert hits == ["KPC-3"], f"expected KPC-3-only, got {hits}"
    assert not any("shv" in h.lower() for h in hits)


def test_narrow_spectrum_alleles_never_match_ceftazidime():
    with open("data/drug_properties.yaml") as fh:
        props = yaml.safe_load(fh)
    pats = [re.compile(p, re.IGNORECASE)
            for p in props["ceftazidime"]["resistance_determinant_patterns"]]

    def matches(tok):
        return any(p.search(tok) for p in pats)

    for narrow in ("shv-11", "blaSHV-11", "shv-1", "shv-28", "tem-1", "blaTEM-1", "tem-2"):
        assert not matches(narrow), f"{narrow} must not be a ceftazidime determinant"
    for esbl in ("shv-12", "shv-2", "shv-5", "ctx-m-15", "kpc-3"):
        assert matches(esbl), f"{esbl} should be a ceftazidime determinant"
