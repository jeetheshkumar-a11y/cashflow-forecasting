"""Smoke test for the demo (needs trained models in artifacts/)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "artifacts" / "calibration.npz").exists(), reason="run the experiment first")
def test_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.metric) == 4
    at.checkbox[0].check().run()  # compare all models
    assert not at.exception
    assert len(at.dataframe) == 1
