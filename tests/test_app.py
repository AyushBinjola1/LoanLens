"""Dashboard regressions using the packaged model and real held-out sample."""
from io import BytesIO
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=60).run()


def test_all_pages_and_empty_filters(app):
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.sidebar.radio[0].set_value("Data Explorer").run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No records match" in item.value for item in app.info)


def test_single_prediction_includes_explanation(app):
    app.sidebar.radio[0].set_value("Single Prediction").run()
    app.button[0].click().run()
    app.button[1].click().run()
    assert not app.exception
    assert len(app.metric) == 3
    assert len(app.get("plotly_chart")) == 1
    assert not any("explanation unavailable" in item.value for item in app.info)


@pytest.mark.parametrize("content", [
    (ROOT / "artifacts" / "demo_examples_heldout.csv").read_bytes(),
    b"",
    b"wrong_column\n1\n",
    b'column\n"unterminated',
])
def test_batch_uploads_do_not_crash(app, monkeypatch, content):
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: BytesIO(content))
    app.sidebar.radio[0].set_value("Batch Prediction").run()
    assert not app.exception
    if content.startswith(b"row_id"):
        assert any("Scored 12 rows" in item.value for item in app.success)
    else:
        assert app.error
