"""Shared test helper: real QWS 2.0 instances, skipped when the data is absent."""
import pytest

from qos.datasets import DATA_DIR, make_instance


def instance(n_tasks, n_candidates, tightness=0.0, seed=None):
    """A QWS 2.0 composition instance; skips the test if the dataset is not downloaded."""
    if not (DATA_DIR / "qws" / "qws2.txt").exists():
        pytest.skip("QWS 2.0 not downloaded: run scripts/download_data.py")
    return make_instance("qws", n_tasks, n_candidates, tightness=tightness, seed=seed)
