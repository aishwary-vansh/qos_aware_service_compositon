import numpy as np
import pytest

from qos.datasets import (DATA_DIR, load, load_qws, make_instance,
                          needs_reuse)

HAVE_QWS = (DATA_DIR / "qws" / "qws2.txt").exists()
HAVE_WSDREAM = (DATA_DIR / "wsdream" / "dataset1" / "rtMatrix.txt").exists()
needs = {
    "qws": pytest.mark.skipif(not HAVE_QWS, reason="run scripts/download_data.py"),
    "wsdream": pytest.mark.skipif(not HAVE_WSDREAM, reason="run scripts/download_data.py"),
}
DATASETS = [pytest.param(d, marks=m) for d, m in needs.items()]


@pytest.mark.parametrize("dataset", DATASETS)
def test_pool_values_are_valid(dataset):
    pool = load(dataset)
    assert pool.size > 2000
    assert np.all(pool.latency > 0)
    assert np.all((pool.reliability > 0) & (pool.reliability <= 1))


@pytest.mark.parametrize("dataset", DATASETS)
def test_instances_are_reproducible_and_use_distinct_services(dataset):
    a = make_instance(dataset, 10, 20, seed=3, tightness=0.5)
    b = make_instance(dataset, 10, 20, seed=3, tightness=0.5)
    np.testing.assert_array_equal(a.latency, b.latency)
    assert a.latency.shape == (10, 20)
    # The same (latency, reliability) pair twice would hint at a reused service.
    pairs = {(lat, rel) for lat, rel in zip(a.latency.ravel(), a.reliability.ravel())}
    assert len(pairs) > 190


@needs["qws"]
def test_task_larger_than_dataset_rejected():
    with pytest.raises(ValueError):
        make_instance("qws", 2, 3000, seed=0)


@needs["qws"]
def test_large_instance_reuses_services_across_tasks_only():
    assert needs_reuse("qws", 100, 50)
    p = make_instance("qws", 100, 50, seed=0)
    assert p.latency.shape == (100, 50)
    for i in range(100):   # never the same service twice within one task
        assert len(set(zip(p.latency[i], p.reliability[i]))) > 45


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError):
        load("nope")


def test_qws_latency_column_switch(tmp_path):
    # Columns: 0 Response Time, 2 Throughput, 4 Reliability, 7 Latency.
    line = "100,1,50,1,80,1,1,42,1\n"
    file = tmp_path / "qws2.txt"
    file.write_text(line * 3, encoding="latin-1")
    rt = load_qws(path=file)
    lat = load_qws(path=file, latency="latency")
    assert np.allclose(rt.latency, 100.0)
    assert np.allclose(lat.latency, 42.0)


def test_unknown_latency_source_rejected():
    with pytest.raises(ValueError):
        load_qws(latency="nope")
