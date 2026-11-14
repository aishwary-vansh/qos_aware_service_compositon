"""Real QoS datasets: QWS 2.0 and WS-DREAM dataset 1.

Both datasets are measurements of real web services. They are used to build
composition instances: the services are split into N abstract tasks with M
candidates each. Neither dataset labels services by function, so the split is
random (seeded), the usual practice in QoS-composition studies.

Attribute sources
-----------------
QWS 2.0 (2,507 services, Al-Masri and Mahmoud):
    latency     = Response Time (ms), measured
    reliability = Reliability (%) / 100, measured
WS-DREAM dataset 1 (339 users x 5,825 services, Zheng et al.):
    latency     = response time seen by one user (s -> ms), measured.
                  Each instance is built for one randomly chosen user.
    reliability = share of the 339 users whose invocation returned a valid
                  measurement (the dataset marks the rest with -1). This is an
                  observed invocation success ratio, used as a reliability proxy.

Cost: neither dataset records a price. Each service is mapped to an AWS EC2
t3 on-demand instance size (us-east-1, Linux) by its measured throughput, used
as an indicator of provisioned capacity: the higher the throughput rank, the
larger and more expensive the instance. Cost is that instance's hourly price,
so a composite's cost is the hourly hosting cost of its services.

Download the raw files with ``python scripts/download_data.py``.
"""
from __future__ import annotations

import functools
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from .model import ProblemInstance

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DATASETS = ("qws", "wsdream")

# AWS EC2 t3 on-demand prices, Linux, us-east-1, USD per hour:
# nano, micro, small, medium, large, xlarge.
EC2_T3_HOURLY_USD = np.array([0.0052, 0.0104, 0.0208, 0.0416, 0.0832, 0.1664])

# WS-DREAM services observed successfully by fewer users than this are
# treated as dead and dropped.
MIN_SUCCESS_RATIO = 0.5


@dataclass(frozen=True)
class ServicePool:
    """All usable services of one dataset, after cleaning."""

    name: str
    reliability: np.ndarray        # (S,) in (0, 1]
    cost: np.ndarray               # (S,) USD per hour
    latency: np.ndarray            # (S,) ms; per-user matrix for WS-DREAM below
    user_latency: np.ndarray | None = None   # (U, S) ms, NaN where invalid
    dropped: int = 0               # rows removed during cleaning

    @property
    def size(self) -> int:
        return self.reliability.size


def price_by_throughput(throughput: np.ndarray) -> np.ndarray:
    """Hourly price of the EC2 tier matching each service's throughput rank.

    Services are split into equal-sized groups by throughput percentile, and
    group k gets the k-th instance size. Ties share the average rank.
    """
    pct = (rankdata(throughput) - 0.5) / throughput.size
    tier = np.minimum((pct * EC2_T3_HOURLY_USD.size).astype(int), EC2_T3_HOURLY_USD.size - 1)
    return EC2_T3_HOURLY_USD[tier]


@functools.lru_cache(maxsize=None)
def load_qws(path: pathlib.Path | None = None) -> ServicePool:
    """Load QWS 2.0 and drop rows with impossible values."""
    path = path or DATA_DIR / "qws" / "qws2.txt"
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            if line[:1].isdigit():
                rows.append([float(v) for v in line.split(",")[:9]])
    data = np.array(rows)
    response_time, throughput, reliability = data[:, 0], data[:, 2], data[:, 4]
    valid = ((response_time > 0) & (reliability > 0) & (reliability <= 100)
             & (throughput >= 0))
    return ServicePool(
        name="QWS 2.0",
        latency=response_time[valid],
        reliability=reliability[valid] / 100.0,
        cost=price_by_throughput(throughput[valid]),
        dropped=int((~valid).sum()),
    )


@functools.lru_cache(maxsize=None)
def load_wsdream(directory: pathlib.Path | None = None) -> ServicePool:
    """Load WS-DREAM dataset 1 and derive per-service reliability and cost."""
    directory = directory or DATA_DIR / "wsdream" / "dataset1"
    rt = np.loadtxt(directory / "rtMatrix.txt")
    tp = np.loadtxt(directory / "tpMatrix.txt")
    valid = rt >= 0                       # -1 marks an invalid invocation
    success = valid.mean(axis=0)
    keep = success >= MIN_SUCCESS_RATIO
    user_latency = np.where(valid, rt * 1000.0, np.nan)[:, keep]
    tp_valid = np.where(tp >= 0, tp, np.nan)[:, keep]
    has_tp = ~np.isnan(tp_valid).all(axis=0)
    tp_median = np.zeros(has_tp.size)     # no valid throughput: lowest tier
    tp_median[has_tp] = np.nanmedian(tp_valid[:, has_tp], axis=0)
    return ServicePool(
        name="WS-DREAM",
        latency=np.nanmedian(user_latency, axis=0),
        reliability=success[keep],
        cost=price_by_throughput(tp_median),
        user_latency=user_latency,
        dropped=int((~keep).sum()),
    )


def load(dataset: str) -> ServicePool:
    if dataset == "qws":
        return load_qws()
    if dataset == "wsdream":
        return load_wsdream()
    raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")


def needs_reuse(dataset: str, n_tasks: int, n_candidates: int) -> bool:
    """True when an instance is too large for every service to be distinct."""
    pool = load(dataset)
    if pool.user_latency is None:
        return n_tasks * n_candidates > pool.size
    observed = (~np.isnan(pool.user_latency)).sum(axis=1)
    return n_tasks * n_candidates > observed.max()


def make_instance(dataset: str, n_tasks: int, n_candidates: int, tightness=0.0,
                  seed=None, weights=(1 / 3, 1 / 3, 1 / 3)) -> ProblemInstance:
    """Build one composition instance from real services.

    ``n_tasks * n_candidates`` distinct services are drawn without
    replacement, so no service appears in two tasks. For WS-DREAM the
    instance is built for one random user, using that user's measured latency
    and only the services that user observed successfully.

    If the dataset is too small for that (see ``needs_reuse``), each task
    draws its own ``n_candidates`` distinct services instead, so a service can
    appear in several tasks but never twice in one task.
    """
    pool = load(dataset)
    rng = np.random.default_rng(seed)
    needed = n_tasks * n_candidates
    reuse = needs_reuse(dataset, n_tasks, n_candidates)
    if pool.user_latency is None:
        available = np.arange(pool.size)
    else:
        observed = ~np.isnan(pool.user_latency)
        users = np.flatnonzero(observed.sum(axis=1) >= (n_candidates if reuse else needed))
        if users.size == 0:
            raise ValueError(f"no WS-DREAM user observed {needed} services")
        user = rng.choice(users)
        available = np.flatnonzero(observed[user])
    if n_candidates > available.size:
        raise ValueError(f"{dataset} has {available.size} services, need {n_candidates} per task")
    if reuse:
        chosen = np.concatenate([rng.choice(available, size=n_candidates, replace=False)
                                 for _ in range(n_tasks)])
    else:
        chosen = rng.choice(available, size=needed, replace=False)
    if pool.user_latency is None:
        latency = pool.latency[chosen]
    else:
        latency = pool.user_latency[user, chosen]
    shape = (n_tasks, n_candidates)
    return ProblemInstance(latency.reshape(shape), pool.cost[chosen].reshape(shape),
                           pool.reliability[chosen].reshape(shape),
                           weights=weights, tightness=tightness)
