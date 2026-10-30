import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression import sh_layout  # noqa: E402
from sh_compression.ply_io import GaussianPly, write_gaussian_ply  # noqa: E402

NON_SH_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)

# Field order matches scene/gaussian_model.py construct_list_of_attributes()
FIELD_ORDER = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + sh_layout.DC_FIELDS
    + sh_layout.REST_FIELDS
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def make_tiny_model(n: int = 100, seed: int = 42) -> GaussianPly:
    rng = np.random.default_rng(seed)
    fields = {name: rng.normal(size=n).astype(np.float32) for name in FIELD_ORDER}
    return GaussianPly(field_names=list(FIELD_ORDER), fields=fields)


@pytest.fixture
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def tiny_ply(tmp_path, tiny_model):
    path = str(tmp_path / "tiny.ply")
    write_gaussian_ply(tiny_model, path)
    return path
