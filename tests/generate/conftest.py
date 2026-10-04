import json

import numpy as np
import pytest

from tests.helpers.factories import make_item
from vjaudit.config import GeneratorsFile, load_yaml


class FakeRenderer:
    """Stand-in for an HF chat template: serializes messages so structure is checkable."""

    def __call__(self, messages):
        return "<fake>" + json.dumps(messages, ensure_ascii=False) + "<assistant>"


class FakeFrameCache:
    def __init__(self):
        self.calls = 0

    def get(self, video_path, video_id, n, t_start, t_end):
        self.calls += 1
        return np.zeros((n, 8, 8, 3), np.uint8), list(range(n))


@pytest.fixture
def gens():
    return load_yaml("configs/generators.yaml", GeneratorsFile).generators


@pytest.fixture
def items():
    return [make_item(i, item_id=f"msvd:test:{i}") for i in range(20)]
