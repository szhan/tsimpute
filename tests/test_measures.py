import pytest
import sys

sys.path.append("./src")
import measures


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], "== 1"),
        ([0, 0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 1], "< 0"),
    ]
)
def test_compute_iqs_diploid(a, b, expected):
    iqs = measures.compute_iqs_diploid(a, b)
    assert eval(str(iqs) + " " + expected)
