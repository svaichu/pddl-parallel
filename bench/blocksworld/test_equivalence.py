import pytest
from check import verify

@pytest.mark.parametrize("N,E,T", [(4, 30, 30), (8, 16, 20)])
def test_matches_reference(N, E, T):
    assert verify(N, E, T, verbose=False)
