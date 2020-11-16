from ..priors import PriorBase, GPriorBase


def test_priors_priorbase():
    pr = PriorBase(bounds=None)
    assert not pr.has_bounds()
    pr = PriorBase(bounds=(3, 4))
    assert pr.has_bounds()
    pr = PriorBase(bounds=[3, 4])
    assert pr.has_bounds()
