from ..util import print_pars
import logging

logger = logging.getLogger(__name__)


def test_print_pars():

    pars = [0.2, -0.1, 0.2, 0.3, 15.0, 123.1]
    print_pars(pars)
    print_pars(pars, front='pars: ')
    print_pars(pars, logger=logger)
