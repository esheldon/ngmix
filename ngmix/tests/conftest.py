import ngmix.prepsfmom
import ngmix.metacal.metacal

import pytest


@pytest.fixture(
    scope="module",
    params=[(False, False), (True, False), (False, True), (True, True)],
)
def prepsfmom_caching(request):
    if request.param[0]:
        ngmix.prepsfmom.turn_on_fft_caching()
    else:
        ngmix.prepsfmom.turn_off_fft_caching()

    if request.param[1]:
        ngmix.prepsfmom.turn_on_kernel_caching()
    else:
        ngmix.prepsfmom.turn_off_kernel_caching()


@pytest.fixture(
    scope="module",
    params=[False, True],
)
def metacal_caching(request):
    if request.param:
        ngmix.metacal.metacal.turn_on_galsim_caching()
    else:
        ngmix.metacal.metacal.turn_off_galsim_caching()
