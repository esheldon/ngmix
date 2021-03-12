__all__ = [
    'DEFAULT_STEP',
    'METACAL_TYPES', 'METACAL_MINIMAL_TYPES',
]


# need all these types for psf='dilate'
METACAL_TYPES = [
    'noshear',
    '1p', '1m', '2p', '2m',
    '1p_psf', '1m_psf', '2p_psf', '2m_psf',
]

# these are the types needed when the new psf is round
METACAL_MINIMAL_TYPES = [
    'noshear',
    '1p', '1m', '2p', '2m',
]

DEFAULT_STEP = 0.01
