"""
model objects for the pre-PSF adaptive moments fitter

The fitter machinery in prepsfadmom.py is model independent: it
preps the epochs, accumulates the k-space moment sums under the
current weight, and packages the result.  Everything specific to
the object model lives on the model objects: the adaptive moments
driver, the profile component table, the flux normalization, the
error sandwich, and how the size and shape are defined by the
converged state.  The base class is in base_model.py and the
concrete models in gauss_model.py, family_model.py, bdf_model.py
and star_model.py; this module collects them and provides the
get_padmom_model factory.

model= for PAdmomFitter and run_prepsf_admom accepts one of these
objects, e.g.

    PAdmomFitter(model=BDFModel(TdByTe=1.0))

a string naming one ('gauss', 'exp', 'dev', 'star'), or the
equivalent dict spec; see get_padmom_model.  Model instances are
immutable configuration and can be shared between fitters; all
per-run quantities live in the model state dict threaded through
the fit.
"""
__all__ = [
    'PAdmomModel', 'GaussModel', 'FamilyModel', 'ExpModel',
    'DevModel', 'BDFModel', 'StarModel', 'get_padmom_model',
]

from .base_model import PAdmomModel
from .gauss_model import GaussModel
from .family_model import FamilyModel, ExpModel, DevModel
from .bdf_model import BDFModel
from .star_model import StarModel


def get_padmom_model(model):
    """
    normalize a model specification to a PAdmomModel instance

    Parameters
    ----------
    model: PAdmomModel, str or dict
        A model object is returned unchanged.  A string names the
        type: 'gauss', 'exp', 'dev' or 'star' ('bdf' requires the
        dict or object form for TdByTe).  The dict form is
        {'type': name} plus, for 'bdf' only, the required 'TdByTe'
        entry (the dev to exp size ratio) and the optional
        shrinkage pair 'fracdev0' and 'fracdev_sigma0' (see
        BDFModel).  Unknown types and unexpected entries raise

    Returns
    -------
    PAdmomModel
    """
    if isinstance(model, PAdmomModel):
        return model

    if isinstance(model, str):
        model = {'type': model}
    else:
        model = dict(model)

    if 'type' not in model:
        raise ValueError("model dict must have a 'type' entry")
    mtype = model['type']
    if mtype not in ('gauss', 'exp', 'dev', 'star', 'bdf'):
        raise ValueError(
            f"bad model '{mtype}', expected 'gauss', 'exp', 'dev', "
            "'star' or 'bdf'"
        )

    if mtype == 'bdf':
        if 'TdByTe' not in model:
            raise ValueError(
                "the bdf model requires a 'TdByTe' entry, e.g. "
                "model={'type': 'bdf', 'TdByTe': 1.0}"
            )
        allowed = {'type', 'TdByTe', 'fracdev0', 'fracdev_sigma0'}
    else:
        allowed = {'type'}

    extra = set(model) - allowed
    if extra:
        raise ValueError(
            f"unexpected model entries {sorted(extra)} for "
            f"'{mtype}'"
        )

    if mtype == 'bdf':
        has0 = 'fracdev0' in model
        hass = 'fracdev_sigma0' in model
        if has0 != hass:
            raise ValueError(
                "the bdf shrinkage requires both 'fracdev0' and "
                "'fracdev_sigma0' (or neither)"
            )
        return BDFModel(
            TdByTe=model['TdByTe'],
            fracdev0=model.get('fracdev0'),
            fracdev_sigma0=model.get('fracdev_sigma0'),
        )

    return {
        'gauss': GaussModel,
        'exp': ExpModel,
        'dev': DevModel,
        'star': StarModel,
    }[mtype]()
