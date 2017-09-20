#ifndef _PYGMIX_MOMENTS_HEADER_GUARD
#define _PYGMIX_MOMENTS_HEADER_GUARD

#include "../_gmix.h"

// not yet rewritten as wrappers
PyObject * PyGMix_get_weighted_moments(PyObject* self, PyObject* args);
PyObject * PyGMix_get_weighted_gmix_moments(PyObject* self, PyObject* args);
PyObject * PyGMix_get_unweighted_moments(PyObject* self, PyObject* args);

#endif
