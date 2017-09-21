#ifndef _PYGMIX_ERRORS_HEADER_GUARD
#define _PYGMIX_ERRORS_HEADER_GUARD

#include <stdlib.h>
#include <stdarg.h>

#define GMIX_RANGE_ERROR 1
#define GMIX_FATAL_ERROR 2

void gmix_set_error(int error, const char *format, ...);

void gmix_get_error(const char **estring, int *error);

#endif
