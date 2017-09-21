#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "errors.h"

static char _gmix_error_string[255];
static int _gmix_error;

//
// get the numerical and string versions of errors
//

void gmix_set_error(int error, const char *format, ...)
{

    va_list args;
    va_start(args, format);
    snprintf(_gmix_error_string,
             sizeof(_gmix_error_string),
             format,
             args);
    va_end(args);

    _gmix_error = error;
}

//
// get the numerical and string versions of errors
//
void gmix_get_error(const char **estring, int *error)
{
    *estring = (const char*) _gmix_error_string;
    *error = _gmix_error;
}
