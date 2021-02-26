BAD_VAR = 2 ** 0  # variance not positive definite

LM_SINGULAR_MATRIX = 2 ** 4
LM_NEG_COV_EIG = 2 ** 5
LM_NEG_COV_DIAG = 2 ** 6
EIG_NOTFINITE = 2 ** 7
LM_FUNC_NOTFINITE = 2 ** 8

DIV_ZERO = 2 ** 9  # division by zero
ZERO_DOF = 2 ** 10  # dof zero so can't do chi^2/dof
