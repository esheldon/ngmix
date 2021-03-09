# flags for LM fitting diagnostics
LM_SINGULAR_MATRIX = 2 ** 4
LM_NEG_COV_EIG = 2 ** 5
LM_NEG_COV_DIAG = 2 ** 6
LM_FUNC_NOTFINITE = 2 ** 8

# for LM this indicates a the eigenvalues of the covariance cannot be found
EIG_NOTFINITE = 2 ** 7

DIV_ZERO = 2 ** 9  # division by zero
ZERO_DOF = 2 ** 10  # dof zero so can't do chi^2/dof

# currently used only in the TemplateFluxFitter for the
# case where the flux_err cannot be calculated

BAD_VAR = 2 ** 0  # variance not positive definite
