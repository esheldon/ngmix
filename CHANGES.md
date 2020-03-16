## v1.3.8 (not released)

## bug fixes

- fix bug not updating pixel array when adding noise in metacal

## v1.3.6

### bug fixes

- fixed bug in T fraction sum for `dev` profiles
- fixed bugs for the full bulge+disk fitter

### new features

- added order 5 fast exponential to fastexp.py which
  is exported as fexp. This has satisfactory accuracy
  but is much faster than expd in some real world
  scenarios.  Modified the tests accordinly.
- added a Gaussian moments fitter
- added 5 gaussian coellip fitting in the coellip
  psf fitter

## v1.3.5

### bug fixes

- better fast exponential function approximation
- bug in gaussian aperture flux calculation for cm

### new features

- unit tests and travis CI for core APIs


## v1.3.4

### new features

- analytic Gaussian aperture fluxes


## v1.3.3

### bug fixes

- fixed bug in BDF Gaussian aperture fluxes


## v1.3.2

### bug fixes

- Use psf_cutout_row and col for center in psf obs for
  meds reader
