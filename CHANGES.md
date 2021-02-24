## unreleased (v2.0.0)

### bug fixes

- fixed computation of Jacobian center for `ngmix.KObservation`s with mixed
  even-odd dimensions
- the total shape amplitude `g` is now reset properly when using the
  `Shape.set_g1g2` method
- fixed large rounding errors in some shape conversions in `ngmix.shape`
- fixed bug in `ngmix.shape.g1g2_to_eta1eta2` in handling numpy array inputs
- fixed bug in `ngmix.shape.e1e2_to_eta1eta2` in handling numpy array inputs
- fixed bug in `ngmix.priors.GPriorBA.get_fdiff` in handling array inputs
- fixed a bug in `ngmix.priors.TwoSidedErf.get_fdiff` in handling array inputs
- fixed bug in `TruncatedSimpleGauss2D` where the truncation radius was applied about
  zero in some methods and not about the center

### new features

- expanded test suite and improved documentation for the following modules
  * `ngmix.shape`
  * `ngmix.jacobian`
  * `ngmix.moments`
  * `ngmix.observation`
  * `ngmix.priors`
  * `ngmix.gexceptions`
- introduced `NGmixBaseException` as parent class for all ngmix-specific exceptions

### deprecated/removed

- `ngmix.lensfit` has been removed in v2.0.0
- all `dbyg1_*`,  `dbyg2_*`, `dlnbyg1_*`, `dlnbyg2_*`, and `get_pqr*` methods,
  along with the code used to test them, have been removed from the classes in
  `ngmix.priors`
- removed travis-ci in favor of GitHub Actions
- the optional `rng` keyword for `ngmix.priors.srandu` has been removed in favor
  of a required keyword
- removed priors `ZDisk2DErf`, `ZAnnulus`, `UDisk2DCut`, `TruncatedStudentPolar`,
  `TruncatedGaussianPolar`


## v1.3.8

### bug fixes

- fix bug not updating pixel array when adding noise in metacal

### new features

- added writeable context for observations, now returned references
  for observation images etc. are read only unless in the writeable
  context
- jacobian getter returns new Jacobian with readonly view, rather
  than a copy
- added more unit tests

## v1.3.7

### new features

- Add option to not store pixels

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
