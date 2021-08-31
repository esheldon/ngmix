## v2.0.1  (not yet released)

### new features

- em fitting now supports fixed covariance matrices using the
  EMFitterFixCov, or equivalently the fixcov=True keyword to the run_em
  convenience function

### bug fixes

- fixed bug with sheared psfs in metacal, negative sheared psfs
  were not being cached.

## v2.0.0

### breaking API changes

- ngmix now evaluates models in flux units rather than surface brightness.
  Users whose images are in flux units previously had to include extra
  factors of pixel scale squared to get the right units, this is no longer
  the case.
- The fitters no longer take the observation in the constructor.
  All fitting/measurement conform to a common interface.  The classes have a `go`
  method that takes the observation and possibly a guess.
- The bootstrap and runner code has been completely replaced by a new api.
  The bootstrappers are now generic, wrapping runners that are themselves
  generic wrappers for fitters and guessers.  Thus the bootstrap/runner
  code is completely generic and can work with any of the fitters/guessers.
- The guessers now optionally take the observation as argument when called; this
  facilitates the cases where a guess can be made based on the observations.
- The em codes take GMix objects as guesses, a simple T guess is no longer
  supported
- The em codes now work in image flux units rather than normalizing the
  mixture
- all PDFs/priors sample methods take the option nrand=.  Some had previously
  taken n=
- fitters renamed:
    - LMSimple -> Fitter
    - LMCoellip -> CoellipFitter
    - TemplateFluxFitter -> PSFFluxFitter
    - GalsimSimple -> GalsimFitter
    - SpergelFitter -> GalsimSpergelFitter
    - MoffatFitter -> GalsimMoffatFitter
    - GalsimTemplateFluxFitter -> GalsimPSFFluxFitter
    - GMixEM -> EMFitter
    - Admom -> AdmomFitter
- Copies of observations now propagate the values of `store_pixels` and `ignore_zero_weight`.

### bug fixes

- metacal was reconvolving by the pixel *twice*, which resulted in larger
  reconvolved PSFs and thus somewhat lower, but still accurate, response than
  the pixel was included once.  This did not cause any bias.
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
- fixed flux unit scaling in metadetect regression tests

### new features

- All fitters and measurement classes adhere to a common interface.
  `fitter.go(obs=obs)` or `fitter.go(obs=obs, guess=guess)`.  The obs
  is no longer taken in the constructor so the same object can be reused
  in the runners/bootstrappers for multiple measurements.
- Replaced boostrapper/runner classes with generic classes/functions that
  expect fitters/measurements and runners to provide common interfaces
  For fitters `fitter.go(obs=obs)` or `fitter.go(obs=obs, guess=guess)`
  For runners `runner.go(obs=obs)`.  New classes are bootstrap.Bootstrapper
  runners.Runner, runners.PSFRunner
- Expanded test suite and improved documentation for all modules
- introduced `NGmixBaseException` as parent class for all ngmix-specific exceptions
- New guessers `TPSFFluxGuesser`, `TPSFFluxAndPriorGuesser`, `GmixPSFGuesser`,
  `SimplePSFGuesser`, `CoellipPSFGuesser`
- New joint prior `PriorGalsimSimpleSep`
- Added guessers `TPSFFluxGuesser`, `TPSFFluxAndPriorGuesser`, `GMixPSFGuesser`,
  `SimplePSFGuesser`, `CoellipPSFGuesser`,
- New specialized EM fitting codes added:
    - GMixEMFixedCen: fit mixtures with fixed centers for each component
    - GMixEMFluxOnly: fit mixtures allowing only the fluxes to vary
- some refactoring of like modules into sub-packages.
- Added `mfrac` attribute to observations to hold masked fraction images.
- Added Python `@property` functions for `store_pixels` and `ignore_zero_weight`
  for `ngmix.Observation`.
- `setup.py` now gets the version from the package.
- Added new `KSigmaMom` fitter to measure pre-PSF weighted moments using the
  `ksigma` kernel from Bernstein et al.

### deprecated/removed

- The `LMSimple` class now supports the "bdf" and "bd" models and has been
  renamed to `LM`.  The `LMSimple` is now an alias for `LM` and is considered
  deprecated.  The `LMBDF` and `LMBD` classes have been removed.
- `ngmix.lensfit` has been removed in v2.0.0
- all `dbyg1_*`,  `dbyg2_*`, `dlnbyg1_*`, `dlnbyg2_*`, and `get_pqr*` methods,
  along with the code used to test them, have been removed from the classes in
  `ngmix.priors`
- removed travis-ci in favor of GitHub Actions
- the optional `rng` keyword for `ngmix.priors.srandu` has been removed in favor
  of a required keyword
- removed priors `ZDisk2DErf`, `ZAnnulus`, `UDisk2DCut`, `TruncatedStudentPolar`,
  `TruncatedGaussianPolar`, `Student`, `Student2D`, `MultivariateLogNormal`,
  `MVNMom`, `TruncatedSimpleGauss2D`, `GPriorGreat3Exp`, `GPriorGreatDES`,
  `GPriorGreatDES2`, `GPriorGreatDESNoExp`, `GPriorM`, `GPriorMErf`, `GPriorMErf2`,
  `FlatEtaPrior`, `BFrac`, `TFluxPriorCosmosExp`, `TFluxPriorCosmosDev`,
  `GPriorCosmosSersicSpline`
- Removed all the old bootstrapper and runner classes; replaced with generic
    `Bootstrapper, Runner, PSFRuner, and MetacalBootstrapper`, and associated functions
- removed `GMixBDF` class.  The "bdf" is now supported by the `GMixModel` class
- removed the `roundify` and `stats` modules
- removed the "prepix" option for metacal
- removed guessers `RoundParsGuesser`, `MomGuesser`,
- switched from print to logging in many modules
- removed simplex.py no longer needed, we just use the scipy simplex fitter

## v1.3.9

### bug fixes

- metacal was reconvolving by the pixel twice, which resulted in larger reconvolved PSFs and thus somewhat lower, but still accurate, response than the pixel was included once. This did not cause any bias.

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
