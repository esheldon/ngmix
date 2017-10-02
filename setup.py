import distutils
from distutils.core import setup, Extension, Command
import numpy

sources=["ngmix/_gmix.c"]
include_dirs=[numpy.get_include()]

ext=Extension("ngmix._gmix", sources, include_dirs=include_dirs)

setup(name="ngmix", 
      packages=['ngmix'],
      version="v0.9.4",
      ext_modules=[ext])




