import distutils
from distutils.core import setup, Extension, Command
import numpy

sources=["ngmix/_gmix.c"]
include_dirs=[numpy.get_include()]

ext=Extension("ngmix._gmix", sources)

setup(name="ngmix", 
      packages=['ngmix'],
      version="0.1",
      ext_modules=[ext],
      include_dirs=include_dirs)


