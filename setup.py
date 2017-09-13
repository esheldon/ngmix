import distutils
from distutils.core import setup, Extension, Command
import numpy

sources=["ngmix/_gmix.c"]
include_dirs=[numpy.get_include()]

ext=Extension(
    "ngmix._gmix",
    sources,
    extra_compile_args=['-fopenmp'],
    include_dirs=include_dirs,
)

# intel
'''
ext=Extension(
    "ngmix._gmix",
    sources,
    extra_compile_args=['-openmp'],
    extra_link_args=[
        '-L/opt/astro/SL64/packages/intel/Compiler/14.0.1/lib/intel64',
        '-liomp5',
        '-lirc',
        '-limf',
        '-lsvml',
        '-lm'
    ],
    include_dirs=include_dirs,
)
'''


setup(
    name="ngmix", 
    packages=['ngmix'],
    version="0.9.3",
    ext_modules=[ext],
)




