import distutils
from distutils.core import setup, Extension, Command
import numpy

sources=["ngmix/_gmix.c"]
include_dirs=[numpy.get_include()]
'''
extra_compile_args=['-fast'
                    '-xSSE4.2',
                    '-no-prec-div',
                    '-opt-prefetch',
                    '-unroll-aggressive',
                    '-m64']
'''

extra_compile_args=[]

# gcc only
# note -O3 didn't make any difference because anaconda was already compiled with it
# -march did make a differencebut but will it work for use with our heterogeneous cluster?
#'-march=native',
# -msse4 didn't matter much on astro0034
# -ffast-math resulted in poor acceptance rates!  but it made a 30% difference...

#extra_compile_args=['-funroll-loops',
#                    #'-ffast-math',
#                    '-march=native']

ext=Extension("ngmix._gmix", sources, extra_compile_args=extra_compile_args)

setup(name="ngmix", 
      packages=['ngmix'],
      version="0.1",
      ext_modules=[ext],
      include_dirs=include_dirs)


