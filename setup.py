from distutils.core import setup

try:
    # for python 3, let 2to3 do most of the work
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # for python 2 don't apply any transformations
    from distutils.command.build_py import build_py

setup(
    name="ngmix",
    author="Erin Sheldon",
    url="https://github.com/esheldon/ngmix",
    description="fast 2-d gaussian mixtures for modeling astronomical images",
    packages=['ngmix', 'ngmix.tests'],
    version="1.3.4",
    cmdclass={'build_py': build_py},
)
