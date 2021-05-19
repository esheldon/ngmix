import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "ngmix",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name="ngmix",
    author="Erin Sheldon",
    url="https://github.com/esheldon/ngmix",
    description="fast 2-d gaussian mixtures for modeling astronomical images",
    packages=find_packages(exclude=["mdet_tests"]),
    version=__version__,
)
