from setuptools import setup, find_packages

setup(
    name="ngmix",
    author="Erin Sheldon",
    url="https://github.com/esheldon/ngmix",
    description="fast 2-d gaussian mixtures for modeling astronomical images",
    packages=find_packages(exclude=["mdet_tests"]),
    version="2.0.0",
)
