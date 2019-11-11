from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='gen-elo',
    version=getenv("VERSION", "LOCAL"),
    description='Generalised Elo model',
    packages=find_packages()
)
