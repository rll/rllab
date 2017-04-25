# setup.py
from setuptools import setup, find_packages

setup(
    name='rllab',
    version='0.1.0',
    packages=find_packages(exclude=['sandbox']),
)
