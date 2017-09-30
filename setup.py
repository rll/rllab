# setup.py
from setuptools import setup

setup(
    name='rllab',
    packages=[package for package in find_packages()
                if package.startswith('rllab')],
    version='0.1.0',
)
