from setuptools import setup
from setuptools import find_packages


setup(name='rllab',
      version='0.0.0',
      description='rllab is a framework for developing and evaluating reinforcement learning algorithms, fully compatible with OpenAI Gym.',
      author='OpenAI',
      url='https://github.com/openai/rllab',
      license='MIT',
      install_requires=[],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())
