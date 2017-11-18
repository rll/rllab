.. _installation:


============
Installation
============

Preparation
===========

You need to edit your :code:`PYTHONPATH` to include the rllab directory:

.. code-block:: bash

    export PYTHONPATH=path_to_rllab:$PYTHONPATH

Express Install
===============

The fastest way to set up dependencies for rllab is via running the setup script.

- On Linux, run the following:

.. code-block:: bash

    ./scripts/setup_linux.sh

- On Mac OS X, run the following:

.. code-block:: bash

    ./scripts/setup_osx.sh

The script sets up a conda environment, which is similar to :code:`virtualenv`. To start using it, run the following:

.. code-block:: bash

    source activate rllab3


Optionally, if you would like to run experiments that depends on the Mujoco environment, you can set it up by running the following command:

.. code-block:: bash

    ./scripts/setup_mujoco.sh

and follow the instructions. You need to have the zip file for Mujoco v1.31 and the license file ready.



Manual Install
==============

Anaconda
------------

:code:`rllab` assumes that you are using Anaconda Python distribution. You can download it from `https://www.continuum.io/downloads<https://www.continuum.io/downloads>`.  Make sure to download the installer for Python 2.7.


System dependencies for pygame
------------------------------

A few environments in rllab are implemented using Box2D, which uses pygame for visualization.
It requires a few system dependencies to be installed first.

On Linux, run the following:

.. code-block:: bash

  sudo apt-get install swig
  sudo apt-get build-dep python-pygame

On Mac OS X, run the following:

.. code-block:: bash

  brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi

System dependencies for scipy
-----------------------------

This step is only needed under Linux:

.. code-block:: bash

  sudo apt-get install build-dep python-scipy

Install Python modules
----------------------

.. code-block:: bash

  conda env create -f environment.yml
