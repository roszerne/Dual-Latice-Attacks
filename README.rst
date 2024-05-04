This repository contains the code and resources related to my master's thesis titled "Analysis and Implementation of Selected Attacks on Lattice-Based Cryptosystems." It is a fork of the original [General Sieve Kernel (G6K)](https://github.com/fplll/g6k/) repository. The fork was created to contribute modifications, improvements, or experiments without affecting the original project.

******************************
The General Sieve Kernel (G6K)
******************************

.. image:: https://github.com/fplll/g6k/workflows/Tests/badge.svg
    :target: https://github.com/fplll/g6k/actions?query=workflow%3ATests

G6K is a C++ and Python library that implements several Sieve algorithms to be used in more advanced lattice reduction tasks. It follows the stateful machine framework from: 

Martin R. Albrecht and Léo Ducas and Gottfried Herold and Elena Kirshanova and Eamonn W. Postlethwaite and Marc Stevens, 
The General Sieve Kernel and New Records in Lattice Reduction.

The article is available `in this repository <https://github.com/fplll/g6k/blob/master/article.pdf>`__ and on `eprint <https://eprint.iacr.org/2019/089>`__ .


Building the library
====================

You will need the current master of FPyLLL. See ``bootstrap.sh`` for creating (almost) all dependencies from scratch:

.. code-block:: bash

    # once only: creates local python env, builds fplll, fpylll and G6K
    ./bootstrap.sh [ -j # ]
    
    # for every new shell: activates local python env
    source ./activate                   

On systems with co-existing python2 and 3, you can force a specific version installation using ``PYTHON=<pythoncmd> ./boostrap.sh`` instead.
The number of parallel compilation jobs can be controlled with `-j #`.

If building via ```./bootstrap.sh``` fails, then the script will return an error code. 
The error codes are documented in ```bootstrap.sh.```

Otherwise, you will need fplll and fpylll already installed and build the G6K Cython extension like so:

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py build_ext --inplace [ -j # ]

This builds G6K **in place**. Alternatively, you can skip ```--inplace``` and run ```python setup.py install``` as usual after building.
    
It's possible to alter the C++ kernel build configuration as follows:

.. code-block:: bash

    make clean
    ./configure [opts...]           # e.g. opts: --enable-native --enable-templated-dim --with-max-sieving-dim=128
                                    # see ./configure --help for more options
    python setup.py build_ext [ -j # ]

Tests
=====

.. code-block:: bash

    python -m pytest


Gathering test coverage
-----------------------

Uncomment the line ``extra_compile_args += ["-DCYTHON_TRACE=1"]`` in ``setup py.`` and recompile. Then run

.. code-block:: bash

    py.test --cov=g6k
