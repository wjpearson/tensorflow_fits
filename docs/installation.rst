Installation
============

Tensorflow 2.x FITS can be installed with pip:

.. code:: bash

    pip install tf-fits

Tensorflow 2.x FITS can be installed from source. Download the latest release 
from the `GitHub repository <https://github.com/wjpearson/tensorflow_fits>`_ 
and extract the contents from the zip file. Or clone the repo:

.. code:: bash

    git clone https://github.com/wjpearson/tensorflow_fits.git

Then go into the extracted or cloned folder and install with pip (reccomended):

.. code:: bash

    pip install .

or python:

.. code:: bash

    python3 setup.py install

**Test the installation**

If you installed from source and have ``pytest`` installed, you can run

.. code:: bash

    pytest

from the root directory. Alternativly, you can run:

.. code:: bash

    python -c "from tf_fits.test.test_tf_fits import runall; runall()"
