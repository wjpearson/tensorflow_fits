.. Tensorflow 2.x FITS documentation master file, created by
   sphinx-quickstart on Mon Oct  5 19:40:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tensorflow 2.x FITS's documentation!
===============================================

Tensorflow 2.x FITS allows you to load data  from `FITS (Flexible Image 
Transfer  System) <https://en.wikipedia.org/wiki/FITS>`_ into Tensorflow 2.x's
``td.data.Dataset``. It can load images, binary tables or ascii tables from a 
specified HDU (i.e. your FITS file can have multiple extensions). My plan was
to have it work like Tensorflow's built in functions to read images.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   tutorial
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
