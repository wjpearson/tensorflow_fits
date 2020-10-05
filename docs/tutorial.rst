Tutorial - How To Use
=====================

Tensorflow 2.x FITS was initiall designed to work like Tensorflow's built in 
image decoders. If you are used to using those it should hopefully feel 
familiar. Read in the fits file as a binary string using Tensorflow's 
``tf.io.read_file`` and then decode with the relavent ``tf_fits`` decoder. 
All decoders take the binary string as the first argument and the header 
number (zero indexed) as the second argument:

.. code:: python

    XXX_decode_fits(binary_string, header)

where ``XXX`` is ``image``, ``bintable`` or ``asciitable`` (see below).

These functions return an N-dimensional ``tf.float32`` tensor.


Currently, the header type (image, binary table or ascii table) is 
not checked, so make sure your header value is correct!

**FITS Images**

The returned image shape will be the same as the image in the FITS (i.e. 2D 
if the FITS HDU is 2D or 3D if it is 3D and so on). It will be returned as 
a ``tf.float32`` tensor so there may  be data loss if the image is stored 
as double precision floats.

FITS images can be loaded as demonstrated:

.. code:: python

    import tensorflow as tf
    from tf_fits.image import image_decode_fits

    fits_file = '/path/to/fits/file.fits'
    header = 0

    img = tf.io.read_file(fits_file)
    img = image_decode_fits(img, header)

**FITS Binary Tables**

The returned binary table will have the same number of rows and columns as the 
binary table in the FITS. HOWEVER, data types will not always be conserved. 
Character strings ('A'), complex numbers ('C' and 'M'), array descriptors ('P' 
and 'Q') and bits ('X') will be converted into single-precision floating point 
values due to the way Tensorflow seems to want to work. Arrays of any data 
type (e.g. '5E') will be returned as the first value in the array. For 
example, columns of data type '5E' will become columns of data type 'E' by 
taking the first value in the array. Non-single-precision floating point 
values will also be converted to single-precision floating points.

FITS binary tables can be loaded as demonstrated:

.. code:: python

    import tensorflow as tf
    from tf_fits.bintable import bintable_decode_fits

    fits_file = '/path/to/fits/file.fits'
    header = 1

    tbl = tf.io.read_file(fits_file)
    tbl = bintable_decode_fits(tbl, header)

**FITS ASCII Tables**

The returned ascii table will have the same number of rows and columns as the 
ascii table in the FITS. HOWEVER, data types will not always be conserved. 
Character strings ('A') will be converted into single-precision floating point
values due to the way Tensorflow seems to want to work. Non-single-precision 
floating point values will also be converted to single-precision floating 
points.

FITS ASCII tables can be loaded as demonstrated:

.. code:: python

    import tensorflow as tf
    from tf_fits.asciitable import asciitable_decode_fits

    fits_file = '/path/to/fits/file.fits'
    header = 1

    tbl = tf.io.read_file(fits_file)
    tbl = asciitable_decode_fits(tbl, header)
