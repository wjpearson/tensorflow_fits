Tensorflow 2.x FITS
===================
This repo contains python scripts to load data from [FITS (Flexible Image 
Transfer System)](https://en.wikipedia.org/wiki/FITS) into Tensorflow 2.x's 
`tf.data.Dataset`. It currently will load images or binary tables from a 
specified HDU (i.e. your fits file can have multiple extensions). My plan was 
to have it work like Tensorflow's built in functions to read images.

The functions require a byte string of your FITS file, which can be generated 
from Tensorflow's `tf.io.read_file(file_path)` function.

The returned image shape will be the same as the image in the FITS (i.e. 2D 
if the FITS HDU is 2D or 3D if it is 3D and so on).

The returned binary table will have the same number of rows and columns as the 
binary table in the FITS. HOWEVER, data types will not always be conserved. 
Character strings ('A'), complex numbers ('C' and 'M'), array descriptors ('P' 
and 'Q') and bits ('X') will be converted into float32 due to the way 
Tensorflow seems to want to work. Arrays of any data type (e.g. '5E') will be
returned as the first value in the array. For example, columns of data type 
'5E' will become columns of data type 'E' by taking the first value in the 
array. Non-single-precision floating point values will also be converted to
single-precision floating points.

The returned ascii table will have the same number of rows and columns as the 
ascii table in the FITS. HOWEVER, data types will not always be conserved. 
Character strings ('A') will be converted into float32 due to the way 
Tensorflow seems to want to work.

As Tensorflow requires these functions to have a known return type, the data
returned from this script will be tf.float32. If you have double-precision 
values, there may be loss of data. There will also be loss of data for 
non-real numbers in tables (see above).

INSTALLING
==========
Clone this repo: `git clone https://github.com/wjpearson/tensorflow_fits.git`  
cd into the folder: `cd tensorflow_fits`  
install with pip (reccomended): `pip install .`  
or install with python: `python3 setup.py install`

Usage
=====
FITS images:
```python
import tensorflow as tf
from tf_fits.image import image_decode_fits

fits_file = '/path/to/fits/file.fits'
header = 0

img = tf.io.read_file(fits_file)
img = image_decode_fits(img, header)
```

FITS binary tables:
```python
import tensorflow as tf
from tf_fits.bintable import bintable_decode_fits

fits_file = '/path/to/fits/file.fits'
header = 1

tbl = tf.io.read_file(fits_file)
tbl = bintable_decode_fits(tbl, header)
```

FITS ascii tables:
```python
import tensorflow as tf
from tf_fits.asciitable import asciitable_decode_fits

fits_file = '/path/to/fits/file.fits'
header = 1

tbl = tf.io.read_file(fits_file)
tbl = asciitable_decode_fits(tbl, header)
```

If you use this code in a publication, shoot me a message (but don't feel 
obliged). I'm curious what people may use it for.

(Potential) Issues
==================
Data types in binary tables may be lost. Only bools ('L') and real numbers 
(unsigned bytes 'B', 16-bit integers 'I', 32-bit integers 'J', 64-bit integers 
'K', single-precision floating point 'E' and double-precision floating point 
'D') will be processed properly. Other data types (bit 'X', character 'A', 
single-precision complex 'C', double-precision complex 'M', 32-bit array 
descriptor 'P' and 64-bit array descriptor 'Q') will not be returned properly 
along with arrays of bools or real numbers (which will return the first value 
in the array). This is due to Tensorflow wanting arrays of a single data type 
and my assumption that people will use bools, real numbers or split complex 
numbers into separate real and imaginary parts inside Tensorflow.

Data types in ascii tables may be lost. Characters 'A' are not processed 
properly. It will convert these data into integers.This is due to Tensorflow 
wanting arrays of a single data type and my assumption that people will not 
be using characters (or strings) inside Tensorflow.

Reading tables is slow...

Does not check the HDU actually contains the requested XTENSION type (IMAGE or 
BINTABLE)
