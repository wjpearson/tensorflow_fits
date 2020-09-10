Tensorflow 2.x FITS
===================
This repo contains python scripts to load data from [FITS (Flexible Image 
Transfer System)](https://en.wikipedia.org/wiki/FITS) into Tensorflow 2.x's 
`tf.data.Dataset`. It currently will load images from a specified HDU (i.e. 
your fits file can have multiple extensions). My plan was to have it work 
like Tensorflow's built in functions to read images.

The function requires a byte string of your FITS file, which can be got from 
Tensorflow's `tf.io.read_file(file_path)` function.

The returned image shape will be the same as the image in the FITS (i.e. 2D 
if the FITS HDU is 2D or 3D if it is 3D and so on).

As Tensorflow requires these functions to have a known return type, the images
returned from this script will be tf.float32. If you have double precision 
images, there may be loss of data.

Usage
=====
```python
from tf_fits import *

fits_file = '/path/to/fits/file.fits'
header = 0

img = tf.io.read_file(fits_file)
img = image_decode_fits(img, header)
```

If you use this code in a publication, shoot me a message (but don't feel 
obliged). I'm curois what people may use it for.

(Potential) Issues
==================
All the HDUs before the one you want should probably also be images. I have 
not sorted out how to deal with tables yet so there may be problems.

To Do
=====
Get it to read tables as well as images
