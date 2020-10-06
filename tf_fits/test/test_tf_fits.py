import numpy as np
import tensorflow as tf
from tf_fits.image import image_decode_fits
from tf_fits.bintable import bintable_decode_fits
from tf_fits.asciitable import asciitable_decode_fits

import os

__all__ = ['runall']

truth = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2/11,8/11,3/11],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2/11,8/11,1,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2/11,9/11,1,1,1,1,9/11,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2/11,10/11,1,1,1,1,1,1,8/11,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3/11,10/11,1,1,1,1,1,1,1,1,7/11,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/11,7/11,1,1,1,1,1,1,1,1,1,1,9/11,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3/11,10/11,1,1,1,1,1,1,1,1,1,1,1,1,3/11],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4/11,1,1,1,1,1,1,1,1,1,1,1,1,1,1,7/11],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/11,10/11,1,1,1,1,1,1,1,1,1,1,1,1,1,5/11,1/11,0],
                  [0,0,0,0,0,0,0,1/11,6/11,2/11,0,0,0,0,0,0,0,0,0,1/11,10/11,1,1,1,1,1,1,1,1,1,1,1,1,10/11,2/11,1/11,0,0,0],
                  [0,0,0,0,3/11,4/11,9/11,1,1,1,2/11,0,0,0,0,0,0,0,3/11,1,1,1,1,1,1,1,1,1,1,1,1,10/11,2/11,0,0,0,0,0,0],
                  [0,0,0,3/11,9/11,10/11,1,1,1,1,1,2/11,0,0,0,0,0,2/11,1,1,1,1,1,1,1,1,1,1,1,10/11,6/11,0,0,0,0,0,0,0,0],
                  [3/11,7/11,10/11,1,1,1,1,1,1,1,1,1,2/11,0,0,0,2/11,1,1,1,1,1,1,1,1,1,1,1,6/11,0,0,0,0,0,0,0,0,0,0],
                  [0,1/11,2/11,8/11,1,1,1,1,1,1,1,1,1,2/11,0,5/11,1,1,1,1,1,1,1,1,1,1,7/11,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,2/11,10/11,1,1,1,1,1,1,1,4/11,1,1,1,1,1,1,1,1,1,10/11,3/11,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,4/11,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4/11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,6/11,10/11,1,1,1,1,1,1,1,1,1,9/11,3/11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,10/11,1,1,1,1,1,1,1,9/11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,4/11,1,1,1,1,1/11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7/11,10/11,3/11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

def test_image():
    #image
    curdir = os.path.dirname(__file__)
    data = tf.io.read_file(curdir+'/tf_fits_test.fits')
    img = image_decode_fits(data, 0)
    bad = np.where(np.abs(img.numpy()-truth)>1e-7)
    result = np.sum(bad)
    assert result == 0

def test_bintable():
    #bintable
    curdir = os.path.dirname(__file__)
    data = tf.io.read_file(curdir+'/tf_fits_test.fits')
    bint = bintable_decode_fits(data, 1)
    bint = tf.transpose(bint)
    bad = np.where(np.abs(bint.numpy()-truth)>1e-7)
    result = np.sum(bad)
    assert result == 0

def test_asciitable():
    #asciitable
    curdir = os.path.dirname(__file__)
    data = tf.io.read_file(curdir+'/tf_fits_test.fits')
    asct = asciitable_decode_fits(data, 2)
    asct = tf.transpose(asct)
    bad = np.where(np.abs(asct.numpy()-truth)>1e-7)
    result = np.sum(bad)
    assert result == 0

def runall():
    print('Running tf-fits test...')
    print('Testing tf_fits.image...')
    test_image()
    print('Testing tf_fits.bintable...')
    test_bintable()
    print('Testing tf_fits.asciitable...')
    test_asciitable()
    print('All tests completed successfully')
    return