import tensorflow as tf
from .common_functions.tf_fits_header import _hdu_condition, _hdu_body

@tf.function
def image_decode_fits(fits_data, header):
    """
    Function to decode fits binary table
    
    Parameters
    ----------
    fits_data : byte string
                byte string of data (from `tf.io.read_file(file_path)`)
    header : int
             header to return
    
    Returns
    -------
    img_data : tf.Tensor
               tf.Tensor with dtype `tf.float32` of image data
    """
    
    h = tf.constant(0)
    offset = 0 #Position of start of HDU
    start = 0
    true_length = 0
    fixed_length = 0
    bitpix = 0
    shape = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='shape')
    
    hdu_loop = tf.while_loop(_hdu_condition, _hdu_body,
                             [fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape],
                              shape_invariants=[None,
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                tf.TensorShape(None)])
    _, _, _, offset, start, true_length, fixed_length, bitpix, shape = hdu_loop
    
    #Get btye data for the chosen header
    byte_data = tf.strings.substr(fits_data, offset+start, true_length)
    #Decode the byte data
    if tf.math.equal(bitpix, 8):
        rcvd_data = tf.io.decode_raw(byte_data, tf.int8, False, fixed_length)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 16):
        rcvd_data = tf.io.decode_raw(byte_data, tf.int16, False, fixed_length)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 32):
        rcvd_data = tf.io.decode_raw(byte_data, tf.int32, False, fixed_length)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 64):
        rcvd_data = tf.io.decode_raw(byte_data, tf.int64, False, fixed_length)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, -32):
        rcvd_data = tf.io.decode_raw(byte_data, tf.float32, False, fixed_length)
    else:# tf.math.equal(bitpix, -64):
        rcvd_data = tf.io.decode_raw(byte_data, tf.float64, False, fixed_length)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
        
    #Swap  the first two axes
    ax0 = shape.read(0)
    ax1 = shape.read(1)
    shape = shape.write(0, ax1)
    shape = shape.write(1, ax0)
    
    #Reshape the data
    shape = shape.stack()
    img_data = tf.reshape(rcvd_data, shape)
    
    return img_data