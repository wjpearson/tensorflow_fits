import tensorflow as tf
from ..common_functions.tf_common import _size_and_shape_condition, _size_and_shape_body

#Loop through Header Keywords
@tf.function
def _read_image_header_condition(fits_data, NAXIS, bitpix, offset, i, string):
    '''while loop condition when read next keyword in header
       and get necessary data from the keyword
       fits_data - byte string of data (from tf.io.read_file(file_path))
       NAXIS - all axes sizes
       TFIELDS - number of columns in bintable
       bitpix - fits data type
       offset - end of last header
       i - current keyword
       string - the keyword string
       TFORM - table column data type strings
       
       returns if keyword is not the final keyword'''
    return tf.math.not_equal(string, b'END                                                                             ')

@tf.function
def _read_image_header_body(fits_data, NAXIS, bitpix, offset, i, string):
    '''Function to read next keyword in header
       and get necessary data from the keyword
       fits_data - byte string of data (from tf.io.read_file(file_path))
       NAXIS - all axes sizes
       bitpix - fits data type
       offset - end of last header
       i - current keyword
       string - the keyword string'''
    
    #Get next keyword
    #All keyword lines are 80 characters long
    string = tf.strings.substr(fits_data, offset+(i*80), 80)
    
    #NAXIS - get number of axes
    if tf.math.equal(b'NAXIS   =', tf.strings.substr(string, 0, 9)):
        NAXIS = NAXIS.write(0, int(tf.strings.substr(string, 9, 21)))
        
    #NAXIS# - get size of axis #
    elif tf.math.equal(b'NAXIS', tf.strings.substr(string, 0, 5)):
        axis = int(tf.strings.substr(string, 5, 3))
        NAXIS = NAXIS.write(axis, int(tf.strings.substr(string, 9, 21)))
        
    #BITPIX - get bit size of values
    elif tf.math.equal(b'BITPIX  =', tf.strings.substr(string, 0, 9)):
        bitpix = int(tf.strings.substr(string, 9, 21))
        
    i += 1
    return fits_data, NAXIS, bitpix, offset, i, string

#Loop through HDUs
@tf.function
def _hdu_image_condition(fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape):
    '''while loop condition when read header of HDU h
       called in tf.while_loop
       fits_data - byte string of data (from tf.io.read_file(file_path))
       h - current header
       header - target header
       offset - end of last header
       start - start of data
       true_length - length of data (including padding)
       fixed_length - length of data (excluding padding)
       bitpix - fits data type
       shape - shape of data
       
       returns if h <= header'''
    return tf.math.less_equal(h, header)
    
@tf.function
def _hdu_image_body(fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape):
    '''Function to read header of HDU h
       called in tf.while_loop
       fits_data - byte string of data (from tf.io.read_file(file_path))
       h - current header
       header - target header
       offset - end of last header
       start - start of data
       true_length - length of data (including padding)
       fixed_length - length of data (excluding padding)
       bitpix - fits data type
       shape - shape of data'''
    
    #All header keywords are 80 characters long
    i0 = tf.constant(0)
    i = 0
    NAXIS = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='NAXIS')
    string = tf.strings.substr(fits_data, offset, 80)
    
    read_loop = tf.while_loop(_read_image_header_condition, _read_image_header_body,
                              [fits_data, NAXIS, bitpix, offset, i, string],
                              shape_invariants=[None,
                                                None,
                                                i0.get_shape(),
                                                i0.get_shape(),
                                                i0.get_shape(),
                                                None],
                              name='read')
    _, NAXIS, bitpix, offset, i, _ = read_loop

    i -= 1
    start = i*80
        
    #All data blocks are 2880 bytes long,
    #so find where the data starts including padding
    #For header, padding is spaces: b' '
    tru_start = start // 2880
    if start % 2880 != 0:
        tru_start += 1
    start = tru_start*2880

    #Get bytes per value
    out_size = 0
    if tf.math.equal(bitpix, 8):
        out_size = 1
    elif tf.math.equal(bitpix, 16):
        out_size = 2
    elif tf.math.equal(bitpix, 32):
        out_size = 4
    elif tf.math.equal(bitpix, 64):
        out_size = 8
    elif tf.math.equal(bitpix, -32):
        out_size = 4
    elif tf.math.equal(bitpix, -64):
        out_size = 8

    #Get byte size of data
    #and shape of the final data
    if tf.math.greater(NAXIS.read(0), 0):
        size = 1
        axis = tf.constant(1)
        axis, NAXIS, size, shape = tf.while_loop(_size_and_shape_condition, _size_and_shape_body,
                                                 [axis, NAXIS, size, shape],
                                                 shape_invariants=[axis.get_shape(),
                                                                   None,
                                                                   axis.get_shape(),
                                                                   None],
                                                 name='size_and_shape')                                 
        fixed_length = size*out_size
    else:
        fixed_length = 0

    #All data blocks are 2880 bytes long,
    #so find how long the data is including pading
    #For data, padding is zeros: b'\x00'
    true_length = int(fixed_length//2880)
    if tf.math.not_equal(fixed_length % 2880, 0):
        true_length += 1
    true_length *= 2880
    
    if tf.math.not_equal(h, header):
        #If not the HDU we want,
        #increment offset by HDU size
        offset += start+true_length
        
    h += 1
    
    return fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape