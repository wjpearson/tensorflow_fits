import tensorflow as tf
from ..common_functions.tf_common import _size_and_shape_condition, _size_and_shape_body

@tf.function
def _TFORM_bitpix(TFORMn):
    '''Function to get TFORM data type
       TFORMn - TFORM data type character
       
       returns bitpix - number of bits for data type'''  
    
    bitpix = 0
    if tf.strings.regex_full_match(TFORMn, b'L'):
        bitpix = 8
    elif tf.strings.regex_full_match(TFORMn, b'X'):
        bitpix = 0
    elif tf.strings.regex_full_match(TFORMn, b'B'):
        bitpix = 8
    elif tf.strings.regex_full_match(TFORMn, b'I'):
        bitpix = 16
    elif tf.strings.regex_full_match(TFORMn, b'J'):
        bitpix = 32
    elif tf.strings.regex_full_match(TFORMn, b'K'):
        bitpix = 64
    elif tf.strings.regex_full_match(TFORMn, b'A'):
        bitpix = 8
    elif tf.strings.regex_full_match(TFORMn, b'E'):
        bitpix = -32
    elif tf.strings.regex_full_match(TFORMn, b'D'):
        bitpix = -64
    elif tf.strings.regex_full_match(TFORMn, b'C'):
        bitpix = -64
    elif tf.strings.regex_full_match(TFORMn, b'M'):
        bitpix = -128
    elif tf.strings.regex_full_match(TFORMn, b'P'):
        bitpix = 32
    else:# tf.strings.regex_full_match(TFORMn, b'Q'):
        bitpix = 64
        
    return bitpix

#Loop through Header Keywords
@tf.function
def _read_bintable_header_condition(fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, TFORM):
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
def _read_bintable_header_body(fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, TFORM):
    '''Function to read next keyword in header
       and get necessary data from the keyword
       fits_data - byte string of data (from tf.io.read_file(file_path))
       NAXIS - all axes sizes
       TFIELDS - number of columns in bintable
       bitpix - fits data type
       offset - end of last header
       i - current keyword
       string - the keyword string
       TFORM - table column data type strings'''
    
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
        
    #TFIELDS - get number of fields
    elif tf.math.equal(b'TFIELDS =', tf.strings.substr(string, 0, 9)):
        TFIELDS = int(tf.strings.substr(string, 9, 21))
        
    #TFORM# - get field # data format
    elif tf.math.equal(b'TFORM', tf.strings.substr(string, 0, 5)):
        field = int(tf.strings.substr(string, 5, 3))-1
        TFORMn = tf.strings.substr(string, 11, 8)
        TFORMn = tf.strings.strip(TFORMn)
        
        len_TFORMn = tf.strings.length(TFORMn)
        num_TFORMn = 1
        if tf.math.greater(len_TFORMn, 1):
            num_TFORMn = int(tf.strings.substr(TFORMn, 0, len_TFORMn-1))
            TFORMn = tf.strings.substr(TFORMn, len_TFORMn-1, 1)
        bitpix_TFORMn = _TFORM_bitpix(TFORMn)
        
        TFORM = TFORM.write(field, [bitpix_TFORMn,num_TFORMn])
        
    #BITPIX - get bit size of values
    elif tf.math.equal(b'BITPIX  =', tf.strings.substr(string, 0, 9)):
        bitpix = int(tf.strings.substr(string, 9, 21))
        
    i += 1
    return fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, TFORM

#Loop through HDUs
@tf.function
def _hdu_bintable_condition(fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape, TFIELDS, TFORM):
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
       TFIELDS - number of columns in bintable
       TFORM - table column data type strings
       
       returns if h <= header'''
    return tf.math.less_equal(h, header)
    
@tf.function
def _hdu_bintable_body(fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape, TFIELDS, TFORM):
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
       shape - shape of data
       TFIELDS - number of columns in bintable
       TFORM - table column data type strings'''
    
    #All header keywords are 80 characters long
    i0 = tf.constant(0)
    i = 0
    NAXIS = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='NAXIS')
    string = tf.strings.substr(fits_data, offset, 80)
    
    read_loop = tf.while_loop(_read_bintable_header_condition, _read_bintable_header_body,
                              [fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, TFORM],
                              shape_invariants=[None,
                                                None,
                                                i0.get_shape(),
                                                i0.get_shape(),
                                                i0.get_shape(),
                                                i0.get_shape(),
                                                None,
                                                None],
                              name='read')
    _, NAXIS, TFIELDS, bitpix, offset, i, _, TFORM = read_loop

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
    
    return fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape, TFIELDS, TFORM