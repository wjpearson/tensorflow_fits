import tensorflow as tf

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
def _read_TFORM_condition(fits_data, offset, i, string, TFIELDS, TFORM):
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
def _read_TFORM_body(fits_data, offset, i, string, TFIELDS, TFORM):
    '''Function to read next keyword in header
       and get necessary data from the keyword
       fits_data - byte string of data (from tf.io.read_file(file_path))
       offset - end of last header
       i - current keyword
       string - the keyword string
       TFIELDS - number of columns in bintable
       TFORM - table column data type strings'''
    
    #Get next keyword
    #All keyword lines are 80 characters long
    string = tf.strings.substr(fits_data, offset+(i*80), 80)
        
    #TFIELDS - get number of fields
    if tf.math.equal(b'TFIELDS =', tf.strings.substr(string, 0, 9)):
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
        
    i += 1
    return fits_data, offset, i, string, TFIELDS, TFORM