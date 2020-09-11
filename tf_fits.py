import tensorflow as tf

#Loop over TFROM data types
@tf.function
def _TFORM_type_condition(j, k, field, TFORM, numbyt, numtyp, substr):
    '''while loop condition when get TFORM data type and number of bytes
       j - position in TFORM
       field - bintable column number
       TFIELDS - number of columns in bintable
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column
       substr - current substr of TFORM being compared
       
       returns if k < 13 (number of bintable data types'''
    return tf.math.less(k, 13)

@tf.function
def _TFORM_type_body(j, k, field, TFORM, numbyt, numtyp, substr):
    '''Function to get TFORM data type and number of bytes
       j - position in TFORM
       field - bintable column number
       TFIELDS - number of columns in bintable
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column
       substr - current substr of TFORM being compared'''
    
    if tf.math.equal(k, 0):
        typ = b'L'
        datbit = 8
    elif tf.math.equal(k, 1):
        typ = b'X'
        datbit = 0
    elif tf.math.equal(k, 2):
        typ = b'B'
        datbit = 8
    elif tf.math.equal(k, 3):
        typ = b'I'
        datbit = 16
    elif tf.math.equal(k, 4):
        typ = b'J'
        datbit = 32
    elif tf.math.equal(k, 5):
        typ = b'K'
        datbit = 64
    elif tf.math.equal(k, 6):
        typ = b'A'
        datbit = 8
    elif tf.math.equal(k, 7):
        typ = b'E'
        datbit = -32
    elif tf.math.equal(k, 8):
        typ = b'D'
        datbit = -64
    elif tf.math.equal(k, 9):
        typ = b'C'
        datbit = -64
    elif tf.math.equal(k, 10):
        typ = b'M'
        datbit = -128
    elif tf.math.equal(k, 11):
        typ = b'P'
        datbit = 32
    else:# tf.math.equal(k, 12):
        typ = b'Q'
        datbit = 64
    
    if tf.strings.regex_full_match(substr, typ):
        numtyp = numtyp.write(field-1, datbit)
        if tf.math.equal(j, 0):
            numbyt = numbyt.write(field-1, 1)
        else:
            numbyt = numbyt.write(field-1, int(tf.strings.substr(TFORM, 0, j)))
            
        k = 13
        j = 8
    k += 1
    return j, k, field, TFORM, numbyt, numtyp, substr

#Loop over TFORM data type string positions
@tf.function
def _get_TFORM_condition(j, field, TFORM, numbyt, numtyp):
    '''while loop condition when compare char in TFORM to data type
       j - position in TFORM
       field - bintable column number
       TFIELDS - number of columns in bintable
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column
       
       returns if j < 8 (number of char in TFORM)'''
    return tf.math.less(j, 8)

@tf.function
def _get_TFORM_body(j, field, TFORM, numbyt, numtyp):
    '''Function to compare char in TFORM to data type
       j - position in TFORM
       field - bintable column number
       TFIELDS - number of columns in bintable
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column'''
    
    substr = tf.strings.substr(TFORM, j, 1)
    
    k = tf.constant(0)
    j, k, field, TFORM, numbyt, numtyp, substr = tf.while_loop(_TFORM_type_condition, _TFORM_type_body,
                                                               [j, k, field, TFORM, numbyt, numtyp, substr],
                                                               shape_invariants=[j.get_shape(),
                                                                                 k.get_shape(),
                                                                                 k.get_shape(),
                                                                                 None,
                                                                                 None,
                                                                                 None,
                                                                                 None])
                                      
    j += 1
    
    return j, field, TFORM, numbyt, numtyp

#Loop through Header Keywords
@tf.function
def _read_header_condition(fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, numbyt, numtyp):
    '''while loop condition when read next keyword in header
       and get necessary data from the keyword
       fits_data - byte string of data (from tf.io.read_file(file_path))
       NAXIS - all axes sizes
       TFIELDS - number of columns in bintable
       bitpix - fits data type
       offset - end of last header
       i - current keyword
       string - the keyword string
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column
       
       returns if keyword is not the final keyword'''
    return tf.math.not_equal(string, b'END                                                                             ')

@tf.function
def _read_header_body(fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, numbyt, numtyp):
    '''Function to read next keyword in header
       and get necessary data from the keyword
       fits_data - byte string of data (from tf.io.read_file(file_path))
       NAXIS - all axes sizes
       TFIELDS - number of columns in bintable
       bitpix - fits data type
       offset - end of last header
       i - current keyword
       string - the keyword string
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column'''
       
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
        
    #TFORM# - get field * data format
    elif tf.math.equal(b'TFORM', tf.strings.substr(string, 0, 5)):
        field = int(tf.strings.substr(string, 5, 3))
        TFORM = tf.strings.substr(string, 11, 8)
        
        j = tf.constant(0)
        _, _, _, numbyt, numtyp = tf.while_loop(_get_TFORM_condition, _get_TFORM_body,
                                                [j, field, TFORM, numbyt, numtyp],
                                                shape_invariants=[j.get_shape(),
                                                                  j.get_shape(),
                                                                  None,
                                                                  None,
                                                                  None])
        
        
    #BITPIX - get bit size of values
    elif tf.math.equal(b'BITPIX  =', tf.strings.substr(string, 0, 9)):
        bitpix = int(tf.strings.substr(string, 9, 21))
    i += 1
    
    return fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, numbyt, numtyp

#Get size and shape of image
@tf.function
def _size_and_shape_condition(axis, NAXIS, size, shape):
    '''while loop condition when get size and shape from NAXIS
       axis - current axis being used
       NAXIS - all axes sizes
       size - liner size of data
       shape - shape of final data
       
       returns if axis <= length of axes'''
    return tf.math.less_equal(axis, NAXIS.read(0))

@tf.function
def _size_and_shape_body(axis, NAXIS, size, shape):
    '''Function to get size and shape from NAXIS
       axis - current axis being used
       NAXIS - all axes sizes
       size - liner size of data
       shape - shape of final data'''
    
    NAXISn = NAXIS.read(axis)
    size *= NAXISn
    shape = shape.write(axis-1, NAXISn)
    axis += 1
    return axis, NAXIS, size, shape

#Loop through HDUs
@tf.function
def _hdu_condition(fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape, TFIELDS, numbyt, numtyp):
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
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column
       
       returns if h <= header'''
    return tf.math.less_equal(h, header)
    
@tf.function
def _hdu_body(fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape, TFIELDS, numbyt, numtyp):
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
       numbyt - number of bytes for data type of bintable column
       numtyp - data type for bintable column'''
    
    #All header keywords are 80 characters long
    i = tf.constant(0)
    NAXIS = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    string = tf.strings.substr(fits_data, offset, 80)
    
    read_loop = tf.while_loop(_read_header_condition, _read_header_body,
                              [fits_data, NAXIS, TFIELDS, bitpix, offset, i, string, numbyt, numtyp],
                              shape_invariants=[None,
                                                None,
                                                i.get_shape(),
                                                i.get_shape(),
                                                i.get_shape(),
                                                i.get_shape(),
                                                None,
                                                None,
                                                None],
                              name='read')
    _, NAXIS, TFIELDS, bitpix, offset, i, _, numbyt, numtyp = read_loop

    i -= 1
    start = i*80
        
    #All data blocks are 2880 bytes long,
    #so find where the data starts including padding
    #For header, padding is spaces: b' '
    tru_start = int(start // 2880)
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
    
    return fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape, TFIELDS, numbyt, numtyp

#Loop over columns
@tf.function
def _bintable_column_condition(j, byte_data, row_data, numcol, numbyt, numtyp, start):
    '''while loop condition when read bintable columns
       j - current column
       byte_data - byte string of table data
       row_data - current row's data
       numcol - the total number of columns
       numbyt - the number of bytes in each column
       numtyp - the number of bits in each column's data type
       start - the start position in byte_data of the column
       
       returns if j < numcol'''
    return tf.math.less(j, numcol)

@tf.function
def _bintable_column_body(j, byte_data, row_data, numcol, numbyt, numtyp, start):
    '''Function to read bintable columns
       j - current column
       byte_data - byte string of table data
       row_data - current row's data
       numcol - the total number of columns
       numbyt - the number of bytes in each column
       numtyp - the number of bits in each column's data type
       start - the start position in byte_data of the column'''
    
    bitpix = numtyp.read(j)
    size = (numbyt.read(j)*abs(bitpix))//8
    
    colm_data = tf.strings.substr(byte_data, start, size)
    
    if tf.math.equal(bitpix, 8):
        rcvd_data = tf.io.decode_raw(colm_data, tf.uint8, False)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 16):
        rcvd_data = tf.io.decode_raw(colm_data, tf.int16, False)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 32):
        rcvd_data = tf.io.decode_raw(colm_data, tf.int32, False)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 64):
        rcvd_data = tf.io.decode_raw(colm_data, tf.int64, False)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, -32):
        rcvd_data = tf.io.decode_raw(colm_data, tf.float32, False)
    else:# tf.math.equal(bitpix, -64):
        rcvd_data = tf.io.decode_raw(colm_data, tf.float64, False)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
        
    #rcvd_data = tf.keras.backend.print_tensor(rcvd_data)
    row_data = row_data.write(j, tf.slice(rcvd_data, [0], [1]))
    start += size
    
    j += 1
    
    return j, byte_data, row_data, numcol, numbyt, numtyp, start

#Loop over bintable rows
@tf.function
def _bintable_row_condition(i, byte_data, table_data, numrow, numcol, numbyt, numtyp, start):
    '''while loop condition when read bintable rows
       i - current row
       byte_data - byte string of table data
       table_data - table's data
       numrow - the total number of rows
       numcol - the total number of columns
       numbyt - the number of bytes in each column
       numtyp - the number of bits in each column's data type
       start - the start position in byte_data of the row
       
       returns if i < numow'''
    return tf.math.less(i, numrow)

@tf.function
def _bintable_row_body(i, byte_data, table_data, numrow, numcol, numbyt, numtyp, start):
    '''Function to read bintable rows
       i - current row
       byte_data - byte string of table data
       table_data - table's data
       numrow - the total number of rows
       numcol - the total number of columns
       numbyt - the number of bytes in each column
       numtyp - the number of bits in each column's data type
       start - the start position in byte_data of the row'''
    
    row_data = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False, name='row_data')
    j = tf.constant(0)
    _, _, row_data, _, _, _, start = tf.while_loop(_bintable_column_condition, _bintable_column_body,
                                                   [j, byte_data, row_data, numcol, numbyt, numtyp, start],
                                                   shape_invariants=[j.get_shape(),
                                                                     None,
                                                                     None,
                                                                     j.get_shape(),
                                                                     j.get_shape(),
                                                                     j.get_shape(),
                                                                     j.get_shape()])
    table_data = table_data.write(i, row_data.stack())
    
    i += 1
    
    return i, byte_data, table_data, numrow, numcol, numbyt, numtyp, start

@tf.function
def image_decode_fits(fits_data, header):
    '''Function to decode fits images
       fits_data - byte string of data (from tf.io.read_file(file_path))
       header - header to return
       
       returns tf.Tensor with dtype tf.float32 of image data'''
    
    h = tf.constant(0)
    offset = 0 #Position of start of HDU
    start = 0
    true_length = 0
    fixed_length = 0
    bitpix = 0
    shape = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='shape')
    TFIELDS = 0
    numbyt = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='numbyt')
    numtyp = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='numtyp')
    
    hdu_loop = tf.while_loop(_hdu_condition, _hdu_body,
                             [fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape,
                              TFIELDS, numbyt, numtyp],
                              shape_invariants=[None,
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                tf.TensorShape(None),
                                                h.get_shape(),
                                                tf.TensorShape(None),
                                                tf.TensorShape(None)])
    _, _, _, offset, start, true_length, fixed_length, bitpix, shape, _, _, _ = hdu_loop
    
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
        
    #Reshape the data
    shape = shape.stack()
    img_data = tf.reshape(rcvd_data, shape)
    
    return img_data

@tf.function
def bintable_decode_fits(fits_data, header=1):
    '''Function to decode fits bintable
       fits_data - byte string of data (from tf.io.read_file(file_path))
       header - header to return
       
       returns tf.Tensor with dtype tf.float32 of table data'''
    
    """WARNING - WILL MANGLE ANYTHING THAT IS NOT A PURE REAL NUMBER OR BOOL"""
    
    '''As Tensorflow wants a single data type returned, strings, complex numbers,
       arrays of numbers will be mangled.'''
    
    h = tf.constant(0)
    offset = 0 #Position of start of HDU
    start = 0
    true_length = 0
    fixed_length = 0
    bitpix = 0
    shape = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='shape')
    TFIELDS = 0
    numbyt = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='numbyt')
    numtyp = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='numtyp')
    
    hdu_loop = tf.while_loop(_hdu_condition, _hdu_body,
                             [fits_data, h, header, offset, start, true_length, fixed_length, bitpix, shape,
                              TFIELDS, numbyt, numtyp],
                              shape_invariants=[None,
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                h.get_shape(),
                                                tf.TensorShape(None),
                                                h.get_shape(),
                                                tf.TensorShape(None),
                                                tf.TensorShape(None)])
    _, _, _, offset, start, true_length, fixed_length, _, shape, TFIELDS, numbyt, numtyp = hdu_loop
    
    #Get btye data for the chosen header
    byte_data = tf.strings.substr(fits_data, offset+start, true_length)
    
    numrow = shape.read(1)
    i = tf.constant(0)
    start = tf.constant(0)
    table_data = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False, name='table_data')
    
    _, _, table_data, _, _, _, _, start = tf.while_loop(_bintable_row_condition, _bintable_row_body,
                                                        [i, byte_data, table_data, numrow, TFIELDS, numbyt, numtyp, start],
                                                        shape_invariants=[i.get_shape(),
                                                                          tf.TensorShape(None),
                                                                          tf.TensorShape(None),
                                                                          i.get_shape(),
                                                                          i.get_shape(),
                                                                          i.get_shape(),
                                                                          i.get_shape(),
                                                                          i.get_shape()])
    
    table_data = table_data.stack()
    if tf.math.equal(numrow, 1):
        table_data = tf.reshape(table_data, (TFIELDS,))
    else:
        table_data = tf.reshape(table_data, (numrow, TFIELDS))
    
    return table_data
