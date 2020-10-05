import tensorflow as tf

#loop over columns
@tf.function
def _asciitable_column_condition(r, NAXIS1, NAXIS2, TFIELDS, byte_data, start, fixed_length, TFORM, table_data):
    '''while loop condition when get data of column r
       r - colmn to get data
       NAXIS1 - number of bytes per row
       NAXIS2 - number of rows
       TFIELDS - noumber of columns
       byte_data - byte data of table being decoded
       start - start index of byte data
       fixed_length - length of byte_data
       TFORM - column butpix and number of type
       table_data = data of table
       
       returns if r < TFIELDS'''
    return tf.math.less(r, TFIELDS)

@tf.function
def _asciitable_column_body(r, NAXIS1, NAXIS2, TFIELDS, byte_data, start, fixed_length, TFORM, table_data):
    '''Function to get data of column r
       r - colmn to get data
       NAXIS1 - number of characters per row
       NAXIS2 - number of rows
       TFIELDS - number of columns
       byte_data - byte data of table being decoded
       start - start index of byte data
       fixed_length - length of byte_data
       TFORM - column butpix and number of type
       table_data = data of table'''
    
    #Get column data structure
    TFORMn = TFORM.read(r)
    j = tf.constant(0)
    bitpix = tf.slice(TFORMn, [0], [1])
    numtyp = tf.slice(TFORMn, [1], [1])
    
    #Get data through tf.strided_slice
    colm_data = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, name='colm_data')
    colm_start = start
    _, _, _, _, _, _, _, _, colm_data = tf.while_loop(_asciitable_column_data_condition, _asciitable_column_data_body,
                                                      [j, NAXIS1, TFIELDS, byte_data, colm_start, fixed_length, bitpix,
                                                      numtyp, colm_data],
                                                      shape_invariants=[j.get_shape(),
                                                                        j.get_shape(),
                                                                        tf.TensorShape(None),
                                                                        tf.TensorShape(None),
                                                                        colm_start.get_shape(),
                                                                        j.get_shape(),
                                                                        bitpix.get_shape(),
                                                                        numtyp.get_shape(),
                                                                        tf.TensorShape(None)],
                                                      name='column_data')
    colm_data = colm_data.stack()
    colm_data = tf.reshape(colm_data, (NAXIS2,))
    
    table_data = table_data.write(r, colm_data)
    
    start += numtyp
    r += 1
    return r, NAXIS1, NAXIS2, TFIELDS, byte_data, start, fixed_length, TFORM, table_data

@tf.function
def _asciitable_column_data_condition(j, NAXIS1, TFIELDS, byte_data, colm_start, fixed_length, bitpix, numtyp, colm_data):
    '''while loop condition when get data of cell j of column r
       j - cell in row r
       NAXIS1 - number of bytes per row
       TFIELDS - noumber of columns
       byte_data - byte data of table being decoded
       colm_start - start index of cell
       fixed_length - length of byte_data
       bitpix - number of bytes in data type
       colm_data - Tensor of column data
       
       returns if colm_start < fixed_length'''
    return tf.math.less(colm_start, fixed_length)

@tf.function
def _asciitable_column_data_body(j, NAXIS1, TFIELDS, byte_data, colm_start, fixed_length, bitpix, numtyp, colm_data):
    '''Function to get data of cell j of column r
       j - cell in row r
       NAXIS1 - number of bytes per row
       TFIELDS - noumber of columns
       byte_data - byte data of table being decoded
       colm_start - start index of cell
       fixed_length - length of byte_data
       bitpix - number of bytes in data type
       colm_data - Tensor of column data'''
    
    #Decode cell data
    #Done cell-by-cell or the DMA string copy error occurs
    cell_data = tf.strings.substr(byte_data, colm_start, numtyp)
    if tf.math.equal(bitpix, 8):
        rcvd_data = tf.io.decode_raw(cell_data, tf.uint8, False)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, 32):
        rcvd_data = tf.strings.to_number(cell_data, tf.int32)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    elif tf.math.equal(bitpix, -32):
        rcvd_data = tf.strings.to_number(cell_data, tf.float32)
    else:# tf.math.equal(bitpix, -64):
        rcvd_data = tf.strings.to_number(cell_data, tf.float32)
        rcvd_data = tf.cast(rcvd_data, tf.float32)
    
    colm_data = colm_data.write(j, tf.reshape(rcvd_data, (1,)))
    colm_start += NAXIS1
    j += 1
    
    return j, NAXIS1, TFIELDS, byte_data, colm_start, fixed_length, bitpix, numtyp, colm_data