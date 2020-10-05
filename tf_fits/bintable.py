import tensorflow as tf
from .common_functions.tf_fits_header import _hdu_condition, _hdu_body
from .table_functions.tf_fits_bintable_header import _read_TFORM_condition, _read_TFORM_body
from .table_functions.tf_fits_bintable_data import _bintable_column_condition, _bintable_column_body

@tf.function
def bintable_decode_fits(fits_data, header=1):
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
    table_data : tf.Tensor
                 tf.Tensor with dtype `tf.float32` of table data
    
    WARNING
    -------
    WILL MANGLE ANYTHING THAT IS NOT A PURE REAL NUMBER OR BOOL
    
    As Tensorflow wants a single data type returned, strings, complex numbers,
    arrays of numbers will be mangled."""
    
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
    
    i = tf.constant(0)
    TFIELDS = 0
    TFORM = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False, name='TFORM')
    string = tf.strings.substr(fits_data, offset, 80)
    
    tform_loop = tf.while_loop(_read_TFORM_condition, _read_TFORM_body,
                               [fits_data, offset, i, string, TFIELDS, TFORM],
                               shape_invariants=[None,
                                                 h.get_shape(),
                                                 h.get_shape(),
                                                 None,
                                                 h.get_shape(),
                                                 None])
    _, _, _, _, TFIELDS, TFORM = tform_loop
    
    #Get btye data for the chosen header
    byte_data = tf.strings.substr(fits_data, offset+start, fixed_length)
    
    r = tf.constant(0)
    start = tf.constant(0)
    NAXIS1 = shape.read(0)
    NAXIS2 = shape.read(1)
    table_data = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False, name='table_data')
    
    col_loop = tf.while_loop(_bintable_column_condition, _bintable_column_body,
                             [r, NAXIS1, NAXIS2, TFIELDS, byte_data, start, fixed_length, TFORM, table_data],
                             shape_invariants=[r.get_shape(),
                                               NAXIS1.get_shape(),
                                               NAXIS2.get_shape(),
                                               TFIELDS.get_shape(),
                                               tf.TensorShape(None),
                                               tf.TensorShape(None),
                                               tf.TensorShape(None),
                                               tf.TensorShape(None),
                                               tf.TensorShape(None)],
                             name='col_loop')
    _, _, _, _, _, _, _, _, table_data = col_loop
    
    table_data = table_data.stack()
    table_data = tf.transpose(table_data)
    
    return table_data
