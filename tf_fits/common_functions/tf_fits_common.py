import tensorflow as tf

#Get size and shape of data
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