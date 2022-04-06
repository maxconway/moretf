import tensorflow as tf

# Create a block diagonal matrix from a set of input matricies. The axes mask controls which dimensions diagonalisation occurs in
def block_diagonal(inputs, axes_mask = 1):
  sizes = tf.constant([x.shape for x in inputs])
  baseshape = tf.reduce_sum(sizes,0)
  pre_pad = tf.zeros(baseshape.shape, dtype=tf.int32)
  paddeds = []
  for x in inputs:
    post_pad = (baseshape-pre_pad-x.shape)*axes_mask
    paddeds.append(tf.pad(x, tf.stack([pre_pad, post_pad], axis=-1)))
    pre_pad = (pre_pad + x.shape)*axes_mask
  return tf.math.reduce_sum(paddeds, axis=0)

# Create windows on a dataset, using different spacings for each layer
# windows is a list of lists. The outer list is has an entry for each input column, the inner one is the list of offsets that are used
# @tf.function
def window(input, windows, padvalue = -1):
  tf.debugging.assert_rank(input, 3)
  maxwindow = max(max(x) for x in windows)
  input_padded =tf.pad(input, [[0,0],[maxwindow,0],[0,0]], constant_values=padvalue)
  kernel = tf.cast(block_diagonal([tf.expand_dims(tf.transpose(tf.reverse(tf.one_hot(x, maxwindow+1),[0,1])),1) for x in windows], [0,1,1]), dtype=input.dtype)
  return tf.nn.conv1d(
      input_padded,
      kernel,
      stride=1,
      padding='VALID'
  )
