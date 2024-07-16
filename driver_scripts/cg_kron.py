import tensorflow as tf
import numpy as np


def main():
  
  # Intialize a numpy array c containing two arrays
  a = np.array([1, 0])
  b = np.array([2, 2])
  c = tf.constant ([a, b])
  print (f"Shape: {tf.shape(c)}", end="\n") # Should be [2, 2]

  td_result = tf.math.reduce_prod (c, axis=0) # Should be [2, 2]
  print (f"td_result: {td_result}")
  
  # td_result = tf.tensordot(a, b, axes=0), [-1]
  print (f"Tensor Result")
  
  # result = tf.reshape(td_result, [-1])
  
  print (f"a: {a}")
  print (f"b: {b}")
  # print(result)
  
  
  

main()