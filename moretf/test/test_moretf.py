import moretf.moretf as mtf
import tensorflow as tf
import unittest

class TestBlockDiagonal(unittest.TestCase):
    def test_block_diagonal(self):
        inputs = [tf.constant([[1,2],[3,4]]),
                tf.constant([[5,6,7],[8,9,10]])
        ]
        output = mtf.block_diagonal(inputs)
        self.assertEqual(output.shape, (4,5))
        self.assertTrue(tf.reduce_all(output == tf.constant([
            [1,2,0,0,0],
            [3,4,0,0,0],
            [0,0,5,6,7],
            [0,0,8,9,10]
        ])))

class TestWindow(unittest.TestCase):
    def test_window(self):
        input = tf.expand_dims(tf.constant([[1,2,3],[4,5,6],[7,8,9]]),0)
        windows = [[0],[0],[0]]
        output = mtf.window(input, windows)
        self.assertEqual(output.shape, (1,3,3))
        self.assertTrue(tf.reduce_all(tf.squeeze(output) == tf.constant([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ])))
    def test_window(self):
        input = tf.expand_dims(tf.constant([[1,2,3],[4,5,6],[7,8,9]]),0)
        windows = [[0,1],[0],[0]]
        output = mtf.window(input, windows)
        self.assertEqual(output.shape, (1,3,4))
        self.assertTrue(tf.reduce_all(tf.squeeze(output) == tf.constant([
            [-1,1,2,3],
            [1,4,5,6],
            [4,7,8,9]
        ])))
# mtf.block_diagonal(inputs)

# input = tf.constant([[
#                         [1,10],
#                         [2,20],
#                         [3,30],
#                         [4,40],
#                         [5,50],
#                         [6,60]
# ]])

# windows = [[0,1,2],[0,1]]

# mtf.window(input, windows, -1)

def run_tests():
    unittest.main()