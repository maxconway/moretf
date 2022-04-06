import moretf.coordinate_transformation as ct
import unittest
import tensorflow as tf
import math
from haversine import haversine, Unit

class TestCoordinateTransformation(unittest.TestCase):
    def test_all(self):
        # Test ct.local_2_global
        tf.debugging.assert_less(
            ct.local_2_global(
                local_coords = tf.constant([[0.0,0.0,1.0]]),
                local_origin_lat = 0.0, 
                local_origin_lon = 0.0, 
                local_origin_head = 0.0, 
                global_origin_lat = 0.001, 
                global_origin_lon = 0.0, 
                global_origin_head = 0.0
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = -0.0001, 
            global_origin_head = 0.0
            )[0,1],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0001, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,1],
            0.0
        )

        tf.debugging.assert_less(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = -0.0001, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,1],
            0.0
        )

        tf.debugging.assert_less(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,-10.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,1],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.001, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.001, 
            local_origin_lon = 0.0, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,0],
            0.0
        )

        tf.debugging.assert_near(
            ct.local_2_global(
            local_coords = tf.constant([[100.0,-90.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi/2, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi/2
            ),
            tf.constant([[100.0,-90.0,1.0]])
        )

        tf.debugging.assert_near(
            ct.local_2_global(
            local_coords = tf.constant([[100.0,-90.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = -math.pi
            ),
            tf.constant([[100.0,-90.0,1.0]])
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[100.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi/2, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,1],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,-100.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi/2, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[-100.0,00.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = -math.pi/2, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[100.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.001, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi/2, 
            global_origin_lat = 0.0000, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,0],
            0.0
        )

        tf.debugging.assert_less(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = -0.001, 
            local_origin_head = -math.pi/2, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = 0.0
            )[0,1],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.001, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = -0.001, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = -math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = -0.001, 
            local_origin_lon = 0.0, 
            local_origin_head = 0.0, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = -0.001, 
            local_origin_lon = 0.0, 
            local_origin_head = -math.pi/2, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,0.0,1.0]]),
            local_origin_lat = -0.001, 
            local_origin_lon = 0.0, 
            local_origin_head = -math.pi/2, 
            global_origin_lat = -0.00001, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,-100.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,100.0,1.0]]),
            local_origin_lat = 0.0, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi, 
            global_origin_lat = 0.0, 
            global_origin_lon = 0.0, 
            global_origin_head = -math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_greater(
            ct.local_2_global(
            local_coords = tf.constant([[0.0,100.0,1.0]]),
            local_origin_lat = 0.001, 
            local_origin_lon = 0.0, 
            local_origin_head = math.pi, 
            global_origin_lat = -0.0001, 
            global_origin_lon = 0.0, 
            global_origin_head = -math.pi/2
            )[0,0],
            0.0
        )

        tf.debugging.assert_less(
            ct.local_2_global(
            local_coords = tf.constant([[-100.0,-100.0,1.0]]),
            local_origin_lat = -30, 
            local_origin_lon = -90, 
            local_origin_head = -math.pi/2, 
            global_origin_lat = -30, 
            global_origin_lon = -90, 
            global_origin_head = -math.pi/2
            )[0,0:2],
            0.0
        )

        assert(10<=ct.__angle_interpolation_divided__(10,20, 0.5, full_circle = 360.0)<=20)
        assert(10==ct.__angle_interpolation_divided__(10,20, 0, full_circle = 360.0))
        assert(20==ct.__angle_interpolation_divided__(10,20, 1, full_circle = 360.0))
        assert(0.0==ct.__angle_interpolation_divided__(355,5, 0.5, full_circle = 360.0))
        assert(355==ct.__angle_interpolation_divided__(355,5, 0, full_circle = 360.0))
        assert(5==ct.__angle_interpolation_divided__(355,5, 1, full_circle = 360.0))
        assert(0.0==ct.__angle_interpolation_divided__(5,355, 0.5, full_circle = 360.0))
        assert(355==ct.__angle_interpolation_divided__(5,355, 1, full_circle = 360.0))
        assert(5==ct.__angle_interpolation_divided__(5,355, 0, full_circle = 360.0))

