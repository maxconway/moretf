import tensorflow as tf
import math
from haversine import haversine, Unit
import numpy as np

import logging
log = logging.getLogger(__name__)
# Driverline functions
log.info("driverline_functions loaded")

def __angle_interpolation_divided__(x, y, f, full_circle = 360.0):
    """Interpolate an angle
    x: start angle in degrees
    y: finish angle in degrees
    f: interpolation factor
    """
    assert(0 <= f <= 1)
    assert(0 <= x <= full_circle)
    assert(0 <= y <= full_circle)
    a = y - x
    a = (a + full_circle/2) % full_circle - full_circle/2
    return (x+a*f) % full_circle

def tableInterpolator(table, x):
    """Interpolate a value from a table
    table: a pandas dataframe with column 'Time'
    x: the time to interpolate
    """
    table = table.sort_values('Time')
    lastrow = table[table.Time<=x].iloc[-1,:]
    nextrow = table[table.Time>x].iloc[0,:]
    lastvalue = lastrow.iloc[1]
    nextvalue = nextrow.iloc[1]
    lasttime = lastrow.iloc[0]
    nexttime = nextrow.iloc[0]
    interpolated = lastvalue + (nextvalue - lastvalue)*(x-lasttime)/(nexttime-lasttime)
    return interpolated

def tableInterpolatorDegrees(table, x):
    """Interpolate an angle from a table in degrees
    table: a pandas dataframe with column 'Time'
    x: the time to interpolate
    """
    table = table.sort_values('Time')
    lastrow = table[table.Time<=x].iloc[-1,:]
    nextrow = table[table.Time>x].iloc[0,:]
    lastvalue = lastrow.iloc[1]
    nextvalue = nextrow.iloc[1]
    lasttime = lastrow.iloc[0]
    nexttime = nextrow.iloc[0]
    interpolated = __angle_interpolation_divided__(lastvalue, nextvalue, (x-lasttime)/(nexttime-lasttime))
    return interpolated

def normalise_homogenous(x):
    """Normalise a matrix in homogenous coordinates
    x: a matrix with shape (4,4)
    """
    tf.debugging.assert_type(x, tf.float32)
    y = tf.transpose(tf.transpose(x)/tf.transpose(x)[-1])
    tf.debugging.assert_shapes([
        (x, ('N', 'd')),
        (y, ('N', 'd'))
    ])
    tf.debugging.assert_equal(y[:,-1], 1.0)
    return y

def get_distances_and_angles(a_lat, a_lon, b_lat, b_lon):
    """Get distances and angles between two points
    a_lat: latitude of point A
    a_lon: longitude of point A
    b_lat: latitude of point B
    b_lon: longitude of point B
    """
    tf.debugging.assert_shapes([
        (a_lat, ()),
        (a_lon, ()),
        (b_lat, ()),
        (b_lon, ())
    ])
    if not -90 <= a_lat <= 90:
        log.error(f"a_lat looks wrong (value was {a_lat})")
    if not -90 <= b_lat <= 90:
        log.error(f"b_lat looks wrong (value was {b_lat})")
    if not -180 <= a_lon <= 180:
        log.error(f"a_lon looks wrong (value was {a_lon})")
    if not -180 <= b_lon <= 180:
        log.error(f"b_lon looks wrong (value was {b_lon})")
    crow_distance_ab = haversine((a_lat,a_lon), (b_lat, b_lon), unit=Unit.METERS)
    east_distance_ab = np.sign(b_lon-a_lon) * haversine(((a_lat+b_lat)/2,a_lon), ((a_lat+b_lat)/2, b_lon), unit=Unit.METERS)
    north_distance_ab = np.sign(b_lat-a_lat) * haversine((a_lat,(a_lon+b_lon)/2), (b_lat, (a_lon+b_lon)/2), unit=Unit.METERS)
    an = math.radians(a_lat)
    ae = math.radians(a_lon)
    bn = math.radians(b_lat)
    be = math.radians(b_lon)
    
    displacement_angle = math.atan2(
        math.sin(be-ae)*math.cos(bn), 
        math.cos(an)*math.sin(bn)-math.sin(an)*math.cos(bn)*math.cos(be-ae)
    )

    if not -0 <= crow_distance_ab <= 10000:
        log.error(f"crow_distance_ab looks wrong (value was {crow_distance_ab})")
    if not -10000 <= east_distance_ab <= 10000:
        log.error(f"east_distance_ab looks wrong (value was {east_distance_ab})")
    if not -10000 <= north_distance_ab <= 10000:
        log.error(f"north_distance_ab looks wrong (value was {north_distance_ab})")
    tf.debugging.assert_near(east_distance_ab**2 + north_distance_ab**2, crow_distance_ab**2, rtol=0.01)
    if not -math.pi <= displacement_angle <= math.pi:
        log.error(f"displacement_angle looks wrong (value was {displacement_angle})")
    
    if displacement_angle == 0:
        assert(east_distance_ab==0)
        assert(north_distance_ab>=0)
    if 0< displacement_angle < math.pi/2:
        assert(east_distance_ab>=0)
        assert(north_distance_ab>=0)
    if displacement_angle == math.pi/2:
        assert(east_distance_ab>=0)
        assert(north_distance_ab==0)
    if math.pi/2 < displacement_angle < math.pi:
        assert(east_distance_ab>=0)
        assert(north_distance_ab<=0)
    if displacement_angle == math.pi:
        assert(east_distance_ab==0)
        assert(north_distance_ab<=0)
    if displacement_angle == -math.pi:
        assert(east_distance_ab==0)
        assert(north_distance_ab<=0)
    if -math.pi < displacement_angle < -math.pi/2:
        assert(east_distance_ab<=0)
        assert(north_distance_ab<=0)
    if displacement_angle == -math.pi/2:
        assert(east_distance_ab<=0)
        assert(north_distance_ab==0)
    if -math.pi/2 < displacement_angle < 0:
        assert(east_distance_ab<=0)
        assert(north_distance_ab>=0)
    
    return crow_distance_ab, east_distance_ab, north_distance_ab, displacement_angle

def latlon_transform_m_2d(
    o_lat, 
    o_lon, 
    o_head, 
    n_lat, 
    n_lon, 
    n_head
    ):
    """Transform a point from coordinates centered on one point and angle to another
    o_lat: latitude of origin
    o_lon: longitude of origin
    o_head: heading of origin
    n_lat: latitude of new origin
    n_lon: longitude of new origin
    n_head: heading of new origin
    """
    tf.debugging.assert_shapes([
        (o_lat, ()),
        (o_lon, ()),
        (o_head, ()),
        (n_lat, ()),
        (n_lon, ()),
        (n_head, ())
    ])
    if(o_head>2*math.pi or o_head<-2*math.pi):
        log.error(f"o_head doesn't look like radians (value was {o_head})")
    if(n_head>2*math.pi or n_head<-2*math.pi):
        log.error(f"n_head doesn't look like radians (value was {o_head})")
    
    crow_distance_ab, east_distance_ab, north_distance_ab, displacement_angle = get_distances_and_angles(n_lat, n_lon, o_lat, o_lon)
    
    assert(math.isclose(east_distance_ab**2+north_distance_ab**2,crow_distance_ab**2,rel_tol=0.01, abs_tol=0.1))
    
    log.info(f"latlon transform o_head: {math.degrees(o_head)}")
    log.info(f"latlon transform n_head: {math.degrees(n_head)}")
    log.info(f"latlon transform displacement angle: {math.degrees(displacement_angle)}")
    
    rotate_to_north_1 = tf.constant([
        [math.cos(-o_head),  math.sin(-o_head), 0.],
        [-math.sin(-o_head), math.cos(-o_head), 0.],
        [0.,                0.,               1.]
    ])

    rotate_to_displacement_direction = tf.constant([
        [math.cos(displacement_angle),  math.sin(displacement_angle), 0.],
        [-math.sin(displacement_angle), math.cos(displacement_angle), 0.],
        [0.,                0.,               1.]
    ])
    
    translate = tf.constant([
        [1., 0., crow_distance_ab],
        [0., 1., 0.],
        [0., 0., 1.]
    ])

    rotate_to_north_2 = tf.constant([
        [math.cos(-displacement_angle),  math.sin(-displacement_angle), 0.],
        [-math.sin(-displacement_angle), math.cos(-displacement_angle), 0.],
        [0.,                0.,               1.]
    ])
    
    rotate_to_new_heading = tf.constant([
        [math.cos(n_head),  math.sin(n_head), 0.],
        [-math.sin(n_head), math.cos(n_head), 0.],
        [0.,                0.,               1.]
    ])
    
    return rotate_to_new_heading @ rotate_to_north_2 @ translate @ rotate_to_displacement_direction @ rotate_to_north_1

def coordinates_2_tensor(
    coordinates,
    values,
    shape,
    resolution,
    offsets = 'center',
    base_value = 0,
    combine = 'add'
    ):
    """Convert a long form set of coordinates and values of points to a wide form tensor
    coordinates: list of coordinates
    values: list of values
    shape: shape of the tensor
    resolution: resolution of the tensor
    offsets: where the coordinates are relative to (must be 'center')
    base_value: value to use for the fill of the tensor
    combine: how to combine multiple coordinates into a single value (must be 'add')"""
    tf.debugging.assert_shapes([
        (coordinates, ('N', 'D+1')),
        (values, ('N')),
        (shape, ('D')),
        (base_value, ())
    ])
    tf.debugging.assert_all_finite(coordinates, message="Non finite coordinate")
    tf.debugging.assert_equal(len(shape)+1, coordinates.shape[1])
    tf.debugging.assert_equal(tf.rank(coordinates),2)
    tf.debugging.assert_equal(offsets,'center')
    tf.debugging.assert_equal(combine,'add')

    if offsets == 'center':
        offsets = tf.cast(shape, tf.float32)/2.0
    # normalise coordinates and drop homogenous dimension
    indices = tf.cast(tf.math.round(offsets+normalise_homogenous(coordinates)[:,0:-1]/resolution), tf.int32)
    
    mask = tf.logical_and(
        tf.reduce_all(indices>=0, axis=1),
        tf.reduce_all(indices < shape, axis=1)
    )
    tf.debugging.assert_shapes([
        (coordinates, ('N', 'D+1')),
        (values, ('N')),
        (shape, ('D')),
        (indices, ('N', 'D')),
        (mask, ('N'))
    ])

    indices = tf.boolean_mask(indices, mask)
    values = tf.boolean_mask(values, mask)

    result = tf.fill(shape, base_value)

    tf.debugging.assert_shapes([
        (values, ('M')),
        (shape, ('D')),
        (indices, ('M', 'D')),
        (result, ('H', 'W'))
    ])

    result = tf.tensor_scatter_nd_add(
        result,
        indices = indices,
        updates = values
    )

    tf.debugging.assert_shapes([
        (values, ('M')),
        (shape, ('D')),
        (indices, ('M', 'D')),
        (result, ('H', 'W'))
    ])

    # if tf.reduce_all(tf.math.is_finite(tf.cast(values,tf.float32))):
    #     tf.debugging.assert_near(tf.reduce_sum(values),tf.reduce_sum(result), rtol=0.01)

    return result

def local_2_global(
    local_coords,
    local_origin_lat, 
    local_origin_lon, 
    local_origin_head, 
    global_origin_lat, 
    global_origin_lon, 
    global_origin_head
    ):
    """Convert local coordinates to global coordinates
    local_coords: list of local coordinates
    local_origin_lat: latitude of the local origin
    local_origin_lon: longitude of the local origin
    local_origin_head: heading of the local origin
    global_origin_lat: latitude of the global origin
    global_origin_lon: longitude of the global origin
    global_origin_head: heading of the global origin
    """
    # calculate local to world transformation
    loc_2_glob = latlon_transform_m_2d(
        o_lat = local_origin_lat, 
        o_lon = local_origin_lon, 
        o_head = local_origin_head, 
        n_lat = global_origin_lat, 
        n_lon = global_origin_lon, 
        n_head = global_origin_head
    )

    # Apply local to global transformation
    global_coords = normalise_homogenous(local_coords @ tf.transpose(loc_2_glob))

    # Do some checks
    tf.debugging.assert_shapes([
        (local_coords, ('N',3)),
        (loc_2_glob, (3,3)),
        (global_coords, ('N', 3))
    ])
    center_local = tf.reduce_mean(local_coords[:,0:2], 0)
    center_global = tf.reduce_mean(global_coords[:,0:2], 0)
    var_local = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(local_coords[:,0:2]-center_local),1)))
    var_global = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(global_coords[:,0:2]-center_global),1)))
    # tf.debugging.assert_near(var_local, var_global, rtol=0.05)
    return global_coords

# @tf.function(experimental_relax_shapes=False)
def tensor_2_coordinates(
    tensor,
    resolution,
    offsets = 'center',
    base = 0.0
    ):
    tf.debugging.assert_equal(base, 0.0)
    offsets = tf.cast(tf.shape(tensor)-1,tf.float32)/2.0
    indices = tf.where(tensor)
    values = tf.gather_nd(tensor, indices)
    coordinates = tf.cast(indices, tf.float32)-offsets
    coordinates = tf.concat([coordinates*resolution, tf.ones([tf.shape(coordinates)[0],1])], axis=1)
    tf.debugging.assert_equal(tf.rank(tensor)+1,tf.shape(coordinates)[1])
    tf.debugging.assert_less_equal(tf.shape(coordinates)[0],tf.size(tensor))
    tf.debugging.assert_shapes([
        (coordinates, ('N', 'dims+1')),
        (values, ('N')),
        (indices, ('N', 'dims')),
    ])
    return coordinates, values