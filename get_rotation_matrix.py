import tensorflow as tf
from keras import backend

# print(os.getcwd())  # path/to/tensorflow_image_classification/
from deg2rad import get_rad


@tf.function
def get_rotation_matrix(angles, translations, scales, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).
    Args:
      angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`).
      translations: 
      scales:
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name: The name of the op.
    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
        given to operation `image_projective_transform_v2`. If one row of
        transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the
        *output* point `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`.
    """
    ZERO = tf.constant(0.0, dtype=tf.float32)
    ONE = tf.constant(1.0, dtype=tf.float32)
    with backend.name_scope(name or "3d_rotation_matrix"):
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(angles['theta'], angles['phi'], angles['gamma'])
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        distance = tf.math.sqrt(
            tf.math.pow(image_height, 2, name=None) + tf.math.pow(image_width, 2, name=None), 
            name=None
        )
        if tf.math.not_equal(tf.math.sin(rgamma), ZERO, name=None):
            factor = tf.math.multiply(tf.constant(2.0), tf.math.sin(rgamma), name=None)
        else:
            factor = ONE
        focal_length = tf.math.divide(distance, factor, name=None)
        dz = focal_length
        # Rotation matrices around the X, Y, and Z axis
        Rx = tf.stack(
            [tf.stack([ONE, ZERO, ZERO, ZERO], axis=0),
             tf.stack([ZERO, tf.math.cos(rtheta), -tf.math.sin(rtheta), ZERO], axis=0),
             tf.stack([ZERO, tf.math.sin(rtheta), tf.math.cos(rtheta), ZERO], axis=0),
             tf.stack([ZERO, ZERO, ZERO, ONE], axis=0)
            ], axis=0
        )
        Ry = tf.stack(
            [tf.stack([tf.math.cos(rphi), ZERO, -tf.math.sin(rphi), ZERO], axis=0),
             tf.stack([ZERO, ONE, ZERO, ZERO], axis=0),
             tf.stack([tf.math.sin(rphi), ZERO, tf.math.cos(rphi), ZERO], axis=0),
             tf.stack([ZERO, ZERO, ZERO, ONE], axis=0)
            ], axis=0
        )
        Rz = tf.stack(
            [tf.stack([tf.math.cos(rgamma), -tf.math.sin(rgamma), ZERO, ZERO], axis=0),
             tf.stack([tf.math.sin(rgamma), tf.math.cos(rgamma), ZERO, ZERO], axis=0),
             tf.stack([ZERO, ZERO, ONE, ZERO], axis=0),
             tf.stack([ZERO, ZERO, ZERO, ONE], axis=0)
            ], axis=0
        )
        # Composed rotation matrix with (RX, RY, RZ)
        R = tf.linalg.matmul(tf.linalg.matmul(Rx, Ry), Rz)
        # Translation matrix
        T = tf.stack(
            [tf.stack([ONE, ZERO, ZERO, translations['dx']], axis=0),
             tf.stack([ZERO, ONE, ZERO, translations['dy']], axis=0),
             tf.stack([ZERO, ZERO, ONE, dz], axis=0),
             tf.stack([ZERO, ZERO, ZERO, ONE], axis=0)
            ], axis=0
        )
        # Scale matrix
        S = tf.stack(
            [tf.stack([scales['sx'], ZERO, ZERO, ZERO], axis=0),
             tf.stack([ZERO, scales['sy'], ZERO, ZERO], axis=0),
             tf.stack([ZERO, ZERO, scales['sz'], ZERO], axis=0),
             tf.stack([ZERO, ZERO, ZERO, ONE], axis=0)
            ], axis=0
        )
        # Projection 2D -> 3D matrix
        P23 = tf.stack(
            [tf.stack([ONE, ZERO, -image_width/2], axis=0),
             tf.stack([ZERO, ONE, -image_height/2], axis=0),
             tf.stack([ZERO, ZERO, ONE], axis=0),
             tf.stack([ZERO, ZERO, ONE], axis=0)
            ], axis=0
        )
        # Projection 3D -> 2D matrix
        P32 = tf.stack(
            [tf.stack([focal_length, ZERO, image_width/2, ZERO], axis=0),
             tf.stack([ZERO, focal_length, image_height/2, ZERO], axis=0),
             tf.stack([ZERO, ZERO, ONE, ZERO], axis=0)
            ], axis=0
        )
        if scales:
            # not empty
            transform_matrix = tf.linalg.matmul(
                P32, tf.linalg.matmul(S, tf.linalg.matmul(T, tf.linalg.matmul(R, P23)))
            )
        else:
            transform_matrix = tf.linalg.matmul(
                P32, tf.linalg.matmul(T, tf.linalg.matmul(R, P23))
            )
        transform_matrix = tf.linalg.inv(transform_matrix, adjoint=False, name=None)
        f = transform_matrix[2][2]
        transform_matrix = transform_matrix / f
        transform_matrix = tf.reshape(transform_matrix, (1, 9))
        transform_matrix = transform_matrix[0][0:8]
        transform_matrix = tf.reshape(transform_matrix, (1, 8))
        return transform_matrix