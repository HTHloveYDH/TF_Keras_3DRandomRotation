# import numpy as np
import tensorflow.compat.v2 as tf
# from keras import backend
# from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
# from keras.utils import image_utils
# from keras.utils import tf_utils
# isort: off
# from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import keras_export
# from tensorflow.tools.docs import doc_controls
from keras.layers.preprocessing.image_preprocessing import BaseImageAugmentationLayer, transform, \
                                                           check_fill_mode_and_interpolation, H_AXIS, \
                                                           W_AXIS  # , IMAGES, LABELS, TARGETS, BOUNDING_BOXES

from modules.custom.layers.preprocessing.get_rotation_matrix import get_rotation_matrix


@keras_export("keras.layers.Random3DRotation", v1=[],)
class Random3DRotation(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly rotates images during training.
    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.
    By default, random rotations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    rotations at inference time, set `training` to True when calling the layer.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype. By default, the layer will output
    floats.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, widtimage_tensorhtheta, channels)`, in `"channels_last"` format
    Arguments:
      factor:
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
        - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
          reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
          filling all values beyond the edge with the same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
          wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
          the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
    """

    def __init__(
        self,
        factor,
        fill_mode="constant",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs,
    ):
        base_preprocessing_layer.keras_kpl_gauge.get_cell('Random3DRotation').set(
            True
        )
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, dict):
            # for angle: theta, phi, gamma
            self.theta_range = factor['angle']['theta_range']
            self.phi_range = factor['angle']['phi_range']
            self.gamma_range = factor['angle']['gamma_range']
        elif isinstance(factor, (list, tuple)):
            self.theta_range = self.phi_range = self.gamma_range = factor
        else:
            self.theta_range = self.phi_range = self.gamma_range = (-factor, factor)
        if self.theta_range[0] > self.theta_range[1] or \
           self.phi_range[0] > self.phi_range[1] or \
           self.gamma_range[0] > self.gamma_range[1]:
            raise ValueError(
                f"Factor cannot have negative values, got {factor}"
        )
        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None
    ):
        # define necessary constants
        ZERO = tf.constant(0.0, dtype=tf.float32)
        ONE = tf.constant(1.0, dtype=tf.float32)
        # for theta angle, x axis
        theta = self._random_generator.random_uniform(
            shape=[], minval=self.theta_range[0], maxval=self.theta_range[1]
        )
        # for phi angle, y axis
        phi = self._random_generator.random_uniform(
            shape=[], minval=self.phi_range[0], maxval=self.phi_range[1]
        )
        # for gamma angle, z axis
        gamma = self._random_generator.random_uniform(
            shape=[], minval=self.gamma_range[0], maxval=self.gamma_range[1]
        )
        # for dx, dy
        if self.factor.get('translation', None) is not None:
            self.dx_range = self.factor['translation']['dx_range']
            self.dy_range = self.factor['translation']['dy_range']
            dx = self._random_generator.random_uniform(
                shape=[], minval=self.dx_range[0], maxval=self.dx_range[1]
            )
            dy = self._random_generator.random_uniform(
                shape=[], minval=self.dy_range[0], maxval=self.dy_range[1]
            )
        else:
            dx = dy = ZERO
        # for sx, sy, sz
        if self.factor.get('scale', None) is not None:
            self.sx_range = self.factor['scale']['sx_range']
            self.sy_range = self.factor['scale']['sy_range']
            self.sz_range = self.factor['scale']['sz_range']
            sx = self._random_generator.random_uniform(
                shape=[], minval=self.sx_range[0], maxval=self.sx_range[1]
            )
            sy = self._random_generator.random_uniform(
                shape=[], minval=self.sy_range[0], maxval=self.sy_range[1]
            )
            sz = self._random_generator.random_uniform(
                shape=[], minval=self.sz_range[0], maxval=self.sz_range[1]
            )
        else:
            sx = sy = sz = ONE
        return {
            'angle': {'theta': theta, 'phi': phi, 'gamma': gamma}, 
            'translation': {'dx': dx, 'dy': dy}, 
            'scale': {'sx': sx, 'sy': sy, 'sz': sz}
        }

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        original_shape = image.shape
        image = tf.expand_dims(image, 0)
        image_shape = tf.shape(image)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        angle = transformation['angle']
        translation = transformation['translation']
        scale = transformation['scale']
        output = transform(
            image,
            get_rotation_matrix(angle, translation, scale, img_hd, img_wd),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        output = tf.squeeze(output, 0)
        output.set_shape(original_shape)
        return output

    def augment_label(self, label, transformation):
        return label

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'factor': self.factor,
            'fill_mode': self.fill_mode,
            'fill_value': self.fill_value,
            'interpolation': self.interpolation,
            'seed': self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    
    # print(os.getcwd())  # path/to/tensorflow_image_classification/
    image = tf.keras.utils.load_img(
        os.path.join('.', '1.jpg'), grayscale=False, color_mode='rgb',
        target_size=None, interpolation='nearest'
    )  # image is PIL format 
    image = tf.keras.preprocessing.image.img_to_array(image)  # image is a np.ndarray
    image_tensor = tf.convert_to_tensor(image, dtype=tf.dtypes.float32)  # tf.dtypes.float32 == tf.float32
    r3d = Random3DRotation({'angle': {'theta_range': (-60, 60), 'phi_range': (-60, 60), 'gamma_range': (-60, 60)}}, fill_mode='constant')
    # image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor2 = r3d(image_tensor)
    plt.figure()
    plt.imshow(tf.cast(image_tensor2, dtype=tf.dtypes.uint8))  # tf.dtypes.uint8 == tf.uint8
    plt.savefig(os.path.join('.', 'modules', 'layers', 'preprocessing', '3d_rotation_test.jpg'))
    print('test ended successfully')
