import unittest

from PIL import Image
import numpy
import numpy as np
from numpy.testing import assert_array_equal


def pillow_image_to_normalized_numpy_array(pillow_image):
    pix = numpy.array(pillow_image)
    pix = pix / 127.5 - 1.
    return pix


class TestStringMethods(unittest.TestCase):

    def test_convert_empty_black_image(self):
        # given
        w, h = 512, 512
        data = np.zeros((h, w, 3), dtype=np.uint8)
        expected_img_as_np_array = np.full((h, w, 3), -1., dtype=np.float)
        img = Image.fromarray(data, 'RGB')
        # when
        img_as_np_array = pillow_image_to_normalized_numpy_array(img)
        # then
        assert_array_equal(img_as_np_array, expected_img_as_np_array)


if __name__ == '__main__':
    unittest.main()
