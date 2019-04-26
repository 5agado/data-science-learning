import unittest
import automata_blender_utils
import numpy as np


class UtilsTest(unittest.TestCase):
    def test_binToColor(self):
        # Blue
        color = automata_blender_utils.bin_to_color(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]))
        self.assertEqual(color, (0., 0., 1.0))
        # Red
        color = automata_blender_utils.bin_to_color(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(color, (1., 0., 0.))
        # White
        color = automata_blender_utils.bin_to_color(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
        self.assertEqual(color, (1., 1., 1.))
        # White uneven length (moved to 2BPC, and last two bits are discarded
        color = automata_blender_utils.bin_to_color(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
        self.assertEqual(color, (1., 1., 1.))
        # Insufficient len, defaults to gray
        color = automata_blender_utils.bin_to_color(np.array([1, 1]))
        self.assertEqual(color, (0.5, 0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
