import unittest

import numpy as np

from .preprocess import get_preprocessing_fn


class TestPreprocess(unittest.TestCase):
    def test_preprocess_fn(self):
        fn = get_preprocessing_fn("unet", "fmow_rgb")

        data = np.ones((8, 224, 224, 3))
        processed = fn(data)
        self.assertAlmostEqual(processed.sum(), 2937294.901659388)

        data = np.ones((1, 224, 224, 3))
        processed = fn(data)
        self.assertAlmostEqual(processed.sum(), 2937294.901659388 / 8)


if __name__ == "__main__":
    unittest.main()
