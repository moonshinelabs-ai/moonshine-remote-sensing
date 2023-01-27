import unittest

from public.moonshine.moonshine.models.unet import UNet50


class TestIoU(unittest.TestCase):
    def test_construct(self):
        unet = UNet50(weights=None)
        n_param = unet.num_params()

        # Should have 32,544,928 parameters
        self.assertEqual(n_param, 32570016)

        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.model_name, "UNet_ResNet50")


if __name__ == "__main__":
    unittest.main()
