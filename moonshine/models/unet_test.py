import unittest

from public.moonshine.moonshine.models.unet import UNet50


class TestUNet(unittest.TestCase):
    def test_construct(self):
        unet = UNet50(weights=None)
        n_param = unet.num_params()

        # Should have 32,544,928 parameters
        self.assertEqual(n_param, 32570016)

        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.model_name, "UNet_ResNet50")

    @unittest.skip("Big download need to find a better unittest")
    def test_load(self):
        unet = UNet50(weights="fmow_rgb")
        n_param = unet.num_params()

        # Should have 32,544,928 parameters
        self.assertEqual(n_param, 32570016)
        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.model_name, "UNet_ResNet50")
 
    def test_fails(self):
        with self.assertRaises(Exception):
            _ = UNet50(weights="no_valid_weights")

if __name__ == "__main__":
    unittest.main()
