import unittest

from .unet import UNet


class TestUNet(unittest.TestCase):
    def test_construct(self):
        unet = UNet(variety="resnet50", weights=None)
        n_param = unet.num_params()

        self.assertEqual(n_param, 32541792)

        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.model_name, "UNet_ResNet50")

    @unittest.skip("Big download need to find a better unittest")
    def test_load(self):
        unet = UNet(variety="resnet50", weights="fmow_rgb")
        n_param = unet.num_params()

        self.assertEqual(n_param, 32541792)
        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.model_name, "UNet_ResNet50")

    def test_fails(self):
        with self.assertRaises(Exception):
            _ = UNet(variety="resnet50", weights="no_valid_weights")
        with self.assertRaises(Exception):
            _ = UNet(variety="not_a_network", weights=None)


if __name__ == "__main__":
    unittest.main()
