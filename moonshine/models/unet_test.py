import unittest

from .unet import UNet


class TestUNet(unittest.TestCase):
    def test_construct(self):
        unet = UNet(name="unet50_fmow_rgb")
        n_param = unet.num_params()

        self.assertEqual(n_param, 32541792)

        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.name, "unet50_fmow_rgb")

    @unittest.skip("Big download need to find a better unittest")
    def test_load_full(self):
        unet = UNet(name="unet50_fmow_full")
        n_param = unet.num_params()

        self.assertEqual(n_param, 32557472)
        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.name, "unet50_fmow_full")

        unet.load_weights(
            encoder_weights="unet50_fmow_full", decoder_weights="unet50_fmow_full"
        )

    @unittest.skip("Big download need to find a better unittest")
    def test_load_rgb(self):
        unet = UNet(name="unet50_fmow_rgb")
        n_param = unet.num_params()

        self.assertEqual(n_param, 32541792)
        self.assertTrue(len(unet.describe()))
        self.assertEqual(unet.name, "unet50_fmow_rgb")

        unet.load_weights(
            encoder_weights="unet50_fmow_rgb", decoder_weights="unet50_fmow_rgb"
        )

    def test_fails(self):
        with self.assertRaises(Exception):
            _ = UNet(name="not_a_network")


if __name__ == "__main__":
    unittest.main()
