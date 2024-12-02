import os
import unittest
import numpy as np
import gputils_api as gpuapi


class GputilApiTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 5
        eye_d = np.eye(n, dtype=np.dtype('d'))
        gpuapi.write_array_to_gputils_binary_file(eye_d, 'eye_d.bt')
        eye_f = np.eye(n, dtype=np.dtype('f'))
        gpuapi.write_array_to_gputils_binary_file(eye_f, 'eye_f.bt')
        xd = np.random.randn(2, 4, 6).astype('d')
        gpuapi.write_array_to_gputils_binary_file(xd, 'rand_246_d.bt')
        xd = np.random.randn(3, 5, 7).astype('f')
        gpuapi.write_array_to_gputils_binary_file(xd, 'rand_357_f.bt')

    def test_asdf(self):
        pass


if __name__ == '__main__':
    unittest.main()