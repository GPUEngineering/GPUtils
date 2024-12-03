import os
import unittest
import numpy as np
import gputils_api as gpuapi


class GputilApiTestCase(unittest.TestCase):

    @staticmethod
    def local_abs_path():
        cwd = os.getcwd()
        return cwd.split('open-codegen')[0]

    @classmethod
    def setUpClass(cls):
        n = 5
        base_dir = GputilApiTestCase.local_abs_path()
        eye_d = np.eye(n, dtype=np.dtype('d'))
        gpuapi.write_array_to_gputils_binary_file(eye_d, os.path.join(base_dir, 'eye_d.bt'))
        eye_f = np.eye(n, dtype=np.dtype('f'))
        gpuapi.write_array_to_gputils_binary_file(eye_f, os.path.join(base_dir, 'eye_f.bt'))
        xd = np.random.randn(2, 4, 6).astype('d')
        gpuapi.write_array_to_gputils_binary_file(xd, os.path.join(base_dir, 'rand_246_d.bt'))
        xd = np.random.randn(3, 5, 7).astype('f')
        gpuapi.write_array_to_gputils_binary_file(xd, os.path.join(base_dir, 'rand_357_f.bt'))
        a = np.linspace(-100, 100, 5*6*7).reshape((5, 6, 7))
        gpuapi.write_array_to_gputils_binary_file(a, os.path.join(base_dir, 'rand_357_f.bt'))

    def test_read_eye_d(self):
        base_dir = GputilApiTestCase.local_abs_path()
        path = os.path.join(base_dir, 'eye_d.bt')
        r = gpuapi.read_array_from_gputils_binary_file(path, dt=np.dtype('d'))
        err = r[:, :, 0] - np.eye(5)
        print(err)


if __name__ == '__main__':
    unittest.main()