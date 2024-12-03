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
        xd[1, 2, 3] = -12.3
        gpuapi.write_array_to_gputils_binary_file(xd, os.path.join(base_dir, 'rand_246_d.bt'))

        xf = np.random.randn(2, 4, 6).astype('f')
        xf[1, 2, 3] = float(-12.3)
        gpuapi.write_array_to_gputils_binary_file(xf, os.path.join(base_dir, 'rand_246_f.bt'))

        a = np.linspace(-100, 100, 4*5).reshape((4,5)).astype('d')
        gpuapi.write_array_to_gputils_binary_file(a, os.path.join(base_dir, 'a_d.bt'))

        b = np.array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [-11, 12]]
        ], dtype=np.dtype('d'))
        gpuapi.write_array_to_gputils_binary_file(b, os.path.join(base_dir, 'b_d.bt'))

    def __test_read_eye(self, dt):
        base_dir = GputilApiTestCase.local_abs_path()
        path = os.path.join(base_dir, f'eye_{dt}.bt')
        r = gpuapi.read_array_from_gputils_binary_file(path, dt=np.dtype(dt))
        err = r[:, :, 0] - np.eye(5)
        err_norm = np.linalg.norm(err, np.inf)
        self.assertTrue(err_norm < 1e-12)

    def test_read_eye_d(self):
        self.__test_read_eye('d')

    def test_read_eye_f(self):
        self.__test_read_eye('f')

    def __test_read_rand(self, dt):
        base_dir = GputilApiTestCase.local_abs_path()
        path = os.path.join(base_dir, f'rand_246_{dt}.bt')
        r = gpuapi.read_array_from_gputils_binary_file(path, dt=np.dtype(dt))
        r_shape = r.shape
        self.assertEqual(2, r_shape[0])
        self.assertEqual(4, r_shape[1])
        self.assertEqual(6, r_shape[2])
        e = np.abs(r[1, 2, 3]+12.3)
        self.assertTrue(e < 1e-6)

    def test_read_rand_d(self):
        self.__test_read_rand('d')

    def test_read_rand_f(self):
        self.__test_read_rand('f')


if __name__ == '__main__':
    unittest.main()