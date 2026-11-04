import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import gputils_api as gpuapi


DTYPES = (np.float32, np.float64)


class FileTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='gputils-io-')
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = str(self.directory / 'tensor.bt')

    def assert_round_trip(self, original):
        gpuapi.write_array_to_gputils_binary_file(original, self.path)
        actual = gpuapi.read_array_from_gputils_binary_file(self.path, original.dtype)
        expected_shape = original.shape + (1,) * (3 - original.ndim)
        self.assertEqual(actual.shape, expected_shape)
        self.assertEqual(actual.dtype, original.dtype)
        np.testing.assert_array_equal(actual, original.reshape(expected_shape))
        return actual


class GputilApiTestCase(FileTestCase):
    def test_read_eye_d(self):
        self.assert_round_trip(np.eye(5, dtype=np.float64))

    def test_read_eye_f(self):
        self.assert_round_trip(np.eye(5, dtype=np.float32))

    def test_single_matrix_round_trip(self):
        self.assert_round_trip(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64))

    def test_read_rand_d(self):
        # Deterministic values and a full comparison replace random spot checks.
        self.assert_round_trip(np.arange(48, dtype=np.float64).reshape(2, 4, 6) / 4 - 5)

    def test_read_rand_f(self):
        self.assert_round_trip(np.arange(48, dtype=np.float32).reshape(2, 4, 6) / 4 - 5)

    def test_read_a_tensor_3d(self):
        self.assert_round_trip(np.arange(18, dtype=np.float64).reshape(3, 3, 2) - 9)

    def test_round_trip_shapes_and_precisions(self):
        for dtype in DTYPES:
            for shape in ((1,), (5,), (1, 1), (1, 5), (5, 1), (2, 3), (2, 3, 1), (2, 3, 2)):
                with self.subTest(dtype=dtype, shape=shape):
                    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) / 4 - 3
                    self.assert_round_trip(data)

    def test_noncontiguous_and_fortran_inputs(self):
        for dtype in DTYPES:
            matrix = np.arange(48, dtype=dtype).reshape(6, 8)
            tensor = np.arange(120, dtype=dtype).reshape(4, 6, 5)
            inputs = (matrix.T, matrix[::2, 1::2], matrix[::-1, ::-1],
                      tensor[::2, 1::2, ::-1], np.asfortranarray(matrix))
            for index, data in enumerate(inputs):
                with self.subTest(dtype=dtype, layout=index):
                    self.assert_round_trip(data)

    def test_writer_matches_independent_binary_layout(self):
        for dtype, code in ((np.float32, 'f'), (np.float64, 'd')):
            for shape in ((2, 3, 1), (2, 3, 2)):
                with self.subTest(dtype=dtype, shape=shape):
                    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) / 4 - 2
                    expected = struct.pack('<QQQ', *shape)
                    # Encode each logical element independently of NumPy reshape/transposition.
                    for k in range(shape[2]):
                        for c in range(shape[1]):
                            for r in range(shape[0]):
                                expected += struct.pack('<' + code, float(data[r, c, k]))
                    gpuapi.write_array_to_gputils_binary_file(data, self.path)
                    self.assertEqual(Path(self.path).read_bytes(), expected)

    def test_reader_accepts_independently_encoded_binary_layout(self):
        for dtype, code in ((np.float32, 'f'), (np.float64, 'd')):
            with self.subTest(dtype=dtype):
                # Two 2x3 matrices, each stored column by column.
                payload = (1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6)
                Path(self.path).write_bytes(struct.pack('<QQQ', 2, 3, 2)
                                           + struct.pack('<12' + code, *payload))
                actual = gpuapi.read_array_from_gputils_binary_file(self.path, dtype)
                expected = np.stack(([[1, 2, 3], [4, 5, 6]],
                                     [[-1, -2, -3], [-4, -5, -6]]), axis=2).astype(dtype)
                np.testing.assert_array_equal(actual, expected)

    def test_binary_preserves_floating_point_values(self):
        for dtype in DTYPES:
            limits = np.finfo(dtype)
            data = np.array([0., -0., limits.tiny, -limits.tiny, limits.max, -limits.max,
                             np.nextafter(dtype(0), dtype(1)),
                             np.nextafter(dtype(1), dtype(2)), np.inf, -np.inf, np.nan], dtype=dtype)
            with self.subTest(dtype=dtype):
                actual = self.assert_round_trip(data).ravel()
                np.testing.assert_array_equal(np.signbit(actual), np.signbit(data))

    def test_reader_rejects_every_truncated_header_length(self):
        header = struct.pack('<QQQ', 2, 3, 1)
        for length in range(24):
            with self.subTest(length=length):
                Path(self.path).write_bytes(header[:length])
                with self.assertRaises(ValueError):
                    gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_reader_rejects_truncated_payloads(self):
        for dtype, code in ((np.float32, 'f'), (np.float64, 'd')):
            payload = struct.pack('<6' + code, 1, 4, 2, 5, 3, 6)
            for length in (0, 1, len(payload) // 2, len(payload) - 1):
                with self.subTest(dtype=dtype, length=length):
                    Path(self.path).write_bytes(struct.pack('<QQQ', 2, 3, 1) + payload[:length])
                    with self.assertRaises(ValueError):
                        gpuapi.read_array_from_gputils_binary_file(self.path, dtype)

    def test_reader_rejects_trailing_bytes_and_elements(self):
        for dtype, code in ((np.float32, 'f'), (np.float64, 'd')):
            valid = struct.pack('<QQQ', 2, 3, 1) + struct.pack('<6' + code, 1, 4, 2, 5, 3, 6)
            for suffix in (b'\xff', struct.pack('<' + code, 7)):
                with self.subTest(dtype=dtype, extra_bytes=len(suffix)):
                    Path(self.path).write_bytes(valid + suffix)
                    with self.assertRaises(ValueError):
                        gpuapi.read_array_from_gputils_binary_file(self.path, dtype)

    def test_reader_rejects_inconsistent_and_oversized_dimensions(self):
        for shape in ((2, 3, 2), (2**64 - 1, 2, 1), (2**63, 2**63, 2)):
            with self.subTest(shape=shape):
                Path(self.path).write_bytes(struct.pack('<QQQ', *shape) + struct.pack('<6d', *range(6)))
                with self.assertRaises(ValueError):
                    gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_wrong_dtype_with_incompatible_payload_size_is_rejected(self):
        gpuapi.write_array_to_gputils_binary_file(np.array([[1., 2., 3.]], dtype=np.float64), self.path)
        with self.assertRaises(ValueError):
            gpuapi.read_array_from_gputils_binary_file(self.path, np.float32)

    def test_dtype_must_be_supplied_for_untyped_format(self):
        original = np.array([1., 2.], dtype=np.float32)
        gpuapi.write_array_to_gputils_binary_file(original, self.path)
        actual = gpuapi.read_array_from_gputils_binary_file(self.path, np.int32)
        # Same-width wrong dtypes cannot be detected: there is no dtype tag in the format.
        np.testing.assert_array_equal(actual[:, 0, 0], original.view(np.int32))

    def test_explicit_byte_order_round_trips(self):
        for dtype in ('<f4', '>f4', '<f8', '>f8', '<i4'):
            with self.subTest(dtype=dtype):
                self.assert_round_trip(np.array([[1, -2, 3], [4, 5, -6]], dtype=dtype))

    def test_invalid_extensions(self):
        for name in ('tensor', 'tensor.txt', 'tensor.BT'):
            with self.subTest(name=name):
                path = str(self.directory / name)
                with self.assertRaises(ValueError):
                    gpuapi.write_array_to_gputils_binary_file(np.ones(2), path)
                with self.assertRaises(ValueError):
                    gpuapi.read_array_from_gputils_binary_file(path)
                self.assertFalse(Path(path).exists())

    def test_missing_input_file(self):
        with self.assertRaises(FileNotFoundError):
            gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_missing_output_directory(self):
        with self.assertRaises(FileNotFoundError):
            gpuapi.write_array_to_gputils_binary_file(np.ones(2), str(self.directory / 'missing' / 'tensor.bt'))

    def test_reading_or_writing_a_directory(self):
        Path(self.path).mkdir()
        with self.assertRaises(IsADirectoryError):
            gpuapi.write_array_to_gputils_binary_file(np.ones(2), self.path)
        with self.assertRaises(IsADirectoryError):
            gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_writer_rejects_more_than_three_dimensions(self):
        with self.assertRaises(ValueError):
            gpuapi.write_array_to_gputils_binary_file(np.ones((1, 1, 1, 1)), self.path)
        self.assertFalse(Path(self.path).exists())

    def test_writer_rejects_empty_arrays(self):
        for shape in ((0,), (0, 3), (2, 0), (2, 3, 0)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    gpuapi.write_array_to_gputils_binary_file(np.empty(shape), self.path)
                self.assertFalse(Path(self.path).exists())

    def test_writer_rejects_zero_dimensional_arrays(self):
        with self.assertRaises(ValueError):
            gpuapi.write_array_to_gputils_binary_file(np.array(3.0), self.path)
        self.assertFalse(Path(self.path).exists())

    def test_reader_rejects_zero_dimensions(self):
        for shape in ((0, 1, 1), (2, 0, 1), (2, 3, 0)):
            with self.subTest(shape=shape):
                Path(self.path).write_bytes(struct.pack('<QQQ', *shape))
                with self.assertRaises(ValueError):
                    gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_invalid_file_is_rejected_before_loading_payload(self):
        inputs = (b'', struct.pack('<QQQ', 0, 1, 1),
                  struct.pack('<QQQ', 2, 3, 1) + b'\x00',
                  struct.pack('<QQQ', 2**64 - 1, 2, 1))
        for content in inputs:
            with self.subTest(header=content):
                Path(self.path).write_bytes(content)
                with mock.patch('gputils_api.gputils_api.np.fromfile') as load:
                    with self.assertRaises(ValueError):
                        gpuapi.read_array_from_gputils_binary_file(self.path)
                    load.assert_not_called()

    def test_reader_rejects_short_read_after_size_validation(self):
        self.assert_round_trip(np.arange(6, dtype=np.float64).reshape(2, 3))
        with mock.patch('gputils_api.gputils_api.np.fromfile', return_value=np.zeros(5)):
            with self.assertRaises(ValueError):
                gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_reader_rejects_file_growth_during_payload_read(self):
        self.assert_round_trip(np.arange(6, dtype=np.float64).reshape(2, 3))
        load = np.fromfile

        def load_then_append(*args, **kwargs):
            data = load(*args, **kwargs)
            with open(self.path, 'ab') as output:
                output.write(b'\xff')
            return data

        with mock.patch('gputils_api.gputils_api.np.fromfile', side_effect=load_then_append):
            with self.assertRaises(ValueError):
                gpuapi.read_array_from_gputils_binary_file(self.path)

    def test_invalid_arrays_do_not_overwrite_existing_file(self):
        sentinel = b'existing file must survive rejected input'
        for data in (np.array(3.), np.empty((2, 0)), np.ones((1, 1, 1, 1))):
            with self.subTest(shape=data.shape):
                Path(self.path).write_bytes(sentinel)
                with self.assertRaises(ValueError):
                    gpuapi.write_array_to_gputils_binary_file(data, self.path)
                self.assertEqual(Path(self.path).read_bytes(), sentinel)

    def test_object_and_zero_size_dtypes_are_rejected(self):
        for dtype in (np.dtype(object), np.dtype([('value', object)]), np.dtype('V0')):
            with self.subTest(dtype=dtype):
                self.assert_round_trip(np.array([1., 2.]))
                sentinel = Path(self.path).read_bytes()
                with self.assertRaises(ValueError):
                    gpuapi.read_array_from_gputils_binary_file(self.path, dtype)
                with self.assertRaises(ValueError):
                    gpuapi.write_array_to_gputils_binary_file(np.zeros(2, dtype=dtype), self.path)
                self.assertEqual(Path(self.path).read_bytes(), sentinel)

    def test_overwrite_truncates_previous_payload(self):
        self.assert_round_trip(np.arange(100, dtype=np.float64))
        self.assert_round_trip(np.array([1., 2.]))
        self.assertEqual(Path(self.path).stat().st_size, 24 + 2 * 8)


@unittest.skipUnless(sys.byteorder == 'little', 'C++ binary interoperability currently requires a little-endian host')
class CppFileExchangeTest(FileTestCase):
    @classmethod
    def setUpClass(cls):
        default = Path(__file__).resolve().parents[2] / 'build' / 'test' / 'gputils_file_exchange'
        explicit = os.environ.get('GPUTILS_FILE_EXCHANGE')
        cls.driver = Path(explicit) if explicit else default
        if not cls.driver.is_file():
            if explicit:
                raise RuntimeError('GPUTILS_FILE_EXCHANGE does not identify a built executable')
            raise unittest.SkipTest('Build gputils_file_exchange or set GPUTILS_FILE_EXCHANGE for C++ interoperability tests')

    @staticmethod
    def exchange_array(shape, dtype):
        r, c, k = np.indices(shape)
        return (r + 10 * c + 100 * k + .25).astype(dtype)

    def run_driver(self, mode, dtype, shape):
        result = subprocess.run([str(self.driver), mode, dtype, self.path, *map(str, shape)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_python_files_read_by_cpp(self):
        for dtype, name in ((np.float32, 'float'), (np.float64, 'double')):
            for shape in ((1, 1, 1), (5, 1, 1), (1, 4, 1), (2, 3, 1), (2, 3, 2)):
                with self.subTest(dtype=name, shape=shape):
                    gpuapi.write_array_to_gputils_binary_file(self.exchange_array(shape, dtype), self.path)
                    self.run_driver('verify', name, shape)

    def test_cpp_files_read_by_python(self):
        for dtype, name in ((np.float32, 'float'), (np.float64, 'double')):
            for shape in ((1, 1, 1), (5, 1, 1), (1, 4, 1), (2, 3, 1), (2, 3, 2)):
                with self.subTest(dtype=name, shape=shape):
                    self.run_driver('write', name, shape)
                    actual = gpuapi.read_array_from_gputils_binary_file(self.path, dtype)
                    self.assertEqual(actual.shape, shape)
                    np.testing.assert_array_equal(actual, self.exchange_array(shape, dtype))


if __name__ == '__main__':
    unittest.main()
