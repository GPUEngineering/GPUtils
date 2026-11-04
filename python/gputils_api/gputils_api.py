import os
import struct

import numpy as np


def _fixed_size_dtype(dt):
    dtype = np.dtype(dt)
    if dtype.hasobject or dtype.itemsize == 0:
        raise ValueError("The binary format requires a fixed-size dtype without objects")
    return dtype


def read_array_from_gputils_binary_file(path, dt=np.dtype('d')):
    """
    Reads an array from a bt file
    :param path: path to file
    :param dt: numpy-compatible data type
    :raises ValueError: if the file name specified `path` does not have the .bt extension
    :raises ValueError: if the header, dimensions, dtype, or payload size are invalid
    """
    if not path.endswith(".bt"):
        raise ValueError("The file must have the .bt extension")
    dtype = _fixed_size_dtype(dt)
    with open(path, 'rb') as f:
        header = f.read(24)
        if len(header) != 24:
            raise ValueError("The binary tensor header must contain exactly 24 bytes")
        # Now decode the 24-byte binary header into three integers...
        nr, nc, nm = struct.unpack('<QQQ', header)
        if not nr or not nc or not nm:
            raise ValueError("Tensor dimensions must be positive")
        count = nr * nc * nm
        payload_size = count * dtype.itemsize
        if payload_size > np.iinfo(np.intp).max:
            raise ValueError("Tensor dimensions exceed NumPy size limits")
        if os.fstat(f.fileno()).st_size - 24 != payload_size:
            raise ValueError("The binary payload size does not match its dimensions and dtype")

        dat = np.fromfile(f, dtype=dtype, count=count)
        if dat.size != count or f.read(1):
            raise ValueError("The binary payload changed size while being read")

        # Each matrix is column-major, with matrices stored consecutively.
        dat = dat.reshape((nm, nc, nr)).swapaxes(0, 2)
    return dat


def write_array_to_gputils_binary_file(x, path):
    """
    Writes a numpy array into a bt file

    :param x: numpy array to save to file
    :param path: path to file
    :raises ValueError: if `x` is empty or does not have between 1 and 3 dimensions
    :raises ValueError: if `x` has a zero-size or object-containing dtype
    :raises ValueError: if the file name specified `path` does not have the .bt extension
    """

    if not path.endswith(".bt"):
        raise ValueError("The file must have the .bt extension")
    x_shape = x.shape
    x_dims = len(x_shape)
    if not 1 <= x_dims <= 3:
        raise ValueError("The array must have between 1 and 3 dimensions")
    if x.size == 0:
        raise ValueError("Tensor dimensions must be positive")
    _fixed_size_dtype(x.dtype)
    nr = x_shape[0]
    nc = x_shape[1] if x_dims >= 2 else 1
    nm = x_shape[2] if x_dims == 3 else 1
    if x_dims == 3:
        x = x.swapaxes(0, 2).reshape(-1)  # column-major storage; axis 2 last
    else:
        x = x.T.reshape(-1)  # column-major storage
    with open(path, 'wb') as f:
        f.write(nr.to_bytes(8, 'little'))  # write number of rows
        f.write(nc.to_bytes(8, 'little'))  # write number of columns
        f.write(nm.to_bytes(8, 'little'))  # write number of matrices
        x.tofile(f)  # write data
