import numpy as np

def read_array_from_gputils_binary_file(path, dt=np.dtype('d')):
    """
    Reads an array from a bt file
    :param path: path to file
    :param dt: numpy-compatible data type
    :raises ValueError: if the file name specified `path` does not have the .bt extension
    """
    if not path.endswith(".bt"):
        raise ValueError("The file must have the .bt extension")
    with open(path, 'rb') as f:
        nr = int.from_bytes(f.read(8), byteorder='little', signed=False)  # read number of rows
        nc = int.from_bytes(f.read(8), byteorder='little', signed=False)  # read number of columns
        nm = int.from_bytes(f.read(8), byteorder='little', signed=False)  # read number of matrices
        dat = np.fromfile(f, dtype=np.dtype(dt))  # read data
        dat = dat.reshape((nr, nc, nm))  # reshape
        dat = np.dstack(np.split(dat.reshape(6, -1), 2))  # I'll explain this to you when you grow up
    return dat


def write_array_to_gputils_binary_file(x, path):
    """
    Writes a numpy array into a bt file

    :param x: numpy array to save to file
    :param path: path to file
    :raises ValueError: if `x` has more than 3 dimensions
    :raises ValueError: if the file name specified `path` does not have the .bt extension
    """
    if not path.endswith(".bt"):
        raise ValueError("The file must have the .bt extension")
    x_shape = x.shape
    x_dims = len(x_shape)
    if x_dims >= 4:
        raise ValueError("given array cannot have more than 3 dimensions")
    nr = x_shape[0]
    nc = x_shape[1] if x_dims >= 2 else 1
    nm = x_shape[2] if x_dims == 3 else 1
    x = np.vstack(np.dsplit(x, 2)).reshape(-1)
    with open(path, 'wb') as f:
        f.write(nr.to_bytes(8, 'little'))  # write number of rows
        f.write(nc.to_bytes(8, 'little'))  # write number of columns
        f.write(nm.to_bytes(8, 'little'))  # write number of matrices
        x.tofile(f)  # write data