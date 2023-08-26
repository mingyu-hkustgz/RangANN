import numpy as np
import struct

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]




def mmap_vecs(fname, dimension, dtype):
    x = np.memmap(fname, dtype=dtype, mode='r')
    return x.reshape(-1, dimension)


def mmap_flat(fname, dtype):
    x = np.memmap(fname, dtype=dtype, mode='r')
    return x.ravel()


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    n, d = data.shape
    with open(filename, 'wb') as fp:
        item = struct.pack('I', n)
        fp.write(item)
        item = struct.pack('I', d)
        fp.write(item)
        for y in data:
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)