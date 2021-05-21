import h5py
import os

from urllib.request import urlopen
from urllib.request import urlretrieve

import torch as th
import numpy as np
import dgl
from dgl import DGLGraph
from pynndescent import NNDescent

def generate_point_cloud(num_points, num_features, device=th.device("cpu")):
    return th.randn((num_points, num_features), device=device)

def generate_batch_point_cloud(nb, np, nf, device=th.device("cpu")):
    return th.randn((nb, np, nf), device=device)

def process_pynndescent_result(res:np.ndarray, k, start_point=0):
    num_points = res.shape[0]
    src = res.reshape(-1) + start_point
    dst = np.arange(start_point, start_point + num_points, dtype=res.dtype)
    dst = np.repeat(dst, k)
    return (src, dst)

def pynndescent_build(data:th.Tensor, k):
    num_graph = data.shape[0]
    src, dst = [], []
    data_idx_offset = 0
    for i in range(num_graph):
        index = NNDescent(data[i], n_neighbors=k, tree_init=False, n_jobs=4, verbose=True)
        res = process_pynndescent_result(index.neighbor_graph[0], k, data_idx_offset)
        data_idx_offset += data[i].shape[0]
        src.append(res[0])
        dst.append(res[1])
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    return dgl.graph((src, dst))

def compare_two_graph(g_res: DGLGraph, g_true: DGLGraph, k):
    res = 0
    print("res num node: {}, true num node: {}".format(g_res.num_nodes(), g_true.num_nodes()))
    assert g_res.num_nodes() == g_true.num_nodes()
    for n in range(g_res.num_nodes()):
        src_res, _ = g_res.in_edges(n)
        src_true, _ = g_true.in_edges(n)
        for i in src_true:
            if th.any(i == src_res).item():
                res += 1
    res_rate = 100 * res / (k * g_res.num_nodes())
    return res_rate

def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_dataset_fn(dataset):
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    return os.path.join('datasets', '%s.hdf5' % dataset)

def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://ann-benchmarks.com/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        return None
    hdf5_f = h5py.File(hdf5_fn, 'r')

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    dimension = hdf5_f.attrs['dimension'] if 'dimension' in hdf5_f.attrs else len(hdf5_f['train'][0])

    return hdf5_f, dimension

# data, dim = get_dataset("mnist-784-euclidean")
# d = data["train"]
# dd = th.tensor(d)
# print(dd.shape)
# print(dd.dtype)
# print(dd.device)
# print(data.keys())
# print(data["train"].shape)
# print(data["distances"].shape)
# print(data["neighbors"].shape)