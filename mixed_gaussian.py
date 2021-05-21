from dgl.transform import knn_graph
import random
from utils import get_dataset, compare_two_graph, generate_point_cloud
from time import time
import numpy as np
import torch as th

def generate_mixed_gaussian(num_list, f, device=th.device("cpu")):
    res = []
    for n in num_list:
        res.append(generate_point_cloud(n, f, device) * random.random() + random.randint(-10, 10))
    return th.cat(res, dim=0)

k_list = [8, 32]
nums = 10 * [100]
fd = 3

for k in k_list:
    for device_type in ["cpu", "cuda"]:
        device = th.device(device_type)
        data = generate_mixed_gaussian(nums, fd, device)

        # warmup
        print("warmup")
        try:
            g = knn_graph(data, k)
        except:
            g = None
        del g

        try:
            print("BLAS...")
            s_time = time()
            g = knn_graph(data, k)
            torch_time = time() - s_time
        except:
            g = None
            torch_time = float("nan")

        try:
            print("NNDescent...")
            s_time = time()
            g_nnd = knn_graph(data, k, algorithm="nn-descent")
            nnd_time = time() - s_time
        except:
            g_nnd = None
            nnd_time = float("nan")
        
        try:
            print("BF...")
            s_time = time()
            gf = knn_graph(data, k, algorithm="bruteforce")
            bf_time = time() - s_time
        except:
            bf_time = float("nan")
        
        if device_type == "cpu":
            try:
                print("KD Tree...")
                s_time = time()
                _ = knn_graph(data, k, algorithm="kd-tree")
                kd_time = time() - s_time
            except:
                kd_time = float("nan")
            # kd_time = 0

        if device_type == "cuda":
            try:
                print("bf sharemem...")
                s_time = time()
                _ = knn_graph(data, k, algorithm="bruteforce-sharemem")
                sharemem_time = time() - s_time
            except:
                sharemem_time = float("nan")
            # sharemem_time = 0

        if g is not None and g_nnd is not None:
            rate_nnd = compare_two_graph(g_nnd, g, k)
        elif g_nnd is not None:
            rate_nnd = compare_two_graph(g_nnd, gf, k)
        else:
            rate_nnd = float("nan")

        if device_type == "cuda":
            print("device: {}, shape: {}, k: {:2d}, blas_time: {:.4f}, bf_time: {:.4f}, bfs_time: {:.4f}, nnd_time: {:.4f}, match_rate = {:.4f}%".format(
                device_type, data.shape, k, torch_time, bf_time, sharemem_time, nnd_time, rate_nnd))
        else:
            print("device: {}, shape: {}, k: {:2d}, blas_time: {:.4f}, bf_time: {:.4f}, kd_time: {:.4f}, nnd_time: {:.4f}, match_rate = {:.4f}%".format(
                device_type, data.shape, k, torch_time, bf_time, kd_time, nnd_time, rate_nnd))
