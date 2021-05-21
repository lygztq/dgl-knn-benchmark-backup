from dgl.transform import knn_graph
from utils import get_dataset, compare_two_graph
from time import time
import numpy as np
import torch as th

# dataset = ShapeNet("./datasets")
# # N = 50000
# # print(dataset.data.x.shape)
# x = dataset.data.x.cpu().numpy()
# print(type(x))
# np.save("./datasets/shapenet.npy", x)
num = 50000
x = np.load("./datasets/shapenet.npy")[:num]

k_list = [8, 32]

for k in k_list:
    for device_type in ["cpu", "cuda"]:
        device = th.device(device_type)
        data = th.tensor(x, device=device)

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

