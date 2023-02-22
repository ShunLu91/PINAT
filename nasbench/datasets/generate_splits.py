import numpy as np
import h5py


def denoise_nasbench(metrics, threshold=0.8):
    val_metrics = metrics[:, -1, :, -1, 2]
    index = np.where(val_metrics[:, 0] > threshold)
    return index[0]


seed = 0
with h5py.File("data/nasbench101/nasbench.hdf5", mode="r") as f:
    total_count = len(f["hash"][()])
    metrics = f["metrics"][()]
random_state = np.random.RandomState(seed)
result = dict()
split_list = [100, 172, 334, 860, 423, 424, 4236, 42362, 127087, 211812, 296537, 381262, 200, 300]
for n_samples in split_list:
    split = random_state.permutation(total_count)[:n_samples]
    result[str(n_samples)] = split

# >91
valid91 = denoise_nasbench(metrics, threshold=0.91)
for n_samples in split_list:
    result["91-" + str(n_samples)] = np.intersect1d(result[str(n_samples)], valid91)
result["denoise-91"] = valid91

result["denoise-80"] = denoise_nasbench(metrics)
np.savez("data/nasbench101/train_samples.npz", **result)
