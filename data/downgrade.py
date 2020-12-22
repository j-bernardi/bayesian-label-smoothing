import numpy as np
import os

xs = np.load(os.path.join("data", "combined.npy")).astype(np.uint8)
ys = np.load(os.path.join("data", "segmented.npy")).astype(np.uint8)

xs2 = np.load(os.path.join("data", "combined.npy"))
ys2 = np.load(os.path.join("data", "segmented.npy"))

print(xs.dtype)
print(ys.dtype)
np.save(os.path.join("data", "combined_uint.npy"), xs)
np.save(os.path.join("data", "segmented_uint.npy"), ys)

xs = xs.astype(np.int64)
ys = ys.astype(np.int64)

assert np.all(xs == xs2)
assert np.all(ys == ys2)
