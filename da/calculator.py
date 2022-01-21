import numpy as np


a = [
     [54.1, 54.1,54.0],
     [53.4, 53.6, 52.2 ]
     ]
mean = np.mean(a, axis=1)
std = np.std(a, axis=1)

print(f"Mean:{mean}, Std:{std}")