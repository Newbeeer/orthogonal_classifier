import numpy as np

z = np.array([[72,73,73 ], [0.22,0.18,0.18], [0.12,0.10,0.10]])

def acc(z):

    for i in range(len(z)):
        print(f"mean:{np.mean(z[i])}, std:{np.std(z[i])}")

acc(z)