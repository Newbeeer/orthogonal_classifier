import numpy as np
import matplotlib.pyplot as plt
from util import mutual_information_q_u
import argparse


'''
plt.switch_backend('agg')
X = np.array(range(1, 7))
Y = np.array(range(1, 7))
label = (1,1,0,0,-1,-1)
plt.scatter(X, Y, c = label, s = 180, cmap = plt.cm.Spectral)
plt.savefig("test.png")
'''


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--latent_size", type=int)
parser.add_argument("--ms", type=int)
args = parser.parse_args()
mutual_information_q_u(args)