import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)
orth = True
src_mat = np.load(f'src_mat_{orth}.npy')
tgt_mat = np.load(f'tgt_mat_{orth}.npy')
src_label = np.load(f'src_label_{orth}.npy')
tgt_label = np.load(f'tgt_label_{orth}.npy')
print(src_mat.shape, tgt_mat.shape, src_label.shape, tgt_label.shape)
X = np.concatenate((src_mat, tgt_mat), axis=0)
print("Shape of X: ", X.shape)

# X_embedded = TSNE(n_components=2).fit_transform(X)
# np.save(f'tsne_{orth}', X_embedded)
X_embedded = np.load(f'tsne_{orth}.npy')
print("Shape of X_e: ", X_embedded.shape)

src_embed = X_embedded[: len(src_mat)]
tgt_embed = X_embedded[len(src_mat):]
src_label = (src_label >= 5).astype(np.int)
tgt_label = (tgt_label >= 5).astype(np.int)
print(src_embed[src_label==0].shape, src_label[src_label==0].shape, src_label.shape, src_embed.shape)
fig, ax = plt.subplots()


c0 = np.ones(len(src_embed[src_label==0])).astype(np.int) * 0
c1 = np.ones(len(src_embed[src_label==1])).astype(np.int)
c2 = np.ones(len(tgt_embed[tgt_label==0])).astype(np.int) * 2
c3 = np.ones(len(tgt_embed[tgt_label==1])).astype(np.int) * 3

ax.scatter(src_embed[src_label==0][:len(c0)//10, 0], src_embed[src_label==0][:len(c0)//10, 1], c='r', label=r'source $\tilde{\mathcal{C}}$', alpha=0.2)
ax.scatter(tgt_embed[tgt_label==0][:len(c2)//5, 0], tgt_embed[tgt_label==0][:len(c2)//5, 1], c='c', label=r'target $\tilde{\mathcal{C}}$', alpha=0.2)
#ax.scatter(src_embed[src_label==1][:len(c1)//5, 0], src_embed[src_label==1][:len(c1)//5, 1], c='y', label='source 1', alpha=0.2)
#ax.scatter(tgt_embed[tgt_label==1][:len(c3)//10, 0], tgt_embed[tgt_label==1][:len(c3)//10, 1], c='blue', label='target 1', alpha=0.2)
ax.legend()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.box(False)
plt.savefig(f'tsne_{orth}.pdf', dpi=300, bbox_inches='tight')
#plt.show()