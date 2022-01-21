import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# kl_re = [0.69,0.69,0.69,  0.69,0.69,0.69,  0.5863, 0.5344, 0.5095,  0.5065, 0.5179, 0.5361,
#          0.5493, 0.5233, 0.5068, 0.5148, 0.5106, 0.5242,  0.5029, 0.5156, 0.5505, 0.5029, 0.5156, 0.5505]
kl_re = np.array([0.69,0.53,0.47,0.69,  0.473,0.445,   0.465,0.433,  0.431,0.441])-0.095
b1_re = np.array([0.9,0.9,0.9,0.9,  0.8,0.8,
          0.7,0.7,  0.6,0.6,])

# kl_re = np.array([0.69,0.438,0.37,0.69,  0.3611,0.3340,0.341,   0.3219,0.3454,  0.331,0.341])
# b1_re = np.array([0.9,0.9,0.9,0.9,  0.8,0.8,0.8,
#           0.7,0.7,   0.6,0.6,])

# kl_oc = [0.5815, 0.57, 0.5275,   0.594, 0.5234, 0.5425,  0.5381, 0.5334, 0.5067,  0.5068, 0.5323, 0.5284,
#          0.5410, 0.5040, 0.5069,   0.5049, 0.5040, 0.5071,  0.5039, 0.5086, 0.5149, 0.5039, 0.5086, 0.5149]
kl_oc = np.array([0.443,0.457,  0.430,0.461,0.438,0.446,  0.432,0.433,0.449,  0.439,0.435])-0.095
b1_oc = np.array([0.9,0.9,    0.8,0.8,0.8,0.8,
         0.7,0.7,0.7,   0.6,0.6,])


f = lambda x: np.log(2 * (0.5**2/(1-x) + 0.5**2/x) + 1)
b1_oc = f(b1_oc)
b1_re = f(b1_re)
sns.set(style='whitegrid', font_scale=1.8,rc={"lines.linewidth": 3.5})
d_2 = {'Test loss': kl_re, '$\log \mathbb{E}_y[ D_f(p_{Z_1} || p_{Z_1|Y=y})+1]$': b1_re}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='$\log \mathbb{E}_y[ D_f(p_{Z_1} || p_{Z_1|Y=y})+1]$', y='Test loss', data=pdnumsqr_2,label='IS',ci=80)

d_2 = {'Test loss': kl_oc, '$\log \mathbb{E}_y[ D_f(p_{Z_1} || p_{Z_1|Y=y})+1]$': b1_oc}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='$\log \mathbb{E}_y[ D_f(p_{Z_1} || p_{Z_1|Y=y})+1]$', y='Test loss', data=pdnumsqr_2,label='$w_x$\ $w_1$',ci=80)

#plt.hlines(0.506,min(b1_re),max(b1_re),colors='k',linestyles='--',label='Oracle')
plt.hlines(0.325,min(b1_re),max(b1_re),colors='k',linestyles='--',label='Oracle')
plt.legend()
plt.savefig('w2_loss.pdf', dpi=300, bbox_inches='tight')
plt.show()


kl_re = np.array([0.69,0.47,0.53,0.69, 0.5351,0.488,  0.441,0.444]) - 0.095
b1_re = np.array([-1,-1,-1,-1, -2,-2,  -3,-3])


kl_oc = np.array([0.443,0.457, 0.441,0.440,  0.439,0.437]) - 0.095
b1_oc = np.array([-1,-1, -2,-2,  -3,-3])

sns.set(style='whitegrid', font_scale=1.8,rc={"lines.linewidth": 3.5})
d_2 = {'Test loss': kl_re, 'Learning rate (log-scale)': b1_re}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='Learning rate (log-scale)', y='Test loss', data=pdnumsqr_2,label='IS',ci=80)

d_2 = {'Test loss': kl_oc, 'Learning rate (log-scale)': b1_oc}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='Learning rate (log-scale)', y='Test loss', data=pdnumsqr_2,label='$w_x$\ $w_1$',ci=80)

plt.hlines(0.325,min(b1_re),max(b1_re),colors='k',linestyles='--',label='Oracle')
plt.legend()
plt.savefig('w2_loss_lr.pdf', dpi=300, bbox_inches='tight')
plt.show()

