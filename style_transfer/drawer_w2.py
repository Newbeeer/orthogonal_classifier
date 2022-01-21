import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


kl_reweight = [1.35,1.26,1.24,1.20,  1.35,1.31,1.40,1.48,  1.54,1.43,1.78,1.61, 2.0,2.38,3.38]
b1 = [0.2,0.2,0.2,0.2, 0.4,0.4,0.4,0.4, 0.6,0.6,0.6,0.6, 0.8,0.8,0.8]

kl_reweight = [0.5208,0.5057,0.5367,0.5077,  0.5194,0.5282,0.5205,0.5086,  0.5240,0.5116,0.5706,   0.5812,0.5728]
b1 = [0.2,0.2,0.2,0.2,  0.4,0.4,0.4,0.4,  0.6,0.6,0.6,  0.8,0.8]

kl_reweight_2 = [0.5457, 0.5379, 0.5657]
kl_reweight_4 = [0.5078, 0.5122, 0.5159]

kl_oc = [0.5118,0.5492,0.5124,   0.5128,0.5116,0.5180,  0.5191,0.5401,0.5106,0.5106,  0.5119,0.5176]
b1_oc = [0.2,0.2,0.2,   0.4,0.4,0.4,  0.6,0.6,0.6,0.6,   0.8,0.8]

sns.set(style='whitegrid', font_scale=1.4,rc={"lines.linewidth": 3.5})
d_2 = {'Average KL$(p||w_2)$': kl_reweight, 'bias degree of $z_1$': b1}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='bias degree of $z_1$', y='Average KL$(p||w_2)$', data=pdnumsqr_2,label='Reweighted',ci=80)

d_2 = {'Average KL$(p||w_2)$': kl_oc, 'bias degree of $z_1$': b1_oc}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='bias degree of $z_1$', y='Average KL$(p||w_2)$', data=pdnumsqr_2,label='$w_x$\ $w_1$',ci=80)

# d_2 = {'Accuracy': acc_oracle, 'r': s_oracle}
# pdnumsqr_2 = pd.DataFrame(d_2)
# ax = sns.lineplot(x='r', y='Accuracy', color='black', data=pdnumsqr_2,label='Oracle',ci=85)
# ax.lines[5].set_linestyle("--")
# oracle
plt.hlines(0.506,0.2,0.8,colors='k',linestyles='--',label='Oracle')

plt.legend()
plt.savefig('w2_bias.pdf', dpi=300, bbox_inches='tight')
plt.show()


kl_reweight_1 = [0.5240, 0.5116, 0.5706]
kl_reweight_2 = [0.5176, 0.5179, 0.5657]
kl_reweight_4 = [0.5078, 0.5122, 0.5159]
kl = kl_reweight_1 + kl_reweight_2 + kl_reweight_4

s = [128,128,128, 256,256,256, 512,512,512]
sns.set(style='whitegrid', font_scale=1.4,rc={"lines.linewidth": 3.5})
d_2 = {'Average KL$(p||w_2)$': kl, 'batch size': s}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='batch size', y='Average KL$(p||w_2)$', data=pdnumsqr_2,label='Reweighted',ci=80)
# d_2 = {'Average KL$(p||w_2)$': kl_reweight_2, 'batch size': s}
# pdnumsqr_2 = pd.DataFrame(d_2)
# sns.lineplot(x='batch size', y='Average KL$(p||w_2)$', data=pdnumsqr_2,label='Reweighted',ci=80)
# d_2 = {'Average KL$(p||w_2)$': kl_reweight_4, 'batch size': s}
# pdnumsqr_2 = pd.DataFrame(d_2)
# sns.lineplot(x='batch size', y='Average KL$(p||w_2)$', data=pdnumsqr_2,label='Reweighted',ci=80)
plt.legend()
plt.savefig('w2_bias_bs.pdf', dpi=300, bbox_inches='tight')
plt.show()