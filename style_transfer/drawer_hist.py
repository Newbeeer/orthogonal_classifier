import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='whitegrid', font_scale=1.8,rc={"lines.linewidth": 3.5})
# r1 = 0.6
# r2 = 0.6
# r3 = 0.8
# # dis = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_data_reweight_w1.npy')
# # label = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_label_reweight_w1.npy')
# dis = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_True_data.npy')
# label = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_True_label.npy')
# print(dis)
# print(label)
# plt.vlines(1/9, ymin=0, ymax=3000, colors='r')
# plt.vlines(9/11, ymin=0, ymax=3000,colors='r')
# plt.hist(dis[label==1],bins=100,range=(0,1),label='$P(Y=1|$ Blue background $)$')
# plt.hist(dis[label==0],bins=100,range=(0,1),label='$P(Y=1|$ Red background $)$')
# plt.title('Oracle Classifier')
# plt.legend()
# plt.xlabel("probability")
# plt.ylabel("# of data")
# plt.savefig(f'bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_oracle.pdf', dpi=300, bbox_inches='tight')
# plt.show()

r1 = 0.0
r2 = 0.8
r3 = 0.8
# dis = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_data_reweight_w1.npy')
# label = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_label_reweight_w1.npy')
dis = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_oc_data.npy')
label = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_oc_label.npy')
dis_re = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_re_data.npy')
label_re = np.load(f'./checkpoint/bias_{r1:.1f}_{r2:.1f}_{r3:.1f}_re_label.npy')
print(dis)
print(label)
plt.vlines(0.1, ymin=0, ymax=2000, colors='r')
plt.vlines(0.9, ymin=0, ymax=2000,colors='r')
p = plt.hist(dis[label==1],bins=100,range=(0,1),label='$P(Y=1|$ B$)$ ($w_x$\ $w_1$)', color='orange', alpha=0.4)
plt.hist(dis[label==0]+0.05,bins=100,range=(0,1),label='$P(Y=1|$ G$)$ ($w_x$\ $w_1$)',color='blue', alpha=0.4)
plt.hist(dis_re[label_re==1],bins=100,range=(0,1),label='$P(Y=1|$ B$)$ (IS)', color='orange')
plt.hist(dis_re[label_re==0],bins=100,range=(0,1),label='$P(Y=1|$ G$)$ (IS)', color='blue')
#plt.title('Orthogonal Classifier')
plt.legend(prop={'size': 20})
plt.xlabel("Predicted probability")
plt.ylabel("# of data")
plt.savefig(f'bias_{r1:.1f}_{r2:.1f}_{r3:.1f}.pdf', dpi=300, bbox_inches='tight')
plt.show()

