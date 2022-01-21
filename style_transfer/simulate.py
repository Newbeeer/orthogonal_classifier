import numpy as np

N = 100
M = 100
QM = 100
p1 = 0.95
p2 = 0.95
q1 = 0.05
q2 = 0.05
seed = 1234
def oc(P_1=p1, P_2=p2, Q_1=q1, Q_2=q2, n=N, m=M, qm=QM):
    np.random.seed(seed)
    eps_p = np.random.randn(100000) / np.sqrt(m) * np.sqrt(P_1 * P_2 * (1-P_1 * P_2)) *0
    eps_q = np.random.randn(100000) / np.sqrt(qm) * np.sqrt(Q_1 * Q_2 * (1-Q_1 * Q_2) ) *0
    eps_p_ = np.random.randn(100000) / np.sqrt(n) * np.sqrt(P_1 * (1-P_1)) *0
    eps_q_ = np.random.randn(100000) / np.sqrt(n) * np.sqrt(Q_1 * (1-Q_1)) *0
    eps = np.random.randn(100000) / np.sqrt(n) *0
    numerator = (P_1 * P_2 + eps_p) / (Q_1 * Q_2 + eps_q)
    denumerator = (P_1 + eps_p_) / (Q_1 + eps_q_)
    #denumerator = (P_1/Q_1 + eps)
    return np.clip(numerator / (numerator + denumerator),1e-3,1-1e-3)

def re(P_1=p1, P_2=p2, Q_1=q1, Q_2=q2,  n=N, m=M, qm=QM):
    np.random.seed(seed)
    eps_1 = np.random.randn(100000) / np.sqrt(m) * np.sqrt(P_1 * P_2 * (1-P_1 * P_2)) *0
    eps_2 = np.random.randn(100000) / np.sqrt(qm) * np.sqrt(Q_1 * Q_2 * (1-Q_1 * Q_2)) *0
    eps_p_ = np.random.randn(100000) / np.sqrt(n) * np.sqrt(P_1 * (1-P_1)) *0
    eps_q_ = np.random.randn(100000) / np.sqrt(n) * np.sqrt(Q_1 * (1-Q_1)) *0
    eps = np.random.randn(100000) / np.sqrt(n) *0
    numerator = (Q_1 + eps_q_) * P_1 * P_2 / (P_1 + eps_p_) + (Q_1 + eps_q_)/(P_1 + eps_p_) * eps_1
    #numerator = (Q_1/P_1 + eps) * P_1 * P_2 + (Q_1/P_1 + eps) * eps_1
    denumerator = Q_1 * Q_2 + eps_2

    return np.clip(numerator / (numerator + denumerator), 1e-3,1-1e-3)

def gt(P_1=0, P_2=p2, Q_1=0, Q_2=q2):

    return P_2 / (P_2 + Q_2)

def loss(P_1=p1, P_2=p2, Q_1=q1, Q_2=q2):
    print("OC...")

    Q1 = [Q_1, 1-Q_1]
    P1 = [P_1, 1-P_1]
    Q2 = [Q_2, 1-Q_2]
    P2 = [P_2, 1-P_2]
    first = - np.log(oc(P1[0], P2[0], Q1[0], Q2[0])) * Q1[0] * P2[0] \
            - np.log(oc(P1[0], P2[1], Q1[0], Q2[1])) * Q1[0] * P2[1] \
            - np.log(oc(P1[1], P2[0], Q1[1], Q2[0])) * Q1[1] * P2[0] \
            - np.log(oc(P1[1], P2[1], Q1[1], Q2[1])) * Q1[1] * P2[1]
    #print(oc(P1[0], P2[1], Q1[0], Q2[1]), oc(P1[1], P2[1], Q1[1], Q2[1]), gt(1-p2,1-q2))
    # print(oc(P1[0],P2[0],Q1[0],Q2[0]), oc(P1[1], P2[0], Q1[1], Q2[0]), gt(p2, q2))
    second = - np.log(1-oc(P1[0],P2[0],Q1[0],Q2[0])) * Q1[0] * Q2[0] \
            - np.log(1-oc(P1[0], P2[1], Q1[0], Q2[1])) * Q1[0] * Q2[1] \
            - np.log(1-oc(P1[1], P2[0], Q1[1], Q2[0])) * Q1[1] * Q2[0] \
            - np.log(1-oc(P1[1], P2[1], Q1[1], Q2[1])) * Q1[1] * Q2[1]
    print("OC loss:", first.mean() + second.mean())

    first = - np.log(re(P1[0],P2[0],Q1[0],Q2[0])) * Q1[0] * P2[0] \
            - np.log(re(P1[0], P2[1], Q1[0], Q2[1])) * Q1[0] * P2[1] \
            - np.log(re(P1[1], P2[0], Q1[1], Q2[0])) * Q1[1] * P2[0] \
            - np.log(re(P1[1], P2[1], Q1[1], Q2[1])) * Q1[1] * P2[1]
    #print(re(P1[0], P2[1], Q1[0], Q2[1]), oc(P1[1], P2[1], Q1[1], Q2[1]), gt(1-p2,1-q2))
    # print(re(P1[0],P2[0],Q1[0],Q2[0]), oc(P1[1], P2[0], Q1[1], Q2[0]), gt(p2, q2))
    second = - np.log(1-re(P1[0],P2[0],Q1[0],Q2[0])) * Q1[0] * Q2[0] \
            - np.log(1-re(P1[0], P2[1], Q1[0], Q2[1])) * Q1[0] * Q2[1] \
            - np.log(1-re(P1[1], P2[0], Q1[1], Q2[0])) * Q1[1] * Q2[0] \
            - np.log(1-re(P1[1], P2[1], Q1[1], Q2[1])) * Q1[1] * Q2[1]
    print("RE loss:", first.mean() + second.mean())

    first = - np.log(gt(P1[0],P2[0],Q1[0],Q2[0])) * Q1[0] * P2[0] \
            - np.log(gt(P1[0], P2[1], Q1[0], Q2[1])) * Q1[0] * P2[1] \
            - np.log(gt(P1[1], P2[0], Q1[1], Q2[0])) * Q1[1] * P2[0] \
            - np.log(gt(P1[1], P2[1], Q1[1], Q2[1])) * Q1[1] * P2[1]

    second = - np.log(1-gt(P1[0],P2[0],Q1[0],Q2[0])) * Q1[0] * Q2[0] \
            - np.log(1-gt(P1[0], P2[1], Q1[0], Q2[1])) * Q1[0] * Q2[1] \
            - np.log(1-gt(P1[1], P2[0], Q1[1], Q2[0])) * Q1[1] * Q2[0] \
            - np.log(1-gt(P1[1], P2[1], Q1[1], Q2[1])) * Q1[1] * Q2[1]
    print(first, second)
    print("GT loss:", first.mean() + second.mean())
loss(p1,p2,q1,q2)

