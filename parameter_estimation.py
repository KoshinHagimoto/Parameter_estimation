import numpy as np
import matplotlib.pyplot as plt

from HH_package.hodgkinhuxley import HodgkinHuxley
from Pf_package.particle_filter import ParticleFilter

"""
global variable
"""
MAX_EPOCHS = 100  # 最大試行回数
SIGMA = 0.5 # 観測ノイズの標準偏差
steps = 1000
# 外部電流
I_inj = np.zeros(steps)
I_inj[:] = 20

def generate_data():
    """
    シミュレーションデータをHHモデルを用いて作成.
    :return V_train(大きさstepsのndarray型)
    """
    HH = HodgkinHuxley()

    m_test = np.zeros(steps)
    h_test = np.zeros(steps)
    n_test = np.zeros(steps)
    V_test = np.zeros(steps)
    m_test[0] = 0.05
    h_test[0] = 0.6
    n_test[0] = 0.32
    V_test[0] = -65.0

    for i in range(steps - 1):
        result = HH.step(I_inj[i])
        m_test[i+1] = result[0]
        h_test[i+1] = result[1]
        n_test[i+1] = result[2]
        V_test[i+1] = result[3]

    noise_v = np.random.normal(0, 0.5, (steps - 1,))
    noise_v = np.insert(noise_v, 0, 0)
    V_train = V_test + noise_v
    return V_train

def gL_estimation(V, m, h, n, I_inj, dt=0.05, g_Na=120.0, g_K=36.0, E_Na=50.0, E_K=-77.0, E_L=-54.387):
    """
    粒子フィルタで推定したV,m,h,nを用いて, gLを推定する関数
    :param V: 推定した膜電位
    :param m: 推定したチャネル変数
    :param h: 推定したチャネル変数
    :param n: 推定したチャネル変数
    :param I_inj:外部電流
    :param dt: 学習率
    :param g_Na: コンダクタンス
    :param g_K: コンダクタンス
    :param E_Na: 平衡電位
    :param E_K: 平衡電位
    :param E_L: 平衡電位
    :return: gL: 推定したコンダクタンス
    """
    gL = np.sum(((V[1:]-V[:-1])/dt + g_Na*m[1:]**3*h[1:]*(V[:-1]-E_Na) + g_K*n[1:]**4*(V[:-1]-E_K) - I_inj[:-1]) * (E_L - V[:-1])) /np.sum((V[:-1]-E_L)**2)
    return gL # g_L = 0.3

def estimate_parameter(V_train):
    gLs = []
    gL_initial = 0.40
    gLs.append(gL_initial)

    for i in range(MAX_EPOCHS):
        pf = ParticleFilter(V_train, g_L=gL_initial)
        pf.simulate()
        V_particle_train = pf.V_average
        m_particle_train = pf.m_average
        h_particle_train = pf.h_average
        n_particle_train = pf.n_average
        gL_estimated = gL_estimation(V_particle_train, m_particle_train, h_particle_train, n_particle_train, I_inj)
        gLs.append(gL_estimated)
        gL_initial = gL_estimated
    return gLs

def show_graph(gLs):
    gL_line = np.zeros(MAX_EPOCHS + 1)
    gL_line[:] = 0.30
    plt.plot(range(MAX_EPOCHS + 1), gL_line, linewidth=3, linestyle="dashed", label='gL (true)')
    plt.plot(range(MAX_EPOCHS + 1), gLs, label='gL (proposed method)')
    plt.title('gL estimation')
    plt.ylim(0.25, 0.5)
    plt.xlabel('step')
    plt.ylabel('gL')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    V_train = generate_data()
    gLs = estimate_parameter(V_train)
    show_graph(gLs)

if __name__ == '__main__':
    main()
