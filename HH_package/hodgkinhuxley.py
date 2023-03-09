import numpy as np

class HodgkinHuxley:
    """Full Hodgkin-Huxley Model implemented in Python"""
    def __init__(self, dt=0.05, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387):
        self.dt = dt  # default 0.01
        self.C_m = C_m  # default 1.0.
        self.g_Na = g_Na  # default 120.0.
        self.g_K = g_K  # default 36.0.
        self.g_L = g_L  # default 0.3.
        self.E_Na = E_Na  # default 50.0.
        self.E_K = E_K  # default -77.0.
        self.E_L = E_L  # default -54.387.

        self.V = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32

    alpha_m = lambda self: 0.1 * (self.V + 40.0) / (1.0 - np.exp(-(self.V + 40.0) / 10.0))
    beta_m = lambda self: 4.0 * np.exp(-(self.V + 65.0) / 18.0)

    alpha_h = lambda self: 0.07 * np.exp(-(self.V + 65.0) / 20.0)
    beta_h = lambda self: 1.0 / (1.0 + np.exp(-(self.V + 35.0) / 10.0))

    alpha_n = lambda self: 0.01 * (self.V + 55.0) / (1.0 - np.exp(-(self.V + 55.0) / 10.0))
    beta_n = lambda self: 0.125 * np.exp(-(self.V + 65) / 80.0)

    I_Na = lambda self: self.g_Na * self.m ** 3 * self.h * (self.V - self.E_Na)
    I_K = lambda self: self.g_K * self.n ** 4 * (self.V - self.E_K)
    I_L = lambda self: self.g_L * (self.V - self.E_L)

    def step(self, I_inj=0):
        """
         1step実行
         Args: I_inj:外部からの入力電流

         Returns: チャネル変数:m,h,n 膜電位:V
         """
        self.m += (self.alpha_m() * (1.0 - self.m) - self.beta_m() * self.m) * self.dt  # m <- m + dm
        self.h += (self.alpha_h() * (1.0 - self.h) - self.beta_h() * self.h) * self.dt  # h <- h + dh
        self.n += (self.alpha_n() * (1.0 - self.n) - self.beta_n() * self.n) * self.dt  # n <- n + dn
        self.V += ((I_inj - self.I_Na() - self.I_K() - self.I_L()) / self.C_m) * self.dt  # V <- V + dV
        return self.m, self.h, self.n, self.V