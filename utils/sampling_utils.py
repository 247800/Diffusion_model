import torch
import torchaudio

def get_time_schedule(sigma_min=1e-5, sigma_max=12, T=50, rho=10):
    i = torch.arange(0, T + 1)
    sigma_i = (sigma_max ** (1/rho) + i * (sigma_min ** (1/rho) - sigma_max ** (1/rho)) / (T - 1)) ** rho
    sigma_i[T] = 0
    return sigma_i

def get_noise(t, S_tmin=1e-5, S_tmax=12, S_churn=None, idx=0):
    if S_churn is None:
        S_churn = t[idx:idx+1]
    if S_churn < S_tmin:
        gamma_i = 0
    elif S_churn > S_tmax:
        gamma_i = 0
    else:
        N = torch.randn(1)
        gamma_i = min(S_churn / N, torch.sqrt(1) - 1)
    return gamma_i
