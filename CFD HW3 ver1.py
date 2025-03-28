# %% Modules

import numpy as np
import time
from matplotlib.pyplot import matplotlib

# %% Param

Dx = 1e-6
Nx = np.arange(0, 3, Dx)
CFL = 0.5

# %% Lax-Friedrichs

def LF(u, CFL):
    u_new = [0 for i in range(int(3/Dx))]
    for j, dx in enumerate(Nx):
        u_new[j+1] = 0.5*(1 - CFL)*u[j+1] + 0.5*(1 + CFL)*u[j-1]
    return u_new

# %% Second order upwind scheme

def Sous(u, CFL):
    u_new = [0 for i in range(int(3/Dx))]
    for j, dx in enumerate(Nx):
        u_new[j+1] = u[j] - 0.5 * CFL * (3 * u[j] - 4 * u[j-1] + u[j - 2])
    return u_new

# %% First order upwind scheme

def Fous(u, CFL):
    u_new = [0 for i in range(int(3/Dx))]
    for j, dx in enumerate(Nx):
        u_new[j+1] = u[j] - CFL * (u[j] - u[j-1])
    return u_new

# %%


