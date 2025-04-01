# %% Modules

import numpy as np
import matplotlib.pyplot as plt

# %% Param

Lx = 3
N  = 300
dx = Lx/N
t  = 1
dt = 0.005
CFL = dt/dx

t_step = int(t/dt)
xg  = np.linspace(0, Lx, N, endpoint = False)

# %% Lax-Friedrichs

def LF(CFL):

    u_time = np.sin(2 * np.pi * xg)  # t = 0, first time layer
    u_next = np.zeros_like(u_time)  
    
    for i in range(t_step):
        for j in range(1, N-1):
            u_next[j] = 0.5 * (1 - CFL) * u_time[j+1] + 0.5 * (1 + CFL) * u_time[j-1]
        
        u_next[0]   = 0.5 * (1 - CFL) * u_time[1] + 0.5 * (1 + CFL) * u_time[-1]      
        u_next[-1]  = 0.5 * (1 - CFL) * u_time[0] + 0.5 * (1 + CFL) * u_time[-2]
        
        u_time = u_next.copy() 
    
    return u_time
       

# %% First-order upwind scheme

def Fous(CFL):
    u_time = np.sin(2 * np.pi * xg)
    u_next = np.zeros_like(u_time)
    
    for i in range(t_step):
        for j in range(N):
            
            u_next[j] = u_time[j] - CFL * (u_time[j] - u_time[j-1])
            
        u_time = u_next.copy()
        
    return u_time

# %% Second order upwind scheme

def Sous(CFL):
    u_time = np.sin(2 * np.pi * xg)
    u_next = np.zeros_like(u_time)
    
    for i in range(t_step):
        for j in range(2, N):
            
            u_next[j] = u_time[j] - 0.5 * CFL * (3 * u_time[j] - 4 * u_time[j-1] + u_time[j-2]) 
            
        u_next[1] = u_time[1] - 0.5 * CFL * (3 * u_time[1] - 4 * u_time[0] + u_time[-1]) 
        u_next[0] = u_time[0] - 0.5 * CFL * (3 * u_time[0] - 4 * u_time[-1] + u_time[-2])
        
        u_time = u_next.copy()
         
    return u_time   

# %% Graphing

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

u1 = LF(CFL)
ax1.plot(xg, u1, 'b-', label='Lax-Friedrichs')
ax1.set_title('Lax-Friedrichs Method')
ax1.legend()

u2 = Fous(CFL)
ax2.plot(xg, u2, 'r-', label='First-order Upwind')
ax2.set_title('First-order Upwind Method')
ax2.legend()

u3 = Sous(CFL)
ax3.plot(xg, u3, color = 'black', linestyle = '-', label='Second-order Upwind')
ax3.set_title('Second-order Upwind Method')
ax3.legend()

plt.tight_layout()
plt.show()

print(f'CFL = {CFL}')


