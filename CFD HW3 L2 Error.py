# %% Modules
import numpy as np
import matplotlib.pyplot as plt

# %% Parameters

Lx = 3    
t = 1
Ns = [100, 300, 600, 900]

CFLs = {"LF": 0.8, "Fous": 0.8, "Sous": 0.4}

# %% Lax-Friedrichs

def LF(CFL, N, xg, t_step):

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

def Fous(CFL, N, xg, t_step):
    u_time = np.sin(2 * np.pi * xg)
    u_next = np.zeros_like(u_time)
    
    for i in range(t_step):
        for j in range(1, N):
            
            u_next[j] = u_time[j] - CFL * (u_time[j] - u_time[j-1])
        
        u_next[0] = u_time[0] - CFL * ((u_time[0] - u_time[-1]))
        
        u_time = u_next.copy()
        
    return u_time

# %% Second-order upwind scheme

def Sous(CFL, N, xg, t_step):
    u_time = np.sin(2 * np.pi * xg)
    u_next = np.zeros_like(u_time)
    
    for i in range(t_step):
        for j in range(2, N):
            
            u_next[j] = u_time[j] - 0.5 * CFL * (3 * u_time[j] - 4 * u_time[j-1] + u_time[j-2]) 
            
        u_next[1] = u_time[1] - 0.5 * CFL * (3 * u_time[1] - 4 * u_time[0] + u_time[-1]) 
        u_next[0] = u_time[0] - 0.5 * CFL * (3 * u_time[0] - 4 * u_time[-1] + u_time[-2])
        
        u_time = u_next.copy()
         
    return u_time   

# %% Error Calculation
def cal_errors(method, Ns):
    errors = []
    for N in Ns:
        dx = Lx / N
        CFL = CFLs[method]
        dt = CFL * dx
        t_step = int(t/dt)
        xg = np.linspace(0, Lx, N, endpoint=False)
        
        if method == "LF":
            u_num = LF(CFL, N, xg, t_step)
        elif method == "Fous":
            u_num = Fous(CFL, N, xg, t_step)
        elif method == "Sous":
            u_num = Sous(CFL, N, xg, t_step)
        
        u_exact = np.sin(2 * np.pi * (xg - t))  
        error = np.sqrt(np.mean((u_num - u_exact)**2)) #L2 Error
        errors.append(error)
    
    return errors

# %% Graphing

error_data = {}
methods = ["LF", "Fous", "Sous"]

for method in methods:
    errors = cal_errors(method, Ns)
    error_data[method] = (Ns, errors)

plt.figure(figsize=(10, 7))
dx_list = [Lx / N for N in Ns]

for method in methods:
    Ns, errors = error_data[method]
    plt.loglog(dx_list, errors, 'o-', label=method)

plt.loglog(dx_list, [dx**1 for dx in dx_list], 'k--', label='O(Δx)')
plt.loglog(dx_list, [dx**2 for dx in dx_list], 'k:', label='O(Δx²)')

plt.xlabel("Δx (Grid Size)")
plt.ylabel("L2 Error")
plt.legend()
plt.title("Grid Convergence Test ")
plt.show()


