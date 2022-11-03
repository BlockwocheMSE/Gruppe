# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:57:29 2022



@author: Jeroen
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem



#dimentions [m]
l = 6       #length of garage
breite = 3       #width of garage
hight = 2.5     #hight of garage
Sc = l * breite + 2 * l * hight + breite * hight
Sdoor = breite * hight
Sroof = l * breite
Va = l * breite * hight



air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
# pd.DataFrame.from_dict(air, orient='index', columns=['air'])
pd.DataFrame(air, index=['Air'])



wall = {'Conductivity': [1.4, 0.16, 0.16],  # W/(m·K)
        'Density': [2300, 600, 600],        # kg/m³
        'Specific heat': [880, 1760, 1760], # J/(kg·K)
        'Width': [0.25, 0.05, 0.1],         # m
        'Surface': [Sc, Sdoor, Sroof],      # m²
        'Slices': [1, 1, 1]}                # number of  slices
wall = pd.DataFrame(wall, index=['Concrete', 'Wood Door', 'Wood Roof'])
wall



#ε_c = 0.85    # long wave emmisivity: concrete
#ε_w = 0.95    # long wave emmisivity: wood
α_c = 0.25    # short wave absortivity: concrete
α_w = 0.4    # short wave absortivity: wood



σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant



h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
h



# Conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
pd.DataFrame(G_cd, columns={'Conductance'})



# Convection
Gw = h * wall['Surface'][0]     # wall
Gdoor = h * wall['Surface'][1]  # wood door
Groof = h * wall['Surface'][2]  # wood roof



C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']



C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns={'Capacity'})



A = np.zeros([12, 10])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 9] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3] = 1                 
A[5, 3], A[5, 4] = -1, 1    
A[6, 4], A[6, 5] = -1, 1    
A[7, 5], A[7, 9] = -1, 1    
A[8, 6] = 1                 
A[9, 6], A[9, 7] = -1, 1    
A[10, 7], A[10, 8] = -1, 1    
A[11, 8], A[11, 9] = -1, 1    
np.set_printoptions(suppress=False)
pd.DataFrame(A)



G = np.diag([Groof.iloc[0]['out'], 2 * G_cd['Wood Roof'], 2 * G_cd['Wood Roof'],
             Groof.iloc[0]['in'], Gdoor.iloc[0]['out'], 2 * G_cd['Wood Door'],
             2 * G_cd['Wood Door'], Gdoor.iloc[0]['in'],Gw.iloc[0]['out'],
             2 * G_cd['Concrete'], 2 * G_cd['Concrete'], Gw.iloc[0]['in']])
np.set_printoptions(precision=3, threshold=16, suppress=True)
pd.set_option("display.precision", 1)
pd.DataFrame(G)



C = np.diag([0, C['Wood Roof'], 0, 0, C['Wood Door'], 0, 0,
            C['Concrete'], 0, C['Air']])
# Uncomment next line to put 'Air' and 'Glass' capacities to zero
# C = np.diag([0, C['Wood Roof'], 0, 0, C['Wood Door'], 0, 0, C['Concrete'], 0, 0])
pd.set_option("display.precision", 3)
pd.DataFrame(C)



# C = np.zeros([10, 10])



b = np.zeros(12)        # branches
b[[0, 4, 8]] = 1   # branches with temperature sources
#print(f'b = ', b)



f = np.zeros(10)        # nodes
f[[0, 3, 6, 9]] = 1     # nodes with heat-flow sources
#print(f'f = ', f)



y = np.zeros(10)         # nodes
y[[9]] = 1              # nodes (temperatures) of interest
#print(f'y = ', y)



'''state space representation'''
[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)
print('As = \n', As, '\n')
print('Bs = \n', Bs, '\n')
print('Cs = \n', Cs, '\n')
print('Ds = \n', Ds, '\n')



'''steady state'''
b = np.zeros(12)        # temperature sources
b[[0, 4, 8]] = 10      # outdoor temperature
#b[[11]] = 20            # indoor set-point temperature



f = np.zeros(10)         # flow-rate sources



'''DAE'''



θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print(f'θ = {θ} °C')



'''state space'''



bT = np.array([10, 10, 10])     # [To, To, To, Tisp]
fQ = np.array([0, 0, 0, 0])         # [Φo, Φi, Qa, Φa]
u = np.hstack([bT, fQ])
print(f'u = {u}')



yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
print(f'yss = {yss} °C')



print(f'Max error between DAE and state-space: \
{max(abs(θ[6] - yss)):.2e} °C')



'''Dynamic'''
λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
dtmax = min(-2. / λ)
print(f'Maximum time step: {dtmax:.2f} s = {dtmax / 60:.2f} min')



dt = 4 * 60     # seconds
print(f'dt = {dt} s = {dt / 60:.0f} min')



t_resp = 4 * max(-1 / λ)
print('Time constants: \n', -1 / λ, 's \n')
print(f'Settling time: {t_resp:.0f} s = {t_resp / 60:.1f} min \
= {t_resp / (3600):.2f} h = {t_resp / (3600 * 24):.2f} days')



duration = 3600 * 24 * 5            # seconds, larger than response time
n = int(np.floor(duration / dt))    # number of time steps
t = np.arange(0, n * dt, dt)        # time vector for n time steps



print(f'Duration = {duration} s')
print(f'Number of time steps = {n}')
pd.DataFrame(t, columns=['time'])





u = np.zeros([7, n])                # u = [To To To Tisp Φo Φi Qa Φa]
u[0:3, :] = 10 * np.ones([3, n])    # To = 10 for n time steps
#u[3, :] = 20 * np.ones([1, n])      # Tisp = 20 for n time steps
print('u = ')
pd.DataFrame(u)



n_s = As.shape[0]                      # number of state variables
θ_exp = np.zeros([n_s, t.shape[0]])    # explicit Euler in time t
θ_imp = np.zeros([n_s, t.shape[0]])    # implicit Euler in time t



I = np.eye(n_s)                        # identity matrix



for k in range(n - 1):
    θ_exp[:, k + 1] = (I + dt * As) @\
        θ_exp[:, k] + dt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (θ_imp[:, k] + dt * Bs @ u[:, k])
        
y_exp = Cs @ θ_exp + Ds @  u
y_imp = Cs @ θ_imp + Ds @  u



fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.T, t / 3600, y_imp.T)
ax.set(xlabel='Time [h]',
       ylabel='$T_i$ [°C]',
       title='Step input: To')
ax.legend(['Implicit', 'Explicit'])
plt.show()



print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θ[9]):.4f} °C')
print(f'- state-space model: {float(yss):.4f} °C')
print(f'- steady-state response to step input: {float(y_exp[:, -2]):.4f} °C')



'''weather data'''



start_date = '2000-01-03 12:00:00'
end_date = '2000-02-05 18:00:00'



print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')



filename = './weather_data/DEU_Stuttgart.107380_IWEC.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data



weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather[(
    weather.index >= start_date) & (
    weather.index < end_date)]



pd.DataFrame(weather)



#for the different sides of the building, different radiations could be calculated



surface_orientation = {'slope': 90,
                       'azimuth': 0,
                       'latitude': 45}
albedo = 0.2
rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, albedo)
pd.DataFrame(rad_surf)



rad_surf['Etot'] = rad_surf.sum(axis=1)



data = pd.concat([weather['temp_air'], rad_surf['Etot']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'To'})
pd.DataFrame(data)



#data['Ti'] = 20 * np.ones(data.shape[0])
data['Qa'] = 0 * np.ones(data.shape[0])
pd.DataFrame(data)



To = data['To']
#Ti = data['Ti']
Φo = α_c * wall['Surface']['Concrete'] * data['Etot']
Φi = α_w * wall['Surface']['Wood Door'] * data['Etot']
#Φi = τ_gSW * α_wSW * wall['Surface']['Glass'] * data['Etot']
Qa = data['Qa']
Φa = α_w * wall['Surface']['Wood Roof'] * data['Etot']



u = pd.concat([To, To, To, Φo, Φi, Qa, Φa], axis=1)
u.columns.values[[3, 4, 6]] = ['Φo', 'Φi', 'Φa']
pd.DataFrame(u)



θ_exp = 7.5 * np.ones([As.shape[0], u.shape[0]])    #7.5 as inital outside temp



for k in range(u.shape[0] - 1):
    θ_exp[:, k + 1] = (I + dt * As) @ θ_exp[:, k]\
        + dt * Bs @ u.iloc[k, :]
        
y_exp = Cs @ θ_exp + Ds @ u.to_numpy().T
#q_HVAC = Kp * (data['Ti'] - y_exp[0, :])



t = dt * np.arange(data.shape[0])   # time vector



fig, axs = plt.subplots(2, 1)
# plot indoor and outdoor temperature
axs[0].plot(t / 3600 / 24, y_exp[0, :], label='$T_{indoor}$')
axs[0].plot(t / 3600 / 24, data['To'], label='$T_{outdoor}$')
axs[0].set(xlabel='Time [days]',
           ylabel='Temperatures [°C]',
           title='Simulation for weather')
axs[0].legend(loc='upper right')
#the inside temperatur is greatly based on dthe absorbtivity of the wood



# plot total solar radiation and HVAC heat flow
#axs[1].plot(t / 3600 / 24,  q_HVAC, label='$q_{HVAC}$')
axs[1].plot(t / 3600 / 24, data['Etot'], label='$Φ_{total}$')
axs[1].set(xlabel='Time [days]',
           ylabel='Heat flows [W]')
axs[1].legend(loc='upper right')



fig.tight_layout()