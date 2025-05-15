import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

controller = False
neglect_air_glass_capacity = False
imposed_time_step = False
Δt = 498    # s, imposed time step

l = 5               #longueur
L = 4               #largeur de la pièce
h = 3               #hauteur de la piece
Sfenetre = 0.750*0.600
Sporte = 2.04*0.7
Stot = l*L*h
Smur = l*L*h - Sfenetre - Sporte        #surface mur ext
hO = 25
hi = 7.7

lambdafenetre = 1.4

lambdaiso = 0.046
lambdabeton = 1.4

lambdaporte = 0.15 #porte en bois

densite_beton = 2300.0
densite_iso = 24.0

specific_heat_beton = 880
specific_heat_iso =1000

epaisseur_beton = 0.25
epaisseur_iso = 0.15

densite_air = 1.2
specific_heat_air = 1000



# ventilation flow rate
Va = l**3                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5']
# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']
Kp = 1e5

Cmur = densite_beton * specific_heat_beton * epaisseur_beton * Smur
Ciso = densite_iso * specific_heat_iso * epaisseur_iso * Smur
Cair = densite_air * specific_heat_air * Va



Cvalues = np.zeros(6)

Cvalues[1] = Cmur
Cvalues[3] = Ciso
Cvalues[5] = Cair


C = np.diag(Cvalues)

print("C =")
print(C)





# temperature nodes
nθ = 6     # number of temperature nodes
θ = [f'θ{i}' for i in range(6)]

# flow-rate branches
nq = 10     # number of flow branches
q = [f'q{i}' for i in range(10)]

A = np.zeros([10, 6])       # n° of branches X n° of nodes
A[0, 0] = 1                
A[1, 0], A[1, 1] = -1, 1    
A[2, 1], A[2, 2] = -1, 1   
A[3, 2], A[3, 3] = -1, 1    
A[4, 3], A[4, 4] = -1, 1   
A[5,4] = -1
A[5, 5] = 1    
A[6, 5] = 1   
A[7, 5] =1    
A[8, 5] = 1                 
A[9, 5]= 1    
print("A=")
print(A)


G0 = hO*Stot

G1 = (lambdabeton/0.125)*Smur
G2 = (lambdabeton/0.125)*Smur

G3 = (lambdaiso/0.075)*Smur
G4 = (lambdaiso/0.075)*Smur

G5 =  hi*Stot

G6 =(lambdafenetre/0.08)*Sfenetre

G7 = (lambdaporte/0.4)*Sporte

G8 = densite_air * specific_heat_air * Va_dot

G9 = Kp


Gvalues = np.array(np.hstack([G0, G1, G2, G3, G4, G5, G6, G7, G8, G9]))

G = np.diag(Gvalues)

print("G =")
print(G)





Qa = 1200

ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

E = 50 #total irradiance receivend by the wall or window

Φo = α_wSW*E*Smur
Φi = τ_gSW*α_gSW*E*Smur


f = [Φo,0,0,0,Φo,Φi] # flow rates


T0 = 10
Tisp = 20

# Input vectors
b = np.zeros(10)  # temperatures

b[0] = T0
b[6] = T0
b[7] = T0
b[8] = T0
b[9] = Tisp

print("b =")
print(b)

print("f =")
print(f)

y = [0,0,0,0,0,0]         # nodes
y[5] = 1              # nodes (temperatures) of interest

print("y =")
print(y)


θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)


print(f'θ = {np.around(θ, 2)} °C')
print(f'The indoor temperature is: {θ[-1]:.3f} °C')
print("---------------------------------------------")
# temperature nodes
nθ = 6     # number of temperature nodes
θ = [f'θ{i}' for i in range(6)]

# flow-rate branches
nq = 10     # number of flow branches
q = [f'q{i}' for i in range(10)]

A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(Gvalues, index=q)
C = pd.Series(Cvalues, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

#Steady-state from differential algebraic equations (DAE)


[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)


print("As =\n", As)
print("\nBs =\n", Bs)
print("\nCs =\n", Cs)
print("\nDs =\n", Ds)
print("\nus =\n", us)

# Input vectors
bss = np.zeros(10)  # temperatures

bss[0] = T0
bss[6] = T0
bss[7] = T0
bss[8] = T0
bss[9] = Tisp

fss = np.zeros(6)         # flow-rate sources f for steady state


A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)


print(f'θss = {np.around(θss, 2)} °C')

#Steady state (DAE) avec flux

bssQ = np.zeros(10)  

bssQ[0] = T0
bssQ[6] = T0
bssQ[7] = T0
bssQ[8] = T0
bssQ[9] = Tisp       # temperature sources b for steady state

fssQ = [Φo,0,0,0,Φo,Φi]       # flow-rate sources f for steady state


θssQ = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bssQ + fssQ)
print(f'θssQ = {np.around(θssQ, 2)} °C')


#Steady-state from state-space representation

bT = np.array([T0, T0, T0, T0, Tisp])     # [To, To, To, T0, Tisp]
fQ = np.array([0, 0, 0])         # [Φo, Φi, Qa]
uss = np.hstack([bT, fQ])           # input vector for state space
print(f'uss = {uss}')


inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yss = (-Cs @ inv_As @ Bs + Ds) @ uss

yss = float(yss.values[0])
print(f'yss = {yss:.2f} °C')

print(f'Error between DAE and state-space: {abs(θss[5] - yss):.2e} °C')

bTQ = np.array([T0, T0, T0, T0, Tisp])     # [To, To, To, T0, Tisp]
fQQ = np.array([0, 0, 0])         # [Φo, Φi, Qa]
ussQ = np.hstack([bTQ, fQQ])           # input vector for state space
print(f'ussQ = {ussQ}')



yssQ = (-Cs @ inv_As @ Bs + Ds) @ ussQ

yssQ = float(yssQ.values[0])
print(f'yssQ = {yssQ:.2f} °C')

print(f'Error between DAE and state-space: {abs(θssQ[5] - yssQ):.2e} °C')

print("---------------------------------------------")


#   PARTIE 2 : Simulate step response

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As

print(λ)

# time step
Δtmax = 2 * min(-1. / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)
dm4bem.print_rounded_time('dt', dt)

print(dt)

"""
if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")

"""


# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)



# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}s")

To = 10 * np.ones(n)        # outdoor temperature
Ti_sp = 20 * np.ones(n)     # indoor temperature set point
Φa = 0 * np.ones(n)         # solar radiation absorbed by the glass
Qa = Φo = Φi = Φa           # auxiliary heat sources and solar radiation

data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
#u = dm4bem.inputs_in_time(us, input_data_set)

