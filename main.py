import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem



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

C0 = Cmur/3
C1 = Cmur/3
C2 = Cmur/3
C3 = Ciso/2
C4 = Ciso/2
C5 = Cair

C_values = [C0, C1, C2, C3, C4, C5]
C = np.diag(C_values)
pd.DataFrame(C, index=θ)
np.set_printoptions(precision=2, suppress=True)
print("C =")
print(C)





# temperature nodes
nθ = 6     # number of temperature nodes
θ = [f'θ{i}' for i in range(6)]

# flow-rate branches
nq = 10     # number of flow branches
q = [f'q{i}' for i in range(10)]

A = np.zeros([10, 6])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1   # branch 4: node 3 -> node 4
A[5,4] = -1
A[5, 5] = 1    # branch 5: node 4 -> node 5
A[6, 5] = 1    # branch 6: node 4 -> node 6
A[7, 5] =1    # branch 7: node 5 -> node 6
A[8, 5] = 1                 # branch 8: -> node 7
A[9, 5]= 1    # branch 9: node 5 -> node 7
print("A=")
print(A)
pd.DataFrame(A, index=q, columns=θ)

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



G_values = [G0, G1, G2, G3, G4, G5, G6, G7, G8, G9]
G = np.diag(G_values)

np.set_printoptions(precision=2, suppress=True)
pd.DataFrame(G, index=q)
print("G =")
print(G)

# Input vectors
b = np.zeros(10)  # temperatures
f = np.zeros(6)  # flow rates

f = pd.Series([0, 0, 0, 0, 0, 0],
              index=θ)

T0 = 10
Tisp = 20

b[0] = T0
b[6] = T0
b[7] = T0
b[8] = T0
b[9] = Tisp

b = pd.Series(['To', 0, 0, 0, 0, 0,'To' , 'To', 'To', 'Tisp'],
              index=q)

y = np.zeros(6)         # nodes
y[[5]] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)



θ_steady_To = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
np.set_printoptions(precision=3)
print('When To = 1°C, the temperatures in steady-state are:', θ_steady_To, '°C')
print(f'The indoor temperature is: {θ_steady_To[-1]:.3f} °C')
print("---------------------------------------------")

bss = np.zeros(10)        # temperature sources b for steady state
bss[[0, 6,7,8]] = T0      # outdoor temperature
bss[[9]] = Tisp            # indoor set-point temperature

fss = np.zeros(6)         # flow-rate sources f for steady state
fss[[5]] = 1000

θssQ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ bss + fss)
print(f'θssQ = {np.around(θssQ, 2)} °C') #temperature en steady state par DAE avec flux Qa = 1000W

"""
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}


"""

































print("---------------------------------------------")
"""
# State matrix
As = -np.linalg.inv(C) @ A.T @ G @ A
# pd.set_option('precision', 1)
pd.DataFrame(As, index=θ, columns=θ)
print("As=",As)
print("---------------------------------------------")

# Input matrix
Bs = np.linalg.inv(C) @ np.block([A.T @ G, np.eye(nθ)])
pd.DataFrame(Bs, index=θ, columns=q + θ)
Bs = Bs[:, [0, -1]]
pd.DataFrame(Bs, columns=['To', 'Qh'])
np.set_printoptions(precision=8)  
print("Bs=",Bs)
print("---------------------------------------------")

# Output matrix
Cs = np.zeros((1, nθ))
# output: last temperature node
Cs[:, -1] = 1
print("Cs=",Cs)
print("---------------------------------------------")

# Feedthrough (or feedforward) matrix
Ds = np.zeros(Bs.shape[1])
print("Ds=",Ds)
print("---------------------------------------------")

bT = np.array([T0, T0, T0, T0, Tisp])     # [To, To, To,To, Tisp]
fQ = np.array([0, 0, 0])         # [Φo, Φi, Qa]
uss = np.hstack([bT, fQ])           # input vector for state space
print(f'uss = {uss}')

inv_As = pd.DataFrame(np.linalg.inv(As))
yss = (-Cs @ inv_As @ Bs + Ds) @ uss

yss = float(yss.values[0])
print(f'yss = {yss:.2f} °C')

print("---------------------------------------------")"""




#   PARTIE 2 : Simulate step response




