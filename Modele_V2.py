import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.25,                   # m
            'Surface': Smur}            # m²

insulation = {'Conductivity': 0.046,        # W/(m·K)
              'Density': 24.0,              # kg/m³
              'Specific heat': 1000,        # J/(kg⋅K)
              'Width': 0.15,                # m
              'Surface': Smur}          # m²

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.08,                     # m
         'Surface': Sfenetre}                   # m²

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')
wall




# ventilation flow rate
Va = l**3                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration


Kp = 0

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

np.set_printoptions(precision=2, suppress=True)
print("C =")
print(C)

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']

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
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
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

G7 = Kp

G8 = air['Density'] * air['Specific heat'] * Va_dot

G9 =(lambdaporte/0.4)*Sporte



G_values = [G0, G1, G2, G3, G4, G5, G6, G7, G8, G9]
G = np.diag(G_values)

np.set_printoptions(precision=2, suppress=True)
print("G =")
print(G)

# Input vectors
b = np.zeros(10)  # temperatures
f = np.zeros(6)  # flow rates


b[0] = 1

θ_steady_To = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)




np.set_printoptions(precision=3)
print('When To = 1°C, the temperatures in steady-state are:', θ_steady_To, '°C')
print(f'The indoor temperature is: {θ_steady_To[-1]:.3f} °C')
