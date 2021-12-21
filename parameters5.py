import carbon as carb
import numpy as np


# ========== #
# PARAMETERS
# ========== #


#Time and Demand
T =     0.5 # over 10 years
Nt =    365 # number of 10 days
t_grid = np.linspace(0,T,Nt,endpoint=True) # grid on [0,T]
Delta_t = t_grid[1]-t_grid[0]

Dt= 20000. - 500. * np.cos(80*np.pi*t_grid) 

#Production
kappa1 = 0.13
kappa2 = 0.1
theta = 5.0
alpha = 40*np.pi
sig0 = 0.01

#Emission
delta = 0.15
sig1 = 0.01 


#Prices
p1 = 7.0 / Delta_t
p2 = 10000.
p3 =  0.00000001
Rmax = 5000.

c1 = 0.0001 
c3 = 1.0
rho_0 = 40.0 / Delta_t
rho_1 = 0.1 / Delta_t


XBar0 = np.array([[0.0],
                  [theta],
                  [0.0],
                  [0.0],
                  [0.0]])

XBarFixed =  np.repeat(0.1, 5*Nt)

XVar0 = np.array([[0., 0., 0., 0., 0.],
                  [0., 0.01, 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.], 
                  [0., 0., 0., 0., 0.]])                        

XBarBefore = np.zeros(5*Nt)
iteration = 25

damp = 1.0
epsilon = 1.e-2
tol = 0.001

sunk_cost = 10000000000000

alpha1 = 1.0
alpha2 = 1.0
alpha3 = 0.01
alpha4 = 10.0
alpha5 = 0.0001
PBar_star  = 0.0
tau_star = 0.0
RENLow=0.
RENHigh=Rmax
