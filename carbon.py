# ================================ #
# ========PACKAGE IMPORT========= #
# ================================ #
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela
import scipy.linalg as sla
import math
from tqdm import tqdm
import time
import numpy.linalg as la 
#import pandas as pd
import random

import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})

random.seed(7)

# ================================ #
# ========== PARAMETERS ========== #
# ================================ #
def Input_Prep_no_REN(kappa1, delta, c1, c2, c3, rho_0, rho_1, tau, p1, Dt, Nt):
    # Running cost
    H = np.reshape(np.array([[-(2*c2 + c3*rho_1)*Dt - c3*rho_0],
                  [np.zeros((Nt))],
                  [np.zeros((Nt))],
                  [np.zeros((Nt))],                             
                  [np.repeat(p1,Nt)]]),(5,Nt))
    F = np.array([[c3*rho_1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    G = np.array([[c2, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    GBar = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    R =     np.reshape(2*c1,(1,1))
    Rinv =  np.reshape(1.0/R, (1,1))
    J = c2*Dt*Dt
    # Terminal cost
    ST = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],                   
                   [0, 0, 0, tau, 0],
                   [0, 0, 0, 0, 0]])
    # Dynamics
    B = np.array([[kappa1],
                  [0],
                  [delta],
                  [0],
                  [1]])    
    return(B, H, F, G, GBar, R, Rinv, J, ST)


def Input_Prep_REN(kappa2, REN, alpha, t_grid, sig0, sig1, theta, Nt):   
    A = np.array([[0, -kappa2 * REN, 0, 0, 0],
                  [0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])
    C = np.reshape(np.array([[kappa2*REN*(alpha*np.cos(alpha*t_grid) + theta)],
                  [np.repeat(theta, Nt)],
                  [np.zeros(Nt)],
                  [np.zeros(Nt)],
                  [np.zeros(Nt)]]),(5,Nt))

    sigma = np.array([[kappa2 * REN * sig0, 0],
                      [sig0, 0],
                      [0, sig1],
                      [0, 0],
                      [0, 0]])
    a =     0.5*np.matmul(sigma, np.transpose(sigma))
    return(A, C, sigma, a)






# ==================================================== #
# ODE for 2nd Order Coefficient P (in paper eta) 
# ==================================================== #
def rate_ODE_P(P_t, t, A, B, G, Rinv):
    P_t = np.reshape(P_t, (5, 5))
    return np.reshape((- np.matmul(np.transpose(A),P_t) - np.matmul(P_t,A) + \
                       Rinv * np.matmul(np.matmul(P_t,np.matmul(B,np.transpose(B))),P_t)\
                      - 2* G),(5*5,))
    
def Solver_ODE_P(P_T, A, B, G, Rinv,T, t_grid): 
    s_grid = T-t_grid
    tSteps = np.size(s_grid)
    P = odeint(rate_ODE_P, np.reshape(P_T,(5*5,)), s_grid, args = (A, B, G, Rinv))
    P_matrix = np.reshape(P, (tSteps,5,5))
    return (np.flip(P_matrix, axis = 0))





# ==================================================== #
# ========= MEAN FIELD CONTROL PREPARATION =========== #
# ==================================================== #


# ============================================= #
# Linear System for 1st order coefficient rho and mean
# ============================================= #

# [mean, rho]^T = M * [mean, rho] + K

# ============================================= #
# Construction of M
# ============================================= #
def Construct_M(A, B, F, P, Rinv, Delta_t, Nt):
    #Submatrix (1,1)
    Sub11_1 = np.zeros((Nt, 5, 5))
    for n in range(1,Nt):
        Sub11_1[n,:,:] = Delta_t * (A - Rinv * np.matmul(np.matmul(B, np.transpose(B)), P[n-1,:,:]))    
    Sub11 = sparse.block_diag(Sub11_1, format ="csr")    
    Sub11_2 = np.ones(Nt*5)    
    row_start_11_2 = -5
    Sub11 = Sub11 + sparse.spdiags(Sub11_2,[row_start_11_2], 5*Nt, 5*Nt,format = "csr")
    
    #Submatrix (1,2)
    Up = sparse.csr_matrix(np.zeros((5,5*Nt)))
    Right = sparse.csr_matrix(np.zeros((5*(Nt-1),5)))
    Sub12_1 = np.zeros((Nt-1, 5, 5))
    for n in range(0,Nt-1):
        Sub12_1[n,:,:] = -1 * Delta_t * Rinv * np.matmul(B,np.transpose(B))
    Sub12_2 = sparse.block_diag(Sub12_1, format ="csr")   
    Sub12_3 = sparse.hstack([Sub12_2, Right], format = "csr")
    Sub12 = sparse.vstack([Up, Sub12_3], format = "csr")    
    
    #Submatrix(2,1)
    Sub21_1 = np.zeros((Nt, 5, 5))
    for n in range(0,Nt-1):
        Sub21_1[n,:,:] = Delta_t * (np.transpose(F) + F)
    Sub21 = sparse.block_diag(Sub21_1, format ="csr")       
    
    #Submatrix(2,2)
    Sub22_1 = np.zeros((Nt, 5, 5))
    for n in range(0,Nt-1):
        Sub22_1[n,:,:] = Delta_t * (np.transpose(A) - \
                                    Rinv * np.matmul(P[n,:,:], np.matmul(B, np.transpose(B))))    
    Sub22 = sparse.block_diag(Sub22_1, format ="csr")    
    Sub22_2 = np.ones((Nt+1)*5)    
    row_start_22_2 = 5
    Sub22 = Sub22 + sparse.spdiags(Sub22_2,[row_start_22_2], 5*Nt, 5*Nt, format = "csr")    

    Sub1 = sparse.vstack([Sub11, Sub21], format = "csr")
    Sub2 = sparse.vstack([Sub12, Sub22], format = "csr")
    M = sparse.hstack([Sub1, Sub2], format = "csr")
    return M                               

# ============================================= #
# Construction of K
# ============================================= #
def Construct_K(XBar0, C, H, P, Nt, Delta_t):
    K = np.zeros(10*Nt)
    K[0:5] = np.reshape(XBar0,(5))
    for n in range(1,Nt):
        K[5*n:5*(n+1)] = Delta_t * C[:,n-1]
    for n in range(0,Nt-1):
        K[5*Nt+5*n:5*Nt+5*(n+1)] = Delta_t * (np.matmul(P[n,:,:],C[:,n]) + H[:,n])
    return K

# ============================================= #
# Solution of Linear System for Mean Processes and Rho
# ============================================= #
def Solver_LinearSystem(M, K, Nt):
    I = sparse.identity(10*Nt)
    XBarRho = sparsela.spsolve(I-M, K)
    return XBarRho

# ============================================= #
# Calculation of s
# ============================================= #
def Calculate_s_MFC(REN, p2, p3, Rmax, a, B, C, P, J, Rinv, XBarRho, Delta_t, Nt):
    runningSum = 0
    for n in range(0,Nt):
        runningSum = runningSum + Delta_t * (np.matmul(a, P[n,:,:]).trace() - \
        0.5 * Rinv * np.matmul(np.reshape(XBarRho[5*Nt+5*n:5*Nt+5*(n+1)],(1,5)), B) ** 2 + \
        np.matmul(np.transpose(C[:,n]), XBarRho[5*Nt+5*n:5*Nt+5*(n+1)]) + J[n])
    s = 0.75*(p2 * REN - p3 * np.sqrt(REN * (Rmax - REN))) + runningSum
    return s

# ============================================= #
# Calculation of Value Function at time 0
# ============================================= #
def Calculate_value_MFC(P, F, XBarRho, s, XVar0, Delta_t, Nt):
    P_sqrt = sla.sqrtm(P[0,:,:])
    #print('P_sqrt:', P_sqrt)
    runningSum = 0
    for n in range(0,Nt):
        runningSum = runningSum + Delta_t * (np.matmul(np.transpose(XBarRho[5*n:5*(n+1)]), \
                                                       np.matmul(F, XBarRho[5*n:5*(n+1)])))
    value = 0.5 * (np.matmul(P_sqrt, np.matmul(XVar0, P_sqrt)).trace() + \
                   np.matmul(np.transpose(XBarRho[0:5]),np.matmul(P[0,:,:], XBarRho[0:5]))) + \
    np.matmul(np.transpose(XBarRho[0:5]), XBarRho[5*Nt:5*(Nt+1)]) + s - runningSum
    return value



# ====================================================== #
# =========== MEAN FIELD GAME PREPARATION ============= #
# ====================================================== #

# ============================================= #
# Linear System for 1st order coefficient rho and mean
# ============================================= #



# ============================================= #
# Construction of M_XBar
# ============================================= #
def Construct_M_XBar(A, B, F, P, Rinv, Delta_t, Nt):
    Blocks = np.zeros((Nt, 5, 5))
    for n in range(1,Nt):
        Blocks[n,:,:] = Delta_t * (A - Rinv * np.matmul(np.matmul(B, np.transpose(B)), P[n-1,:,:]))    
    DiagMat = sparse.block_diag(Blocks, format ="csr")    
    BelowDiag = np.ones(Nt*5)    
    row_start = -5
    M_XBar = DiagMat + sparse.spdiags(BelowDiag,[row_start], 5*Nt, 5*Nt,format = "csr")
    return M_XBar                              

# ============================================= #
# Construction of K_XBar
# ============================================= #
def Construct_K_XBar(XBar0, B, Rinv, r, C, Nt, Delta_t):
    K_XBar = np.zeros(5*Nt)
    K_XBar[0:5] = np.reshape(XBar0,(5))
    for n in range(1,Nt):
        K_XBar[5*n:5*(n+1)] = -Delta_t * (Rinv * np.matmul(np.matmul(B, np.transpose(B)), \
                                                          r[5*(n-1):5*n]) - C[:,n-1]) 
    return K_XBar

# ============================================= #
# Construction of M_R
# ============================================= #
def Construct_M_R(A, B, P, Rinv, Delta_t, Nt):
    Blocks = np.zeros((Nt, 5, 5))
    for n in range(0,Nt-1):
        Blocks[n,:,:] = Delta_t * (np.transpose(A) - Rinv * np.matmul(P[n,:,:], \
                                                                      np.matmul(B, np.transpose(B))))    
    DiagMat = sparse.block_diag(Blocks, format ="csr")    
    AboveDiag = np.ones((Nt+1)*5)    
    row_start = 5
    M_R = DiagMat + sparse.spdiags(AboveDiag,[row_start], 5*Nt, 5*Nt, format = "csr")    
    return M_R                               

# ============================================= #
# Construction of K_R
# ============================================= #
def Construct_K_R(XBar, C, H, F, P, Nt, Delta_t):
    K_R = np.zeros(5*Nt)
    for n in range(0,Nt-1):
        K_R[5*n:5*(n+1)] = Delta_t * (np.matmul(P[n,:,:],C[:,n]) + H[:,n] + \
                                      np.matmul(np.transpose(F), XBar[5*n:5*(n+1)]))
    return K_R

# ============================================= #
# Solution of Linear System for Mean Processes
# ============================================= #
def Solver_LinearSystem_XBar(M_XBar, K_XBar, Nt):
    I = sparse.identity(5*Nt)
    XBar = sparsela.spsolve(I-M_XBar, K_XBar)
    return XBar

# ============================================= #
# Solution of Linear System for Mean Processes and Rho
# ============================================= #
def Solver_LinearSystem_R(M_R, K_R, Nt):
    I = sparse.identity(5*Nt)
    r = sparsela.spsolve(I-M_R, K_R)
    return r

# ============================================= #
# Calculation of s in MFG
# ============================================= #
def Calculate_s_MFG(REN, p2, p3, Rmax, a, B, C, P, r, GBar, J, Rinv, XBar, Delta_t, Nt):
    runningSum = 0
    for n in range(0,Nt):
        runningSum = runningSum + Delta_t * (np.matmul(a, P[n,:,:]).trace() - \
        0.5 * Rinv * np.matmul(np.reshape(r[n*5:(n+1)*5],(1,5)), B) ** 2 + \
        np.matmul(np.transpose(C[:,n]), r[n*5:(n+1)*5]) + \
        np.matmul(np.transpose(XBar[n*5:(n+1)*5]), np.matmul(GBar,XBar[n*5:(n+1)*5])) + J[n])
    s = 0.75*(p2 * REN - p3 * np.sqrt(REN * (Rmax - REN))) + runningSum
    return s

# ============================================= #
# Calculation of Value Function at time 0
# ============================================= #
def Calculate_value_MFG(P, XBar, r, s, XVar0):
    P_sqrt = sla.sqrtm(P[0,:,:]) 
    value = 0.5 * (np.matmul(P_sqrt, np.matmul(XVar0, P_sqrt)).trace() + \
                   np.matmul(np.transpose(XBar[0:5]),np.matmul(P[0,:,:], XBar[0:5]))) + \
    np.matmul(np.transpose(XBar[0:5]), r[0:5]) + s
    return value



# ============================================================================= #
# ======================== MAIN FUNCTIONS FOR MINORS ========================== #
# ============================================================================= #


# =================================================================== #
# ======================== MEAN FIELD GAME ========================== #
# =================================================================== #

# ============================================== #
# Bisection Search Algorithm for Mean-Field Game
# ============================================== #
def bisection_search(upper, lower, kappa2, alpha, sig0, sig1, theta, XBarAfter, XVar0, Nt, t_grid, \
                     B, H, F, G, GBar, Rinv, J, ST, tol, T, Delta_t, p2, p3, Rmax):    
    #Initializatiom
    golden = (1+math.sqrt(5))/2
    middle1 = upper - (upper - lower) / golden
    middle2 = lower + (upper - lower) / golden
    while(abs(middle1 - middle2) > tol):
        #Expected cost with middle 1
        (A_1, C_1, sigma_1, a_1) = Input_Prep_REN(kappa2, middle1, alpha, t_grid, sig0, sig1, theta, Nt)
        P_1 = Solver_ODE_P(2*ST, A_1, B, G, Rinv,T, t_grid)
        M_R_1 = Construct_M_R(A_1, B, P_1, Rinv, Delta_t, Nt)
        K_R_1 = Construct_K_R(XBarAfter, C_1, H, F, P_1, Nt, Delta_t)
        r_1 = Solver_LinearSystem_R(M_R_1, K_R_1, Nt)
        s_1 = Calculate_s_MFG(middle1, p2, p3, Rmax, a_1, B, C_1, P_1, r_1, GBar, J, Rinv, XBarAfter, Delta_t, Nt)
        value_middle1 = Calculate_value_MFG(P_1, XBarAfter, r_1, s_1, XVar0)
        
        #Expected cost with middle 2
        (A_2, C_2, sigma_2, a_2) = Input_Prep_REN(kappa2, middle2, alpha, t_grid, sig0, sig1, theta, Nt)
        P_2 = Solver_ODE_P(2*ST, A_2, B, G, Rinv,T, t_grid)
        M_R_2 = Construct_M_R(A_2, B, P_2, Rinv, Delta_t, Nt)
        K_R_2 = Construct_K_R(XBarAfter, C_2, H, F, P_2, Nt, Delta_t)
        r_2 = Solver_LinearSystem_R(M_R_2, K_R_2, Nt)
        s_2 = Calculate_s_MFG(middle2, p2, p3, Rmax, a_2, B, C_2, P_2, r_2, GBar, J, Rinv, XBarAfter, Delta_t, Nt)
        value_middle2 = Calculate_value_MFG(P_2, XBarAfter, r_2, s_2, XVar0)
        
        #Expected cost comparison
        if(value_middle1 < value_middle2):
            upper = middle2
        else:
            lower = middle1
        middle1 = upper - (upper - lower) / golden
        middle2 = lower + (upper - lower) / golden
        
    value = (value_middle1 + value_middle2)/2
    optREN = (upper + lower) / 2
    return(value, optREN)

# ============================================== #
# Calculation of MF Game Equilibrium with Walk-away Option
# ============================================== #
def main_MFG_walkaway(kappa1, kappa2, alpha, delta, c1, c2, c3,\
                  rho_0, rho_1, tau, p1, p2, p3, Rmax, sig0, sig1, theta, Dt, t_grid, Nt, damp, \
                  XVar0, XBar0, XBarBefore, iteration, RENLow, RENHigh, tol, epsilon, sunk_cost, T, Delta_t):
    (B, H, F, G, GBar, R, Rinv, J, ST) = Input_Prep_no_REN(kappa1, delta, c1, c2, c3, rho_0, rho_1, tau, p1, Dt, Nt)
    XBarAfter = np.ones(5*Nt)
    convergence = la.norm(XBarAfter - XBarBefore)

    it = 0
    while(la.norm(XBarAfter - XBarBefore) > epsilon):
        (value, optREN) = bisection_search(RENHigh, RENLow, kappa2, alpha, sig0, sig1, theta, XBarAfter, XVar0, Nt, t_grid, \
                     B, H, F, G, GBar, Rinv, J, ST, tol, T, Delta_t, p2, p3, Rmax)     
        if (value < sunk_cost):    
            (A, C, sigma, a) = Input_Prep_REN(kappa2, optREN, alpha, t_grid, sig0, sig1, theta, Nt)
            P = Solver_ODE_P(2*ST, A, B, G, Rinv, T, t_grid)
            M_R = Construct_M_R(A, B, P, Rinv, Delta_t, Nt)
            K_R = Construct_K_R(XBarAfter, C, H, F, P, Nt, Delta_t)
            opt_r = Solver_LinearSystem_R(M_R, K_R, Nt)
            opt_s = Calculate_s_MFG(optREN, p2, p3, Rmax, a, B, C, P, opt_r, GBar, J, Rinv, XBarAfter, Delta_t, Nt)
            value = Calculate_value_MFG(P, XBarAfter, opt_r, opt_s, XVar0)
        
            XBarBefore = XBarAfter.copy()
            M_XBar = Construct_M_XBar(A, B, F, P, Rinv, Delta_t, Nt)
            K_XBar = Construct_K_XBar(XBar0, B, Rinv, opt_r, C, Nt, Delta_t)
            XBarTemp = Solver_LinearSystem_XBar(M_XBar, K_XBar, Nt)
            XBarAfter = damp * XBarTemp + (1-damp) * XBarBefore
        else:
            optREN = 0.0
            XBarBefore = XBarAfter.copy()
            XBarTemp = np.tile(np.reshape(XBar0,(5)),Nt)
            XBarAfter = damp * XBarTemp + (1-damp) * XBarBefore
            value = 0.0
        it = it+1    
        convergence = np.append(convergence, la.norm(XBarAfter - XBarBefore)) 
        if (it%iteration == 1):
            print(convergence[-1])
        #print("convergence: ", convergence)
    return (value, optREN, XBarBefore, convergence)


# =================================================================== #
#======================= MEAN FIELD CONTROL ======================== #
# =================================================================== #

# ============================================== #
# Calculation of MFC Eq with Walk-Away (Bisection added)
# ============================================== #
def main_MFC_walkaway(kappa1, kappa2, alpha, delta, c1, c2, c3, rho_0, rho_1, tau, p1, p2, p3, Rmax, \
             sig0, sig1, Dt, XVar0, XBar0, t_grid, Nt, RENLow, RENHigh, tol, sunk_cost, Delta_t, theta, T):
    (B, H, F, G, GBar, R, Rinv, J, ST) = Input_Prep_no_REN(kappa1, delta, c1, c2, c3, \
                                                           rho_0, rho_1, tau, p1, Dt, Nt)
    #Initializatiom
    golden = (1+math.sqrt(5))/2
    middle1 = RENHigh - (RENHigh - RENLow) / golden
    middle2 = RENLow + (RENHigh - RENLow) / golden
    
    while(abs(middle1-middle2)>tol):
        #Expected cost with middle 1
        (A_1, C_1, sigma_1, a_1) = Input_Prep_REN(kappa2, middle1, alpha, t_grid, sig0, sig1, theta, Nt)
        P_1 = Solver_ODE_P(2*ST, A_1, B, G, Rinv,T, t_grid)
        M_1 = Construct_M(A_1, B, F, P_1, Rinv, Delta_t, Nt)
        K_1 = Construct_K(XBar0, C_1, H, P_1, Nt, Delta_t)
        XBarRho_1 = Solver_LinearSystem(M_1, K_1, Nt)
        s_1 = Calculate_s_MFC(middle1, p2, p3, Rmax, a_1, B, C_1, P_1, J, Rinv, XBarRho_1, Delta_t, Nt)
        value_middle1 = Calculate_value_MFC(P_1, F, XBarRho_1, s_1, XVar0, Delta_t, Nt)
        
        #Expected cost with middle 2    
        (A_2, C_2, sigma_2, a_2) = Input_Prep_REN(kappa2, middle2, alpha, t_grid, sig0, sig1, theta, Nt)
        P_2 = Solver_ODE_P(2*ST, A_2, B, G, Rinv,T, t_grid)
        M_2 = Construct_M(A_2, B, F, P_2, Rinv, Delta_t, Nt)
        K_2 = Construct_K(XBar0, C_2, H, P_2, Nt, Delta_t)
        XBarRho_2 = Solver_LinearSystem(M_2, K_2, Nt)
        s_2 = Calculate_s_MFC(middle2, p2, p3, Rmax, a_2, B, C_2, P_2, J, Rinv, XBarRho_2, Delta_t, Nt)
        value_middle2 = Calculate_value_MFC(P_2, F, XBarRho_2, s_2, XVar0, Delta_t, Nt)
        
        #Expected cost comparison
        if(value_middle1 < value_middle2):
            RENHigh = middle2
        else:
            RENLow = middle1
        middle1 = RENHigh - (RENHigh - RENLow) / golden
        middle2 = RENLow + (RENHigh - RENLow) / golden
        
    value = (value_middle1 + value_middle2)/2
    if (value < sunk_cost):    
        optREN = (RENLow + RENHigh) / 2
        (A, C, sigma, a) = Input_Prep_REN(kappa2, optREN, alpha, t_grid, sig0, sig1, theta, Nt)
        P = Solver_ODE_P(2*ST, A, B, G, Rinv,T, t_grid)
        M = Construct_M(A, B, F, P, Rinv, Delta_t, Nt)
        K = Construct_K(XBar0, C, H, P, Nt, Delta_t)
        XBarRho = Solver_LinearSystem(M, K, Nt)
        s = Calculate_s_MFC(optREN, p2, p3, Rmax, a, B, C, P, J, Rinv, XBarRho, Delta_t, Nt)
        value = Calculate_value_MFC(P, F, XBarRho, s, XVar0, Delta_t, Nt)
    else:
        optREN = 0.0
        XBarRho = np.tile(np.reshape(XBar0,(5)),Nt)
        value = 0.0
    return (value, optREN, XBarRho)


# ========================================================================= #
#======================= ADDITION OF THE REGULATOR ======================== #
# ========================================================================= #
def Calculate_cost_major(alpha1, alpha2, alpha3, alpha4, alpha5, PBar_star, tau_star, Dt, XBar, XBar0, tau, Nt, Delta_t, c2):
    runningSum = 0
    for n in range(0,Nt):
        runningSum = runningSum + Delta_t * (alpha4 * (XBar[5*n] - Dt[n]) * (XBar[5*n] - Dt[n]))
    major_cost = alpha1 * max(XBar[5*Nt-2] - PBar_star, 0) - alpha2 * tau * (max(XBar[5*Nt-2]-XBar0[3],0)) + \
    alpha3 * (tau-tau_star) * (tau-tau_star) + runningSum + alpha5 * (c2**2)
    return major_cost
    

def main_Stackelberg_MFG(kappa1, kappa2, alpha, delta, c1, c3, rho_0, rho_1, \
                         p1, p2, p3, Rmax, sig0, sig1, theta, Dt, t_grid, Nt, damp, XVar0, XBar0, XBarBefore, \
                         iteration, RENLow, RENHigh, tol, epsilon, alpha1, alpha2, alpha3, alpha4, alpha5, \
                         PBar_star, tau_star, taxGrid, c2Grid, sunk_cost, T, Delta_t):
    MajorCost = np.zeros((np.size(taxGrid), np.size(c2Grid)))
    optREN_all = -5*np.ones((np.size(taxGrid), np.size(c2Grid)))
    MinorCost = np.zeros((np.size(taxGrid), np.size(c2Grid)))
    i=0 # tax ID
    j=0 # c2 ID
    Prod_overtime = -100*np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    Prod_NonREN_overtime = -100 * np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    Prod_REN_overtime_mean = -100 * np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    Prod_sum = -100 * np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    MinorResults = {}     
    for tau in tqdm(taxGrid):
        for c2 in tqdm(c2Grid):
            MinorResults[i*np.size(c2Grid)+j] = \
            main_MFG_walkaway(kappa1, kappa2, alpha, delta, c1, c2, c3, \
                     rho_0, rho_1, tau, p1, p2, p3, Rmax, sig0, sig1, theta, Dt, t_grid, Nt, damp, \
                     XVar0, XBar0, XBarBefore, iteration, RENLow, RENHigh, tol, epsilon, sunk_cost, T, Delta_t)
            value = MinorResults[i*np.size(c2Grid)+j][0]
            optREN = MinorResults[i*np.size(c2Grid)+j][1]
            XBar = MinorResults[i*np.size(c2Grid)+j][2]
            optREN_all[i,j] = optREN
            print("at tax, c2=", tau, c2, "value: ", value[0][0])
            MinorCost[i,j] = value[0][0]
            MajorCost[i,j] = Calculate_cost_major(alpha1, alpha2, alpha3, alpha4, alpha5, PBar_star, \
                                                  tau_star, Dt, XBar, XBar0, tau, Nt, Delta_t, c2)         
            Prod_overtime[i,j,:] = XBar[np.arange(0,5*Nt,5)] + kappa2 * optREN * theta
            Prod_NonREN_overtime[i,j,:] =  kappa1 * (XBar[np.arange(4 ,5*Nt, 5)]-XBar0[4])
            Prod_REN_overtime_mean[i,j,:] =  kappa2 * optREN * (np.sin(alpha*t_grid) + theta)
            Prod_sum[i,j,:] = kappa1 * (XBar[np.arange(4 ,5*Nt, 5)]-XBar0[4]) + kappa2 * optREN * (np.sin(alpha*t_grid)+theta)
            j = j+1     
        print('at tax=', tau,'optRen:', optREN_all[i,:])             
        j = 0
        i = i+1  
    return (MajorCost, MinorCost, optREN_all, Prod_overtime, Prod_NonREN_overtime, \
            Prod_REN_overtime_mean, Prod_sum, MinorResults)  


def main_Stackelberg_MFC(kappa1, kappa2, alpha, delta, c1, c3, rho_0, rho_1, \
                         p1, p2, p3, Rmax, sig0, sig1, theta, Dt, XVar0, XBar0, t_grid, Nt, Delta_t, RENLow, RENHigh, tol, \
                         alpha1, alpha2, alpha3, alpha4, alpha5, PBar_star, tau_star, \
                         taxGrid, c2Grid, sunk_cost, T):
    MajorCost = np.zeros((np.size(taxGrid), np.size(c2Grid)))
    optREN_all = -5*np.ones((np.size(taxGrid), np.size(c2Grid)))
    MinorCost = np.zeros((np.size(taxGrid), np.size(c2Grid)))
    i=0
    j=0
    Prod_overtime = -100*np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    Prod_NonREN_overtime = -100 * np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    Prod_REN_overtime_mean = -100 * np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    Prod_sum = -100 * np.ones((np.size(taxGrid), np.size(c2Grid), Nt))
    MinorResults = {} 
    for tau in tqdm(taxGrid):
        for c2 in tqdm(c2Grid):
            MinorResults[i*np.size(c2Grid)+j] = main_MFC_walkaway(kappa1, kappa2, alpha, delta, \
                                                             c1, c2, c3, rho_0, rho_1, \
                                                             tau, p1, p2, p3, Rmax, sig0, sig1, Dt, XVar0, XBar0, \
                                                             t_grid, Nt, RENLow, RENHigh, tol, sunk_cost, Delta_t, theta, T)
                                                                  
            value = MinorResults[i*np.size(c2Grid)+j][0]
            optREN = MinorResults[i*np.size(c2Grid)+j][1]
            XBarRho = MinorResults[i*np.size(c2Grid)+j][2]       
            optREN_all[i,j] = optREN
            print("at tax, c2=", tau, c2, "value: ", value[0][0])
            MinorCost[i,j] = value[0][0]
            MajorCost[i,j] = Calculate_cost_major(alpha1, alpha2, alpha3, alpha4, alpha5, PBar_star, \
                                                  tau_star, Dt, XBarRho[0:5*Nt], XBar0, tau, Nt, Delta_t, c2)
            Prod_overtime[i,j,:] = XBarRho[np.arange(0,5*Nt,5)] + kappa2 * optREN * theta
            Prod_NonREN_overtime[i,j,:] =  kappa1 * (XBarRho[np.arange(4 ,5*Nt, 5)]-XBar0[4])
            Prod_REN_overtime_mean[i,j,:] =  kappa2 * optREN * (np.sin(alpha*t_grid)+theta)
            Prod_sum[i,j,:] = kappa1 * (XBarRho[np.arange(4 ,5*Nt, 5)]-XBar0[4]) + kappa2 * optREN * (np.sin(alpha*t_grid)+theta)
            j = j+1
        print('at tax=', tau,'optRen:', optREN_all[i,:])             
        j = 0
        i = i+1
    return (MajorCost, MinorCost, optREN_all, Prod_overtime, Prod_NonREN_overtime,\
            Prod_REN_overtime_mean, Prod_sum, MinorResults)

