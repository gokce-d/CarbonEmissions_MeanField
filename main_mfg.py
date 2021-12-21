import carbon as carb
from parameters5 import *
import pickle

taxGrid = np.array([60.])
c2Grid = np.array([50., 100., 250., 500., 750., 1000., 1500., 2000., 2500., 3000., 4000., 5000.])

MFG = {}
iteration=5
MFG[0] = carb.main_Stackelberg_MFG(kappa1, kappa2, alpha, delta, c1, c3, rho_0, rho_1, \
                         p1, p2, p3, Rmax, sig0, sig1, theta, Dt, t_grid, Nt, damp, XVar0, XBar0, XBarBefore, \
                         iteration, RENLow, RENHigh, tol, epsilon, alpha1, alpha2, alpha3, alpha4, alpha5, \
                         PBar_star, tau_star, taxGrid, c2Grid, sunk_cost, T, Delta_t)

filenameMFG='data/MFG_60_Jun15'
with open(filenameMFG, "wb") as f:
    pickle.dump(MFG[0], f)                         




