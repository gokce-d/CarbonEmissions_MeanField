import carbon as carb
from parameters5 import *
import pickle


#v3 (par 5)
taxGrid = np.arange(0., 101., 5.)
c2Grid = np.array([50., 100., 250., 500., 750., 1000., 1500., 2000., 2500., 3000., 4000., 5000.])

MFC ={}

MFC[0] = carb.main_Stackelberg_MFC(kappa1, kappa2, alpha, delta, c1, c3, rho_0, rho_1, \
                         p1, p2, p3, Rmax, sig0, sig1, theta, Dt, XVar0, XBar0, t_grid, Nt, Delta_t, RENLow, RENHigh, tol, \
                         alpha1, alpha2, alpha3, alpha4, alpha5, PBar_star, tau_star, \
                         taxGrid, c2Grid, sunk_cost, T)
                         
filenameMFC='data/MFC_Jun15'
with open(filenameMFC, "wb") as f:
    pickle.dump(MFC[0], f)                         




