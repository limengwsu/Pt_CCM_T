#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last test time Nov 22nd 2022 using Spyder 4.0.1
Python 3.7.7
@author: Li, Meng

Modified from Chlp_pump model
"""
##This is the code equivalent for loadparams1()
###This code gives p as parameters object, can be modified to dictionary if
###necessary meanwhile the Fobs is output as dictionary
import csv
import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import odeint #older solver
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_pdf import PdfPages#save multi fig to pdf

def d_to_csv(d, filename = None):
    """
    save a dictionary to csv
    Parameters
    ----------
    filename : string, optional
    d : dictionary
    """
    if filename == None:
        w = csv.writer(open('output.csv', 'w'))
    else:
        w = csv.writer(open(filename, 'w'))
    for key, val in d.items():
        w.writerow([key, '{:1.4e}'.format(val)])

def frac_C(k1, k2, H_c, DIC_sp='HCO3-'):
    #calculate the fraction of a dic species. 
    #k1, k2 are the dissociation constants for following rxn resp.
    #CO2 + H2O <--> H+ + HCO3-    (k1)
    #HCO3- <--> H+ + CO3--        (k2)
    #H_c and DIC_sp are proton concentration and DIC species resp.
    if DIC_sp == 'CO2':
        f_dic = H_c**2/(H_c**2 + k1*H_c + k1*k2)
    elif DIC_sp == 'HCO3-':
        f_dic = k1*H_c/(H_c**2 + k1*H_c + k1*k2)
    elif DIC_sp == 'CO3--': 
        f_dic = k1*k2/(H_c**2 + k1*H_c + k1*k2)
    else:
        f_dic = 0
    return f_dic
def CiFluxes(dicD, d):#dicD, d: dictionaries of DIC and parameters resp.ly
    Fd = {}#Fluxes dictionary
    # Fluxes followed are diffusional fluxes, the 1e-9 is to convert nmol to mol
    # The Matlab code on Diff flux does not seem right.
    Fd['Cexc_srf'] = d['fc_bl']*(dicD['CO2_exc']-dicD['CO2_srf'])*1e-9#CO2 flux from extracellular (bulk) solution to cell surface
    Fd['Bexc_srf'] = d['fb_bl']*(dicD['Bic_exc']-dicD['Bic_srf'])*1e-9#HCO3- flux from extracellular (bulk) solution to cell surface
    Fd['Csrf_cyt'] = d['fc_sm']*(dicD['CO2_srf']-dicD['CO2_cyt'])*1e-9#CO2 flux from surface to cytoplasm
    Fd['Bsrf_cyt'] = d['fb_sm']*(dicD['Bic_srf']-dicD['Bic_cyt'])*1e-9#HCO3- flux from surface to cytoplasm
    Fd['Ccyt_chp'] = d['fc_p'] *(dicD['CO2_cyt']-dicD['CO2_chp'])*1e-9#CO2 flux from cytoplasm into chloroplast stroma
    Fd['Bcyt_chp'] = d['fb_p'] *(dicD['Bic_cyt']-dicD['Bic_chp'])*1e-9#HCO3- flux from cytoplasm into chloroplast stroma
    Fd['Cchp_pyr'] = d['fc_y'] *(dicD['CO2_chp']-dicD['CO2_pyr'])*1e-9#CO2 flux from chloroplast stroma into pyrenoid
    Fd['Bchp_pyr'] = d['fb_y'] *(dicD['Bic_chp']-dicD['Bic_pyr'])*1e-9#HCO3- flux from chloroplast stroma into pyrenoid
    # all Fluxes above are diffusional fluxes
    # CO2 dehydration fluxes followed, could be catalyzed
    Fd['hyd_exc']   = d['kuf'] *dicD['CO2_exc']*d['Ve']/d['N'] *1e-9 #CO2 hydration rate in bulk/surface solution
    Fd['dehyd_exc'] = d['kur'] *dicD['Bic_exc']*d['Ve']/d['N'] *1e-9#HCO3- dehydration rate in bulk/surface solution
    Fd['hyd_srf']   = d['ksf'] *dicD['CO2_srf']*d['Vs']  *1e-9      #PER CELL FLUX OF CO2/HCO3- AT SURFACE, CONTINUE THE NOTE NEXT LINE
    Fd['dehyd_srf'] = d['ksr'] *dicD['Bic_srf']*d['Vs']  *1e-9      #note the difference in ksf between Python and Matlab    
    Fd['hyd_cyt']   = d['kcf'] *dicD['CO2_cyt']*d['Vc']  *1e-9
    Fd['dehyd_cyt'] = d['kcr'] *dicD['Bic_cyt']*d['Vc']  *1e-9
    Fd['hyd_chp']   = d['kpf'] *dicD['CO2_chp']*d['Vp']  *1e-9
    Fd['dehyd_chp'] = d['kpr'] *dicD['Bic_chp']*d['Vp']  *1e-9
    Fd['hyd_pyr']   = d['kyf'] *dicD['CO2_pyr']*d['Vy']  *1e-9
    Fd['dehyd_pyr'] = d['kyr'] *dicD['Bic_pyr']*d['Vy']  *1e-9
    
    return Fd

def C13frac(delta, Fluxes, cdict, Yss, pfilename, par_title):
    #natural abundance 13C fractionation prediction using fluxes determined
    #during photosynthesis and known or assumed fractionation factors
    
    #describe fractionation factors and convert to alpha notation
    Ecdiff = 0                   #diffusional fractionation of CO2
    Ecup = 0                     #fractionation associated with transport of CO2
    #Ebdiff = 0   #if this value is not 0, change the way VepsDiff is generated
                  #fractionation of HCO3 from diffusion
    Ebup= 0                      #fractionation of HCO3 from uptake 
    Erub = -27                   #rubisco fractionation #NOT REALLY SURE WHERE I GOT -22, THERE'S NO DATA ON DIATOMS. If you look at Tcherkez 2006 correlations between specificity and E would predict diatom RubisCO is ~-26, similar to -25 I used to use
    Ecb = -13                    #fractionation factor for production of HCO3 from CO2 (OLeary from Zeebe and Wolfe Gladrow 2001; at 25C).
    Ebc = Ecb + delta[0]-delta[1]#dC13CO2diff      #fractionation factor for production of CO2 from HCO3, not currently distinguishing between CA produced or uncatalyzed; must move toward equilibrium value 
    #Ebc = -10;
    #vectors associating each flux with a fractionation factor (Epsilon)
    VepsDiff = np.array([Ecdiff]*16, dtype = float)
    VepsDiff[0] = VepsDiff[0] + delta[0]
    VepsDiff[2] = VepsDiff[2] + delta[1]
    VepsDiff = VepsDiff.reshape((16,1))
    
    # VepsDiff = [Ecdiff+del(1,1); Ecdiff; Ebdiff + del(2,1); Ebdiff;...         #diffusion from bulk solution to surface layer; include isotope composition of bulk CO2 and HCO3- because they're fixed
    #             Ecdiff; Ecdiff; Ebdiff; Ebdiff;...         #diffusion from surface layer to cytoplasm
    #             Ecdiff; Ecdiff; Ebdiff; Ebdiff;...         #diffusion from cytoplast to chloroplast stroma
    #             Ecdiff; Ecdiff; Ebdiff; Ebdiff];           #diffusion from chloroplast stroma to pyrenoid 
    # above is from MATLAB code
    
    VepsHyd  = np.array([[Ecb],[Ebc]]*5, dtype = float) #[Ecb; Ebc; Ecb; Ebc; Ecb; Ebc; Ecb; Ebc; Ecb; Ebc];
    VepsAct  = np.array([[Ecup], [Ebup], [Ecup], [Ebup], [Erub]], dtype = float)
            
    #define matrix for linear system, rows are the systems dCs/dt, dBs/dt,etc
    #all at steady state. Column represents constant terms
    
    # %mass transfer vector:
    vMT = np.array([cdict['fc_bl'], cdict['fb_bl'], cdict['fc_sm'], cdict['fb_sm'], \
           cdict['fc_p'], cdict['fb_p'],   cdict['fc_y'],  cdict['fb_y']])
    v1 = vMT * Yss[0:8,0] #element wise multiply: [CO2]*fc, [HCO3-]*fb
    v2 = vMT * Yss[2:10,0] #next compartment: [CO2]*fc, [HCO3-]*fb
    #Flux_diff = v1- v2# calculate diffusional flux
    
    ####====the following lines probably caused some problems in Matlab code====###
    # v1 = vMT.*Yss(1:8,1);
    # v2 = vMT.*Yss(3:10,1);
    # t = [v1'; v2']; this line might be t = [v1, v2]; t = t';
    # Flux.Diff = t(:);
    
    """
    v1 is the flux of CO2, HCO3- from outside to inside, NOTE: DIFFUSIVE
    Bulk --> Srf; Srf-->Cyt; Cyt-->Chp, Chp-->Pyr
    
    v2 is the opposite and v1-v2 = the net flux from outside to inside
    """
    Flux_active = np.array([0, Fluxes['Bup_cyt'], 0, Fluxes['Bup_chp'], \
                            Fluxes['PS_rate']], dtype = float)
    
    #M = zeros(8,8);
    M = np.zeros((8,8))
    
    """
    M matrix would look like this after assigning numbers
    
    [[Loss_CO2_srf, CO2_hyd_gain, CO2_dif_gain, 0, 0, 0, 0, 0], #dif cyt_to_srf
    [Bic_hyd_gain, Bic_loss_srf, 0,  Bic_dif_gain, 0, 0, 0, 0], #dif cyt_to_srf
    [CO2_gn_f_srf, 0, CO2_loss_cyt, CO2_hyd_gain, CO2_dif_gain, 0, 0, 0], #dif_Chp_to Cyt
    [0, Bic_gn_f_srf, Bic_hyd_gn, Bic_loss_cyt, 0, Bic_dif_gn, 0, 0], #dif_chp_to_cyt
    [0, 0, CO2_gn_f_cyt, 0, CO2_loss_chp, CO2_gn_hyd, CO2_dif_gn, 0], #dif_pyr_to_chp
    [0, 0, 0, Bic_gn_f_Cyt, Bic_gn_hyd, Bic_loss_Chp, 0, Bic_dif_gn], #dif_pyr_to_chp
    [0, 0, 0, 0, CO2_gn_f_chp, 0, CO2_loss_pyr, CO2_gn_hyd],          #Pyr CO2
    [0, 0, 0, 0, 0, Bic_gn_f_chp, Bic_gn_hyd, Bic_loss_pyr],          #Pyr Bic
    ]
    
    """
    #dCs/dt equation
    #M(1,1) = -(Flux.Diff(2,1) + Flux.Diff(5,1) + Flux.Hyd(3,1) + Flux.Active(1,1));
    #M(1,2) =  Flux.Hyd(4,1);
    #M(1,3) =  Flux.Diff(6,1);
    # loss of CO2 at srf due to diffusion to exc, cyt, and hydration, + active uptake (0)
    # odd number uses v1, even number uses v2, index of v1 or v2 times 2 + 1(in v1)\
    # or (2 in v2) = INDEX (in Flux.Diff(INDEX, 1))
    M[0,0] = - (v2[0] + v1[2] + Fluxes['hyd_srf'] + Flux_active[0])
    M[0,1] = Fluxes['dehyd_srf']
    M[0,2] = v2[2]
    
    #dBs/dt
    # M(2,1) = Flux.Hyd(3,1);
    # M(2,2) = -(Flux.Diff(4,1) + Flux.Diff(7,1) + Flux.Hyd(4,1) + Flux.Active(2,1));
    # M(2,4) = Flux.Diff(8,1);
    M[1,0] = Fluxes['hyd_srf']
    M[1,1] = -(v2[1] + v1[3] + Fluxes['dehyd_srf'] + Flux_active[1])
    M[1,3] = v2[3]
    
    #dCc/dt
    #M(3,1) = Flux.Diff(5,1) + Flux.Active(1,1);
    #M(3,3) = -(Flux.Diff(6,1) + Flux.Diff(9,1) + Flux.Hyd(5,1) + Flux.Active(3,1));
    #M(3,4) = Flux.Hyd(6,1);
    #M(3,5) = Flux.Diff(10,1);
    M[2,0] = v1[2] + Flux_active[2]
    M[2,2] = -(v2[2] + v1[4] + Fluxes['hyd_cyt'] + Flux_active[2])
    M[2,3] = Fluxes['dehyd_cyt']
    M[2,4] = v2[4]
    
    #dBc/dt
    #M(4,2) = Flux.Diff(7,1) + Flux.Active(2,1);
    #M(4,3) = Flux.Hyd(5,1);
    #M(4,4) = -(Flux.Diff(8,1) + Flux.Diff(11,1) + Flux.Hyd(6,1) + Flux.Active(4,1));
    #M(4,6) = Flux.Diff(12,1);
    M[3,1] = v1[3] +Flux_active[1]
    M[3,2] = Fluxes['hyd_cyt']
    M[3,3] = -(v2[3] + v1[5] + Fluxes['dehyd_cyt'] + Flux_active[3])
    M[3,5] = v2[5]
    
    #dCp/dt
    #M(5,3) = Flux.Diff(9,1) + Flux.Active(3,1);
    #M(5,5) = -(Flux.Diff(10,1) + Flux.Diff(13,1) + Flux.Hyd(7,1));
    #M(5,6) = Flux.Hyd(8,1);
    #M(5,7) = Flux.Diff(14,1);
    M[4,2] = v1[4] + Flux_active[2]
    M[4,4] = -(v2[4] + v1[6] + Fluxes['hyd_chp'])
    M[4,5] = Fluxes['dehyd_chp']
    M[4,6] = v2[6]
    
    #dBp/dt
    #M(6,4) = Flux.Diff(11,1) + Flux.Active(4,1);
    #M(6,5) = Flux.Hyd(7,1);
    #M(6,6) = -(Flux.Diff(12,1) + Flux.Diff(15,1) + Flux.Hyd(8,1));
    #M(6,8) = Flux.Diff(16,1);
    M[5,3] = v1[5] + Flux_active[3]
    M[5,4] = Fluxes['hyd_chp']
    M[5,5] = -(v2[5] + v1[7] + Fluxes['dehyd_chp'])
    M[5,7] = v2[7]
    
    #dCy/dt
    #M(7,5) = Flux.Diff(13,1);
    #M(7,7) = -(Flux.Diff(14,1) + Flux.Hyd(9,1) + Flux.Active(5,1));
    #M(7,8) = Flux.Hyd(10,1);
    M[6,4] = v1[6]
    M[6,6] = -(v2[6] + Fluxes['hyd_pyr'] + Flux_active[4])
    M[6,7] = Fluxes['dehyd_pyr']
    
    #dBy/dt
    # M(8,6) = Flux.Diff(15,1);
    # M(8,7) = Flux.Hyd(9,1);
    # M(8,8) = -(Flux.Diff(16,1) + Flux.Hyd(10));
    M[7,5] = v1[7]
    M[7,6] = Fluxes['hyd_pyr']
    M[7,7] = -(v2[7] + Fluxes['dehyd_pyr'])
    
    #construct vector of constants
    #diffusion matrix (effect of diffusive fluxes on Cs, Bs, etc)
    Mdiff = [[1,-1, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
             [0, 0, 1,-1, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1,-1, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1,-1, 0, 0,-1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0,-1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0,-1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1]]
    
    #hydration/dehydration matrix
    Mhyd = [[0, 0,-1, 1, 0, 0, 0, 0, 0, 0],         
            [0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0,-1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,-1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1,-1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0,-1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1,-1]]
        
    Mactive = [[-1, 0, 0, 0, 0],             #active flux matrix
               [ 0,-1, 0, 0, 0],
               [ 1, 0,-1, 0, 0],
               [ 0, 1, 0,-1, 0],
               [ 0, 0, 1, 0, 0],
               [ 0, 0, 0, 1, 0],
               [ 0, 0, 0, 0,-1],
               [ 0, 0, 0, 0, 0]]
    
    Flux_diff = np.concatenate((v1,v2)).reshape((2,8)).transpose().flatten().reshape((16,1))
    #individual diffusion flux for Matrix/np.array calculation
    Flux_hyd = np.array([[Fluxes['hyd_exc']], [Fluxes['dehyd_exc']],\
                         [Fluxes['hyd_srf']], [Fluxes['dehyd_srf']],\
                         [Fluxes['hyd_cyt']], [Fluxes['dehyd_cyt']],\
                         [Fluxes['hyd_chp']], [Fluxes['dehyd_chp']],\
                         [Fluxes['hyd_pyr']], [Fluxes['dehyd_pyr']]])
    Flux_active = Flux_active.reshape(5,1)
    V = -( Mdiff @(VepsDiff * Flux_diff) + \
            Mhyd @(VepsHyd  * Flux_hyd)  + \
         Mactive @(VepsAct  * Flux_active))  
    
    C13data = np.linalg.solve(M, V)#solution of a x = b for x; or here M@C13data = V
    #linalg.solve(a,b) if a is square; linalg.lstsq(a,b) otherwise
    dC13Fix = C13data[6,0] + Erub
    
    dC13_values = np.concatenate((np.array(delta, dtype=float).reshape((2,1)), C13data))
    
    dC13_list = ['13C-CO2_bulk', '13C-Bic_bulk', '13C-CO2_srf', '13C-Bic_srf',\
                 '13C-CO2_cyt',  '13C-Bic_cyt',  '13C-CO2_chp', '13C-Bic_chp',\
                 '13C-CO2_pyr', '13C-Bic_pyr']
    dC13_dict = {}
    for i, akey in enumerate(dC13_list):
        dC13_dict[akey] = dC13_values[i][0]
    dC13_dict['d13C_fixed'] = dC13Fix
    
    d_to_csv(dC13_dict, pfilename.replace('T.csv', par_title+'_C13isotope.csv'))
    return dC13_dict
#### unquote the following to use original "visco" and "diff_coef" functions 
# def visco(Sal, Pp, TC): #Function for viscosity calculaiton
#     v =  1.7910 - TC*(6.144e-02 - TC*(1.4510e-03 - TC*1.6826e-05))\
#        - 1.5290e-04*Pp + 8.3885e-08*Pp**2 + 2.4727e-03*Sal\
#        + (6.0574e-06*Pp - 2.6760e-09*Pp**2)*TC + (TC*(4.8429e-05\
#        - TC*(4.7172e-06 - TC*7.5986e-08)))*Sal
#     return (v)

# def diff_coef(TC, Sal, P):
# #Diffusion coefficients in sea water system constant calculation function
# #Modified from Xinping Hu's code on his website which was: 
# #Modified from FORTRAN Code by Dr. Bernie Boudreau at the Dalhousie Univ.
# #Modified from Matlab code by Dr. Bryan Hopkinson at UGA
    
# #  Note: 1) enter S in ppt(parts per thousand), T in deg. C and P in atm.
# #  2) diffusion coefficients are in units of cm**2/s
# #  3) H2O viscosity is in unit of centipoise
#     D = {} #dictionary of Diffusion coefficients
    
# # Viscosity calculation 
# # Valid for 0<T<30 and 0<S<36, Calculated viscosity is in centipoise.
#     Vis = visco(Sal, P, TC) # calculate in situ viscosity
#     Vis0 = visco(0, 1, TC)  # calculate pure water viscosity at in situ temperature at 1 atmospheric pressure

# #  Start calculations of diffusion coefficients in pure water at sample temperature.
#     TK = TC + 273.15
#     A = 12.5e-09 * np.exp(-5.22e-04 * P)
#     B = 925.0 * np.exp(-2.6e-04 * P)
#     T0 = 95.0 + 2.61e-02 * P
#     D['H2O'] = A * np.sqrt(TK) * np.exp(-B/(TK-T0)) * 1.0e+04

# #  Dissolved gases : from Wilke and Chang (1955)
# #    note: 1) MW = molecular weight of water
# #          2) VM = molar volumes (cm**3/mol) (Sherwood et al., 1975)
# #  The factor PHI(X, chi) is reduced to 2.26 as suggested by Hayduk and Laudie (1974).
# #  Prediction of Diffusion Coefficients for Nonelectrolytes in Dilute
# #  Aqueous Solutions  AlChE Journal (Vol. 20, No. 3) Wilke and Chang eq(2)
#     PHI = 2.26
#     MW = 18.0
#     A1 = np.sqrt(PHI*MW) * TK/Vis0
#     #VM_gas = [27.9, 37.3, 24.5, 35.2, 37.7] actual in the table
#     VM_gas = [25.6, 34.0, 25.8, 32.9, 37.7]
#     for i, a_gas in enumerate(['O2','CO2','NH3','H2S','CH4']):
#         D[a_gas] = 7.4e-08*(A1/(VM_gas[i]**0.6))


# #  The coefficients in pure water for the following species are
# #  calculated by the linear functions of temperature (deg C)
# #  found in Boudreau (1997, Springer-Verlag).

# #  i.e. NO3-,HS-,H2PO4-,CO3=,SO4=,Ca++,Mg++,Mn++,Fe++,NH4+,H+ & OH-, HCO3-, HPO4=, PO4(3-)

#     FAC = 1.0e-06
#     D['HCO3-'] = (5.06 + 0.275*TC)*FAC
#     D['CO3--'] = (4.33 + 0.199*TC)*FAC
#     D['NH4+']  = (9.5  + 0.413*TC)*FAC
#     D['HS-']   = (10.4 + 0.273*TC)*FAC
#     D['NO3-']  = (9.50 + 0.388*TC)*FAC
#     D['H2PO4-']= (4.02 + 0.223*TC)*FAC
#     D['HPO4--']= (3.26 + 0.177*TC)*FAC
#     D['PO4---']= (2.62 + 0.143*TC)*FAC
#     D['H+']    = (54.4 + 1.555*TC)*FAC
#     D['OH-']   = (25.9 + 1.094*TC)*FAC
#     D['Ca++']  = (3.60 + 0.179*TC)*FAC
#     D['Mg++']  = (3.43 + 0.144*TC)*FAC
#     D['Fe++']  = (3.31 + 0.150*TC)*FAC
#     D['Mn++']  = (3.18 + 0.155*TC)*FAC
#     D['SO4--'] = (4.88 + 0.232*TC)*FAC

# # H3PO4 : Least (1984) determined D(H3PO4) at 25 deg C and 0 ppt S.
# #         Assume that this value can be scaled by the Stokes-Einstein
# #         relationship to any other temperature.
#     TS1 = 25.0
#     SS1 = 0.0
#     Vis_P = visco(SS1, 1, TS1) 
#     D['H3PO4'] = 0.87e-05*Vis_P/298.15*TK/Vis0
      
# # B(OH)3 : Mackin (1986) determined D(B(OH)3) at 25 deg C and
# #           about 29.2 ppt S.
# #           Assume that this value can be scaled by the Stokes-Einstein
# #           relationship to any other temperature.
#     TS2 = 25.0
#     SS2 = 29.2      
#     Vis_B = visco(SS2, 1, TS2)
#     D['B(OH)3'] = 1.12e-05*Vis_B/298.15*TK/Vis0
      
# # B(OH)4 : No information on this species whatsoever! Boudreau and
# #           Canfield (1988) assume it is 12.5% smaller than B(OH)3.
#     D['B(OH)4-'] = 0.875*D['B(OH)3']

# #  H4SiO4 : Wollast and Garrels (1971) found D(H4SiO4) at 25 deg C
# #           and 36.1 ppt S.
# #           Assume that this value can be scaled by the Stokes-Einstein
# #           relationship to any other temperature.
#     TS3 = 25.0
#     SS3 = 36.1
#     Vis_Si = visco(SS3, 1, TS3)
#     D['H4SiO4'] = 1.0E-05*Vis_Si/298.15*TK/Vis0

# #  To correct for salinity, the Stokes-Einstein relationship is used.
# #  This is not quite accurate, but is at least consistent.

#     FAC1 = Vis0/Vis
#     for akey in list(D.keys()):
#         D[akey] = D[akey]*FAC1#ajust D by viscosity
#     #Diff_Coef=FAC1.*D #this calculation was not returned to the function in .m file
#     return D       

def visc(Tc, S, V = False):
    """
    Calculate the viscosity of water or seawater
    Reference:
        Sharqawy et al. 2009
        Thermophysical properties of seawater: a review of \
            existing correlations and data
        https://doi.org/10.5004/dwt.2010.1079
        Eq 22 and Eq 23
        viscosity unit in kg/m/s
    Parameters
    ----------
    Tc : scalar
        Temperature in °C.
    S : scalar
        Salinity in per mil (g/kg).
    V:  Bool
        whether or not return calculated viscosity, default is False

    Returns
    -------
    Viscosity or Seawater to water ratio.

    """
    Sref = S/1000 #reference salinity is written in kg/kg
    A = 1.541 + 1.998e-2 * Tc - 9.52e-5 * Tc**2
    B = 7.974 - 7.561e-2 * Tc + 4.724e-4 *Tc**2
    Sw_w = 1 + A*Sref + B*Sref**2#seawater to water ratio

    if V:        
        V_water = 4.2844e-5 + 1/( 0.157* (Tc+ 64.993)**2 -91.296)
        #water viscosity
        V_sw = V_water * Sw_w # seawater viscosity
        Sw_w = (Sw_w, V_sw)
        print("given viscosity is in kg/m/s, which is 1000 times of centipoise")
    
    return Sw_w

def Di(Tc, S=0, Ci=None):
    """
    Calculate Diffussion coefficient of CO2, HCO3-, CO3--, according to 
    R.E. Zeebe / Geochimica et Cosmochimica Acta 75 (2011) 2483–2498
    Original equation covers Tc in [0,100]

    Parameters
    ----------
    Tc :scalar
        temperature in  °C.
    S: scalar
        salinity in per mil (or g/kg)
    Ci : string, optional, default is None
        CO2, HCO3-, CO3--

    Returns
    -------
    A value of diffussion coefficient of a carbon species 
    or a dictionary if Ci is not specified.

    """
    Ci_list = ['CO2', 'HCO3-','CO3--']
    D0i = np.array([14.6836, 7.0158, 5.4468])*1e-5
    #unit in cm**2/s, or change 1e-5 to 1e-9 for m**2/s
    Ti  = np.array([217.2056, 204.0282, 210.2646])
    gamma_i = np.array([1.997, 2.3942, 2.1929])
    T = Tc + 273.15
    D = D0i * ((T/Ti)-1) **gamma_i # water
    if S!=0:
        D=D/visc(Tc, S) # adjust D for seawater
    if Ci in Ci_list:
        i = Ci_list.index(Ci)
        D = D[i]
    else:
        # print(type(D))
        D = dict(zip(Ci_list, D))

    return D


def f_Ci(y,t, Vm_Bc, Km_Bc, Vm_Bp, Km_Bp, mRub, kcat_R, Km_R, kuf,kur,\
         ksf,ksr, kcf,kcr, kpf,kpr, kyf,kyr, N, Ve,Vs,Vc,Vp,Vy,\
         fc_bl,fb_bl, fc_sm,fb_sm, fc_p,fb_p, fc_y,fb_y, RS): 
    """
    The differential equations of DIC and PS

    Returns
    -------
    dy/dt as a list

    """
    #Cideriv, differential equations of inorganic carbons
    CO2_exc, Bic_exc, CO2_srf, Bic_srf, CO2_cyt, Bic_cyt, \
    CO2_chp, Bic_chp, CO2_pyr, Bic_pyr = y
    #NOTE on nomenclature:
    #C for CO2, B for bicarbonate, followed by m.w. and location, 
    #such as C45e is mass_45 CO2 at extracellular compartment
    #c for co2, b for bicarbonate, second letter e,s,c,p,y for extracellular,
    #surface, cytosolic, chloroplast, pyrenoid; 
    # written # ce,cc,bc,cp, bp, cy, by = y        
    #CO2, Bic, exc, srf, cyt, chp, pyr are used in python code
    

    
    #calculate bicarbonate uptake and photosynthetic rates
    # ind = np.where(np.array(ON) <= t)[0]      #ind_array
    # if len(ind) == 0 or t > OFF[ind[-1]]:     #determine if light is off
    #     Bup_cyt, Bup_chp, PS = (0, 0, 0)
    # else:   #light on
    t0 = 0 #this can be modified if the starting time point is not zero
    f_active = 1-np.exp(-0.0045*(t-t0))
    Bup_cyt  = (Vm_Bc * Bic_srf)/(Km_Bc + Bic_srf) *1e9  *f_active# in nmol/cell
    Bup_chp  = (Vm_Bp * Bic_cyt)/(Km_Bp + Bic_cyt) *1e9  *f_active# in nmol/cell
    
    PS       = mRub * (kcat_R * CO2_pyr)/(Km_R + CO2_pyr) *1e9 *f_active # in nmol/cell
    #print(PS)

    #Bup is for bicarbonate uptake
    #PS for photosynthesis, using single letter P is confusing for p,P in earlier params
    """
    kuf, kur, ksf, ksr, kcf, kcr, kpf, kpr, kyf, kyr are kinetic constants for following rxn
    CO2[aq] --kxf--> HCO3-; CO2[aq] <--kxr-- HCO3-
    k for kinetic constant; x for location, u, s, c, p, y represent uncatalyzed,
    surface, cytosolic, chloroplastic, pyrenoid; f and r for forward and reverse.
    
    similarly fc_x and fb_x are mass transfer rate factor/constant
    for CO2 and bicarbonte respectively, bl for bulk, sm for surface across membrane
    V is for volume in Vs, Vc, Vp, Vy
    """
    # Note that the following Matlab code equation was changed by moving the
    # Vs (surface volume) term out of ksf/ksr calculations, so that the ksf/ksr
    # units are in s^-1, not cm3/s...    
    """    
    dcs = (1./p.Vs).*(-p.ksf.*cs  + p.ksr.*bs  + p.fc_bl.*(ce - cs) + p.fc_sm.*(cc - cs));
    dbs = (1./p.Vs).*( p.ksf.*cs  - p.ksr.*bs  + p.fb_bl.*(be - bs) + p.fb_sm.*(bc - bs) - Bupc);
    should be these lines of translated if abiding original setup
    dCO2_srf =(-ksf * CO2_srf + ksr * Bic_srf + fc_bl  *(CO2_exc - CO2_srf) + fc_sm *(CO2_cyt - CO2_srf))/Vs
    dBic_srf =( ksf * CO2_srf - ksr * Bic_srf + fb_bl  *(Bic_exc - Bic_srf) + fb_sm *(Bic_cyt - Bic_srf) - Bup_cyt)/Vs  
    but these following ones make more sense (line 171 and line 172)
    """
    
    dCO2_exc = -kuf * CO2_exc + kur * Bic_exc + (N/Ve) * fc_bl*(CO2_srf - CO2_exc)
    dBic_exc =  kuf * CO2_exc - kur * Bic_exc + (N/Ve) * fb_bl*(Bic_srf - Bic_exc)
    
    dCO2_srf = -ksf * CO2_srf + ksr * Bic_srf + (1/Vs) *(fc_bl*(CO2_exc - CO2_srf) + fc_sm*(CO2_cyt - CO2_srf))
    dBic_srf =  ksf * CO2_srf - ksr * Bic_srf + (1/Vs) *(fb_bl*(Bic_exc - Bic_srf) + fb_sm*(Bic_cyt - Bic_srf) - Bup_cyt)

    #dO2 = (PS - RS*1e9)* N/Ve #* 1e9 #convert mol/cm3/s to µM/s 
    #add variable O2 if one is interested in O2 evolution over time
            
    dCO2_cyt = -kcf * CO2_cyt + kcr * Bic_cyt + \
        (1/Vc) *(fc_sm*(CO2_srf - CO2_cyt) + fc_p *(CO2_chp - CO2_cyt)) + (1/Vc)* RS*1e9   
    dBic_cyt =  kcf * CO2_cyt - kcr * Bic_cyt + \
        (1/Vc) *(fb_sm*(Bic_srf - Bic_cyt) + fb_p *(Bic_chp - Bic_cyt) + Bup_cyt - Bup_chp)    
    dCO2_chp = -kpf * CO2_chp + kpr * Bic_chp + \
        (1/Vp) *(fc_p *(CO2_cyt - CO2_chp) + fc_y *(CO2_pyr - CO2_chp))
    dBic_chp =  kpf * CO2_chp - kpr * Bic_chp + \
        (1/Vp) *(fb_p *(Bic_cyt - Bic_chp) + fb_y *(Bic_pyr - Bic_chp) + Bup_chp)
    dCO2_pyr = -kyf * CO2_pyr + kyr * Bic_pyr + \
        (1/Vy) *(fc_y *(CO2_chp - CO2_pyr) - PS)
    dBic_pyr =  kyf * CO2_pyr - kyr * Bic_pyr + \
        (1/Vy) *(fb_y *(Bic_chp - Bic_pyr))
    #dydt = [dO2, dCO2_exc, dBic_exc, dCO2_srf, dBic_srf, dCO2_cyt, dBic_cyt, dCO2_chp, dBic_chp, dCO2_pyr, dBic_pyr]
    dydt = [dCO2_exc, dBic_exc, dCO2_srf, dBic_srf, dCO2_cyt, dBic_cyt, dCO2_chp, dBic_chp, dCO2_pyr, dBic_pyr]
    return dydt



#infile = 'Users/to/.../filename'
def get_parameter(infile, col = 1):
    """
    infile is a csv file storing parameter names and numbers in first and
    second,..., column respectively, with firstline as the headline (no numbers)
    The function returns a dictionary which can be extracted for calculations
    and the column title, one can use pandas to achieve similar goals
    """
    with open(infile, 'r') as f:
        lines=f.read().splitlines()
    col_title = lines[0].split(',')[col]
    pardict = {}   
    for aline in lines[1:]:
        alist = aline.split(',')
        if alist[col] in ['STOP', 'calculated']:
            pardict[alist[0]] = alist[col]
        else:
            pardict[alist[0]] = float(alist[col])
    return pardict, col_title

def fkT(T, a, b, c):
    #this function describes kcat Km vs temperature relations
    lnk =  a*np.log(T) + b/T + c
    k = np.exp(lnk)
    return k

def r_sphere(V):
    """
    Calculate r of V = 4/3 πr^3

    Parameters
    ----------
    V : Scalar.

    Returns
    -------
    r of a sphere/ball.

    """
    r = (0.75*V/np.pi)**(1/3)
    return r

#### loading data and store parameters
global par_title
#par_title =''

def sim_a_condition(parameter_file, column):
            
    #now getting the basic parameters stored in dictionary bpdict
    bpdict, par_title =get_parameter(parameter_file, col=column)

    ####store parameters in p object####
    class p(object):
        N  =  bpdict['N']     #number of cells/mL
        #volumes and dimensions    
        Ve = bpdict['Ve']   #volume of solution (cm3)
        Vc = bpdict['Vc']   #total volume of a single cell (cm3)
        reff = (3/4 * Vc/np.pi)**(1/3) #effective cellular radius, cm    
        r_bl = bpdict['r_bl'] * 1e-4 #effective surface thickness, µm to cm
        Vs = 4/3 * np.pi * (reff + r_bl)**3 - Vc #surface volume, cm3    
        Vp = bpdict['Vp']   #chloroplast stroma volume, cm3, ≈ 0.1 Vc
        Rp = r_sphere(Vp)   #chloroplast radius assuming spheric shape
        mRub  = bpdict['mRub']     #mols of RubisCO molecules per cell
        if bpdict['Rpyr'] =='calculated':
            Vy = mRub/8/0.000628*1e15 # unit in µm3
            Rpyr = r_sphere(Vy) # unit in µm
            #Based on 628 µM Rubisco [8mer] in Pyrenoid according to 
            #Cell 2017 Sep 21 171(1): 148-162.e19
        else:
            Rpyr = bpdict['Rpyr']
        Rpyr = Rpyr * 1e-4    #pyrenoid radius, µm to cm
        Npyr = bpdict['Npyr']           #number of pyrenoids per cell
        Vy = Npyr * 4/3 *np.pi *(Rpyr**3) #total pyrenoid volume, cm3  
        
        
        pH = bpdict['pHe']  #pH of assay buffer/extracellular pH
        pHc = bpdict['pHc'] #cytoplasmic pH
        pHp = bpdict['pHp'] #chloroplast stroma pH
        
        TC = bpdict['temp'] #temperature (in C on file), here stores as C
        T  = TC + 273.15    #temperature from C to K
        S  = bpdict['S']    #salinity
        H  = 10**-pH        #H+ of extracellular solution  
        Hc = 10**-pHc       #H+ concentration in the cytoplasm
        Hp = 10**-pHp       #H+ concentration in the stroma/pyrenoid
        
        #equilibrium constants for DIC, water, and borate
        pK1 = 3633.86/T - 61.2172 + 9.6777*np.log(T) - 0.011555*S + 1.152e-4 *S**2
        K1 = 10**-pK1          
        #K1 equilibrium constant between CO2 and HCO3 Lueker, Dickson, Keeling Mar Chem. 2000.
        pK2 = 471.78/T + 25.929 - 3.16967 * np.log(T) - 0.01781*S + 1.122e-4 *S**2
        K2 = 10**-pK2
        #K2 equilbrium constant from Lueker, Dickson, Keeling Mar Chem 2000
        Kw = np.exp(148.96502 - (13847.26 / T) - 23.6521 * np.log(T)\
                + (S**0.5)*((118.67 / T) - 5.977 + 1.0495 * np.log(T)) - 0.01615 * S) 
        #ion product of water, CO2 methods DOE 1994
        Kb = np.exp(((-8966.9 - 2890.53 * S**0.5 - 77.942 * S + 1.728 * S**1.5\
            - 0.0996 * S**2)/T) + 148.0248 + 137.1942 * S**0.5 + 1.62142 * S\
            -(24.4344 + 25.085 * S**0.5 + 0.2474 * S) * np.log(T) + 0.053105 * S**0.5 * T) 
        #Kb boric acid/borate equilibrium constant
        
        bfrac_e = (1/ (1+ K2/H))       # fraction of "B" pool that is HCO3- in extracellular solution
        bfrac_i = (1/ (1+ K2/Hc))       # fraction of "B" pool that is HCO3- in cytoplasm
        bfrac_x = (1/ (1+ K2/Hp))       # fraction of "B" pool that is HCO3-   in chloroplast and pyrenoid
        """
        Note that the equations above calculate HCO3- fractions of Bc_C, which 
        are not intended for total DIC. It is used for calculating CO2 production 
        rate from HCO3-. Bc_C is the concentration pool of HCO3- and CO3--
        see #CO2 hydration/HCO3- dehydration rates for how those are used 
        """
    
        #initial conditions for simulation  
        if bpdict['DIC'] == 'calculated':
            CO2 = bpdict['CO2'] #*1e-9         #1e-9 for µM to mol/cm3
            DIC = CO2/frac_C(K1, K2, H,'CO2') #total DIC
        else:
            DIC = bpdict['DIC'] #*1e-9        #1e-9 for µM to mol/cm3
        if bpdict['CO2'] =='calculated':
            CO2 = DIC * frac_C(K1, K2, H, 'CO2')
        else:
            CO2 = bpdict['CO2'] #*1e-9 #µM
    
        Bc_C = DIC - CO2        #pooled HCO3- + CO32-
        
        #kinetic pameters    
        ksf = bpdict['ksf']     #surface CA activity
        kcf = bpdict['kcf']     #cytoplasmic CA activity
        kpf = bpdict['kpf']     #chloroplast stroma CA activity
        kyf = bpdict['kyf']     #pyrenoid CA activity
    
        kp1 = np.exp(1246.98 - 6.19E4 / T - 183 * np.log(T))  # CO2 + H2O -> H+ + HCO3- Johnson 1982 as presented in Zeebe and Wolf-Gladrow
        kp4 = 4.70E7 * np.exp(-23.2 / (8.314E-3 * T))  # CO2 + OH- -> HCO3- Johnson 1982
        km1 = kp1/K1            # H+ + HCO3- -> CO2 + H2O
        km4 = kp4*Kw/K1         # HCO3- -> CO2  + OH-
        kph5 = 5.0E10           # CO32- + H+ -> HCO3- /(M s) Zeebe and Wolf Gladrow
        kmh5 = kph5 * K2        # HCO3- -> CO32- + H+
        kpoh5 = 6.0E9           # HCO3- + OH- -> CO32- + H2O Zeebe and Wolf-Gladrow
        kmoh5 = kpoh5 * Kw / K2 # CO32- + H2O -> HCO3- + OH-    
        
        #CO2 hydration/HCO3- dehydration rates
        if bpdict['kuf'] =='calculated':        
            kuf = kp1 + kp4 * (Kw/H)      #CO2 hydration rate in bulk solution
            kur = bfrac_e * kuf * (H/K1)  #HCO3- dehyration rate in bulk solution
        else:
            kuf = bpdict['kuf']       #CO2 hydration rate in bulk solution
            kur = bpdict['kur']       #HCO3- dehyration rate in bulk solution
        
        if ksf == 'calculated':      #when kuf&ksf are not measured
            ksf = kuf
        Ke = kuf/kur             # extracellular CO2-->HCO3- simple equilibrium constant
        Kc = 10**(pHc - pH) * Ke # cytosolic simple equilibrium constant
        Kp = 10**(pHp - pH) * Ke # chloroplast stroma simple equilibrium constant
        
        ksr = ksf / Ke
        kcr = kcf / Kc
        kpr = kpf / Kp
        kyr = kyf / Kp
        # ksr = bfrac_e * ksf * (H/K1)  #HCO3- dehydration rate in surface layer (cm3/s)
        # kcr = bfrac_i * kcf * (Hc/K1) #HCO3- dehyration rate in cytoplasm (/s)
        # kpr = bfrac_x * kpf * (Hp/K1) #HCO3- dehyration rate in chloroplast stroma (/s)
        # kyr = bfrac_x * kyf * (Hp/K1) #HCO3- dehyration rate in pyrenoid (/s)
        
        #diffusion coefficients and mass transfer coefficients
        # DCF = diff_coef(TC,S, 1)
        # #diffusion coefficients (in cm2/s) at current temp, salinity, final variable is pressure: assume 1 atm.
        # Dc  = DCF['CO2']        #difffusivity of CO2 in cm2/s
        # Db  = DCF['HCO3-']       #diffusivity of HCO3- in cm2/s
        DCF = Di(TC, S)
        Dc, Db = DCF['CO2'], DCF['HCO3-']
        
        fc    = bpdict['fc'] # can be measured from experiments
        fb    = bpdict['fb'] # can be measured from experiments
        #additional mass transfer coefficients
        if bpdict['fc_bl'] =='calculated':
            fc_bl = 4 * np.pi * Dc * (reff + r_bl)  #parameter for diffusive CO2 flux to cell surface layer
            fb_bl = 4 * np.pi * Db * (reff + r_bl)  #parameter for diffusive HCO3- flux to cell surface layer
        else:
            fc_bl, fb_bl = bpdict['fc_bl'], bpdict['fb_bl']
        
        # 1/fc = 1/fc_bl + 1/fc_sm
        fc_sm = 1/(1/fc - 1/fc_bl) #MTC for CO2 across cytoplasmic membrane
        fb_sm = 1/(1/fb - 1/fb_bl) #MTC for HCO3- across cytoplasmic membrane, generally set to zero
    
        if bpdict['fb_sm'] != 'calculated':
            fb_sm = bpdict['fb_sm'] #MTC for HCO3- across cytoplasmic membrane, generally set to zero
        if bpdict['fc_p'] == 'calculated':
            fc_p = 4 * np.pi * Dc * Rp
        else:
            fc_p = bpdict['fc_p']   #MTC for CO2 into chloroplast, from Hopkinson et al. 2011
        fb_p = bpdict['fb_p']       #MTC for HCO3- into chloroplast, generally set to zero
        
        #now pyrenoid considering the physical boundary slowing diffusion
        pyr_factor = bpdict['pyr_factor']
        if bpdict['fc_y'] == 'calculated':
            fc_y = Npyr * 4 * np.pi * Dc * Rpyr * pyr_factor
        else:
            fc_y = bpdict['fc_y']       #MTC for CO2 into pyrenoid, from Hopkinson et al. 2011
        fb_y = Npyr * 4 * np.pi * Db * Rpyr * pyr_factor     #MTC for HCO3- into pyrenoid.
       
        
        ####it may be a good idea to curate the dimension numbers####
        """
        In the supplemental file for New Phyto paper, the cytoplasmic/cellular volume is set at 2.5e-9 cm3
        Also the volume calculation may be wrong too, since the shape of Fc isn't strictly a sphere
        See "Biovolumes and Size-Classes of Phytoplankton in the Baltic Sea ----Baltic Sea Eviro. Proceedings No. 106"
        Cell volume can change, data is best from measurements
        """ 
        #enzyme kinetics
        #### here are the critical data to see temperature differences ####   
        #curve fit parameters of kcat and Km, species dependent    
        popt_kcat = np.array([  -16.45485721, -12156.61863416,  135.72669369])
        popt_Km   = np.array([  -16.45485721, -9891.56251537,   130.94186378])
    
        if bpdict['kcat_R'] =='calculated':
            kcat_R = fkT(T, *popt_kcat)     #T in unit of Kelvin
        else:
            kcat_R= bpdict['kcat_R']        #Rubisco turnover rate (/s)
        if bpdict['Km_R'] =='calculated':
            Km_R = fkT(T, *popt_Km)         #T in unit of Kelvin
        else:
            Km_R = bpdict['Km_R']           #Rubisco Km (uM)
        Km_R = Km_R #* 1e-9                  #uM to mol/cm3
        
    
        Vm_Bc = bpdict['Vm_Bc']          #maximum uptake rate of cytoplasmic HCO3- transporter in mol/cell/s
        Km_Bc = bpdict['Km_Bc']# *1e-9    #Km for cytoplasmic HCO3- transporter (µM to mol/cm3)
        Vm_Bp = bpdict['Vm_Bp']          #maximum uptake rate of chloroplast pump HCO3- transporter in mol/cell/s
        Km_Bp = bpdict['Km_Bp']# *1e-9    #Km for chloroplast pump HCO3- transporter (µM to mol/cm3)    
        
        RS = bpdict['RS']   #respiration rate from measurements
        #### the following converts the unit of ksf and ksr to s^-1 from cm3/s in matlab code
        #when ksf is expressed in cm3/s, the value is very small, much smaller than kuf
        if ksf < kuf:
            ksf = ksf / Vs
            ksr = ksr / Vs
    
    class sim_constants(object):
        def __init__(self):
            C = p()
            self.Vm_Bc = C.Vm_Bc
            self.Km_Bc = C.Km_Bc
            self.Vm_Bp = C.Vm_Bp
            self.Km_Bp = C.Km_Bp
            self.mRub = C.mRub
            self.kcat_R = C.kcat_R
            self.Km_R = C.Km_R
            self.kuf = C.kuf        
            self.kur = C.kur        
            self.ksf = C.ksf  #new parameter
            self.ksr = C.ksr  #new parameter          
            self.kcf = C.kcf
            self.kcr = C.kcr
            self.kpf = C.kpf
            self.kpr = C.kpr
            self.kyf = C.kyf
            self.kyr = C.kyr
            self.N = C.N
            self.Ve = C.Ve        
            self.Vs = C.Vs    #new parameter        
            self.Vc = C.Vc
            self.Vp = C.Vp
            self.Vy = C.Vy        
            self.fc_bl = C.fc_bl   #new parameter
            self.fb_bl = C.fb_bl   #new parameter
            self.fc_sm = C.fc_sm   #new parameter
            self.fb_sm = C.fb_sm   #new parameter        
            self.fc_p = C.fc_p
            self.fb_p = C.fb_p
            self.fc_y = C.fc_y
            self.fb_y = C.fb_y
            self.RS = C.RS
        def to_tuples(self):
            constants = (self.Vm_Bc, self.Km_Bc, self.Vm_Bp, self.Km_Bp, self.mRub,\
                         self.kcat_R, self.Km_R, self.kuf,self.kur, self.ksf,self.ksr,\
                         self.kcf,self.kcr, self.kpf,self.kpr, self.kyf,self.kyr, \
                         self.N, self.Ve, self.Vs, self.Vc, self.Vp, self.Vy, \
                         self.fc_bl, self.fb_bl, self.fc_sm, self.fb_sm,\
                         self.fc_p,self.fb_p, self.fc_y,self.fb_y, self.RS)
            return constants
    
        def to_dict(self):
            d={"Vm_Bc":self.Vm_Bc, "Km_Bc":self.Km_Bc, "Vm_Bp": self.Vm_Bp, \
               "Km_Bp":self.Km_Bp, "mRub":self.mRub, "kcat_R": self.kcat_R, \
               "Km_R":self.Km_R, "kuf":self.kuf,"kur": self.kur, "ksf":self.ksf,\
               "ksr":self.ksr, "kcf":self.kcf, "kcr": self.kcr, "kpf": self.kpf,\
               "kpr":self.kpr, "kyf":self.kyf, "kyr": self.kyr, "N": self.N,\
               "Ve":self.Ve, "Vs":self.Vs, "Vc":self.Vc, "Vp":self.Vp, "Vy":self.Vy,\
               "fc_bl": self.fc_bl, "fb_bl":self.fb_bl, "fc_sm":self.fc_sm, "fb_sm":self.fb_sm,\
               "fc_p": self.fc_p, "fb_p": self.fb_p, "fc_y": self.fc_y, "fb_y": self.fb_y,\
                   "RS": self.RS}
            return d
    
    
    #the following assign intial values, replacing initcond.m, CO2 and Bc_C saved in p
    #y0 =  np.array([p.CO2, p.Bc_C, p.CO2, p.Bc_C, p.CO2, p.Bc_C, p.CO2, p.Bc_C, p.CO2, p.Bc_C],dtype = float)
    y0 =  np.array([p.CO2, p.Bc_C, p.CO2, p.Bc_C, p.CO2, p.CO2/p.Kc, \
                               p.CO2, p.CO2/p.Kp, p.CO2, p.CO2/p.Kp],dtype = float)
    cons = sim_constants().to_tuples()#store constants as tuple
    def odefun(t,y):
        return f_Ci(y,t, *cons)
    #the wrapper (odefun) passes all cons to f_Ci
    
    
    tbrks = [0, 1800] # simulation start t and end t; can be multi-section as in Chlp_pump
    for i in range(len(tbrks)-1):
        #t = np.linspace(tbrks[i],tbrks[i+1],int((tbrks[i+1]-tbrks[i])*100)+1)
        #the factor 100 determines each ∂t is 0.01s, 
        #it can be adjusted for faster computing or smoother curve
        sol = solve_ivp(odefun,[tbrks[i], tbrks[i+1]],y0, method = "BDF", \
                         rtol=1e-6, atol=1e-10, max_step=5)#t_eval=t, let ivp determine t
        #plt.plot(sol.t,sol.y[0,:]) #uncomment to see the each tbrk.
        #the following notes are Matlab codes
        #options = odeset('RelTol', 1E-6, 'AbsTol', 1E-10,'MaxStep',5);
        #[t_ode, Ys] = ode15s(@Cideriv, tspan, yinit, options, p);
        y0 = sol.y[:,-1] # assign new y0
        # if i < len(p.DIC):
        # #python starts from 0, len(p.DIC) == len(p.ADD), when maximum i =9, 
        # #it is okay in matlab, but out of range in python, so "<" instead of "<=".
        #     y0[1] = y0[1] + p.DIC[i]
        if i == 0:
            t_all = sol.t
            Y = sol.y
        else:
            t_all = np.concatenate((t_all, sol.t))
            Y = np.concatenate((Y, sol.y,), axis=1)#Y,sol.y are (10,N) 2darrays in Antiatom model
    
    ####====now plot modeling/simulation results====####
    Fig1 = plt.figure(1, figsize =(6.4,3),dpi = 300)
    plt.rcParams.update({'font.size': 6})
    splots = [1,6,2,7,3,8,4,9,5,10]#subplot numbers
    titles = ['$CO_2$ exc','$HCO_3$$^-$ exc','$CO_2$ srf','$HCO_3$$^-$ srf','$CO_2$ cyt','$HCO_3$$^-$ cyt',\
              '$CO_2$ chp','$HCO_3$$^-$ chp','$CO_2$ pyr','$HCO_3$$^-$ pyr']
    colors = ['m','b']*5
    max_CO2 = max(np.max(Y[0,:]),np.max(Y[2,:]),np.max(Y[4,:]),np.max(Y[6,:]),\
                  np.max(Y[8,:]))
    max_Bic = max(np.max(Y[1,:]),np.max(Y[3,:]),np.max(Y[5,:]),np.max(Y[7,:]),\
                  np.max(Y[9,:]))
    
    
    for i in range(10):
        ax = Fig1.add_subplot(2,5,splots[i])
        ax.set_title(titles[i], color = colors[i])
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.tick_params(pad=0.5, labelsize=6)
        ax.plot(t_all,Y[i,:],colors[i],linewidth=0.5)
        if i in [1,3,5,7,9]:
            ax.set_xlabel('time/s')
            ax.set_ylim(0, 1.05*max_Bic)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_ylim(0, 1.05*max_CO2)
        if i in [0,1]:
            ax.set_ylabel('µM')
        ax.set_xlim(0, 1.05*np.max(t_all))       
    plt.tight_layout()
    plt.show()
    #plt.close()
    
    
    
    #### ====now calculate some params====####
    
    cycles = 1#len(t_steady)
    Yss = np.zeros((10,1))
    Yss[:,0] = np.mean(Y[:,-5:],axis=1) #steady states numbers,
    # Yss[:,0] = Y[:,-1] #steady states numbers,
    
    DIC_list = ['CO2_exc','Bic_exc','CO2_srf','Bic_srf','CO2_cyt','Bic_cyt',\
                'CO2_chp','Bic_chp','CO2_pyr','Bic_pyr']
    DICs ={} #DIC steady states, dictionary
    for i in range(10):
        DICs[DIC_list[i]] = Yss[i,:][0] #if multiple, remember to change it back to Y[i,:] for array
    
    cdict = sim_constants().to_dict() #constants in dictionary
    Fluxes = CiFluxes(DICs, cdict) #diffusional
    
    #NetCO2influx = cdict['fc_bl'] * (DICs['CO2_exc'] - DICs['CO2_srf']) \
    #               + Fluxes['dehyd_srf'] - Fluxes['hyd_srf'] #flux of CO2 to surface≈Cup_cyt
    PS_rate = cdict['mRub']*(cdict['kcat_R'] * DICs['CO2_pyr'])/(cdict['Km_R'] + DICs['CO2_pyr'])#photosynthetic rate
    net_PS  = PS_rate - p.RS
    Cup_cyt = Fluxes['Csrf_cyt']#cdict['fc_sm'] * (DICs['CO2_srf'] - DICs['CO2_cyt']) #net CO2 uptake red
    Bup_cyt = cdict['Vm_Bc'] * DICs['Bic_srf']/(cdict['Km_Bc'] + DICs['Bic_srf'])  #HCO3- uptake into cell
    #Bup_cyt = cdict['Vm_Bc'] * DICs['Bic_exc']/(cdict['Km_Bc'] + DICs['Bic_exc'])  #HCO3- uptake into cell as in Matlab
    Bup_chp = cdict['Vm_Bp'] * DICs['Bic_cyt']/(cdict['Km_Bp'] + DICs['Bic_cyt'])  #chloroplast pump HCO3- transport rate
    #deh_exc = Fluxes['dehyd_exc']/cdict['Ve']
    #In the matlab code, there is a flux.Active varialbe includes attributes of Bup_cyt, Bup_chp, and PS_rate
    newfluxes = [PS_rate, net_PS, Cup_cyt, Bup_cyt, Bup_chp]
    for i, akey in enumerate( ['PS_rate', 'net_PS', 'Cup_cyt', 'Bup_cyt', 'Bup_chp']):
        Fluxes[akey] = newfluxes[i]
    #### ====now calculate the C13 differences ====####
    
    dC13bic = 0; #del 13 C of HCO3-
    dC13CO2diff = 24.12 - 9866/p.T #CO2C13offset calculates equilibrium CO2 fractionation offset from HCO3- as a function of tem
    #Zeebe and Wolf Gladrow 2001; from Mook 1986, Zhang 1995 is similar
    dC13CO2 = dC13bic + dC13CO2diff
    delta = [dC13CO2, dC13bic]
    
    C13data = C13frac(delta,Fluxes, cdict, Yss, parameter_file, par_title)
    
    ####====now save the modeling result into csv files====####
    np.savetxt(parameter_file.replace('T.csv', par_title+'_pyOUT.csv'),\
               np.column_stack((t_all,Y.transpose())),fmt = '%1.4e', \
               header=','.join(['t/s']+DIC_list), delimiter=',', comments ='')
    d_to_csv(DICs,parameter_file.replace('T.csv', par_title+'_DICs_steadystate.csv'))
    d_to_csv(Fluxes, parameter_file.replace('T.csv', par_title+'_DIC_net_Fluxes.csv'))
        
    ####====save figs to pdf file====####
    pp = PdfPages(parameter_file.replace('T.csv', par_title+'_model_plots.pdf'))
    for fig in [Fig1]:
        pp.savefig(fig)
    pp.close()
    plt.close('all')

parameter_file = 'Pt_T.csv'
for i in range(1,4):
    sim_a_condition(parameter_file, i)
