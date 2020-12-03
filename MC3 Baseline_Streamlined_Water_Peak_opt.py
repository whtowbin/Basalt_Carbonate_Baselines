# %%
import sys
import numpy as np
import mc3
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from pathlib import Path
%matplotlib inline

#%%

# %%
# Loads One of Sarah's Spectra. We can easily modify this to load them all like I have done with Dan's below
wn_high = 2400
wn_low = 1250
Sarah_path = Path.cwd().joinpath("Sarah's FTIR Spectra/Fuego2018FTIRSpectra_Transmission/AC4_OL49_021920_30x30_H2O_b.CSV")

def open_spectrum(path, wn_high=wn_high, wn_low=wn_low):
    df = pd.read_csv(path, index_col=0, header=0,
                     names=['Wavenumber', 'Absorbance'])
    spec = df.loc[wn_low:wn_high]
    return spec

Spectrum = open_spectrum(Sarah_path)

Spec_name = 'AC4_OL49'

#This line subtracts the mean from your data
#sarahFTIR = StandardScaler(with_std=False).fit_transform(Spectrum)
sarahFTIR = Spectrum
spec = sarahFTIR.to_numpy()[:,0]
data = spec

uncert = np.ones_like(spec)*0.001 #change multiplier to change error for whole spectrum 

# %%
Nvectors = 5
ALT_PCA_DF = pd.read_csv("Devol_Baseline_Avg+PCA.csv", index_col= "Wavenumber")

PCA_DF = pd.read_csv("Baseline_Avg+PCA.csv", index_col= "Wavenumber")

Wavenumber = np.array([PCA_DF.index])

PCAmatrix = np.matrix(PCA_DF.to_numpy())
x = Wavenumber

Peak_1635_PCA = pd.read_csv("Water_Peak_1635.csv", index_col= "Wavenumber")
Peak_1635_PCAmatrix = np.matrix(Peak_1635_PCA.to_numpy())
# peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, peak_L1635, width_L1635, L1635_amplitude, slope, offset = P[Nvectors:None]
# %%


def Gauss(x, mu, sd, A=1):
    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))
    return G

def Lorentzian(x, center, half_width, amp=1):
    L = amp* (half_width**2/(half_width**2 + (2*x - 2*center)**2))
    return L


#%%
def linear2(x, tilt, offset):
    offset = np.ones_like(x) * offset
    tilt = np.arange(0, max(x.shape)) * tilt 
    tilt_offset = tilt + offset 
    return tilt_offset

# %%

def carbonate(P, x, PCAmatrix, Peak_1635_PCAmatrix,  Nvectors=5): 
    
    PCA_Weights = np.array( [P[0:Nvectors]] )

    Peak_Weights = np.array( [P[-4:None]] )

    peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, slope, offset = P[Nvectors:-4]

    Peak_1635 = Peak_Weights * Peak_1635_PCAmatrix.T   

    G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = linear2(x,slope, offset) 

    baseline =  PCA_Weights * PCAmatrix.T
    model_data = baseline + linear_offset + G1515 + G1430 + Peak_1635

    return np.array(model_data)[0,:]

# Parameter names: Average_bsl, PCA1,PCA2, PCA3, PCA4, peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, slope, offset, Average_1635Peak, 1635PeakPCA1, 1635PeakPCA2, 1635PeakPCA3
Test_Params =                 [1.0, 0.1 , 0.5, 0.5, 0.001, 1430.0, 30, 0.1, 1515, 30, 0.1, 7e-5, 0.65, 1.0, 0.01, 0.01, 0.01 ]
# Now I need a funtion to map my funciton to this. 

# %%
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Define the modeling function as a callable:
func = carbonate

# List of additional arguments of func (if necessary):
indparams = [x, PCAmatrix, Peak_1635_PCAmatrix, Nvectors]
# Parameter names: Average_bsl, PCA1,PCA2, PCA3, PCA4, peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, slope, offset, Average_1635Peak, 1635PeakPCA1, 1635PeakPCA2, 1635PeakPCA3


# Array of initial-guess values of fitting parameters:

P =             [1.0, 0.1 , 0.5, 0.5, 0.001, 1430.0, 30, 0.1, 1515, 30, 0.1, 7e-5, 0.65, 1.0, 0.01, 0.01, 0.01 ]
params = np.float64(np.array(P))
# Lower and upper boundaries for the MCMC exploration:
pmin = np.array([0.0, -10.0 , -5, -5, -5, 1420.0, 25, 0.001, 1500, 25, 0.01, -5e3, -100.0, 0.0, -2.0, -0.5, -0.5 ])
pmax = np.array([50.0, 10.0 ,  5,  5,  5, 1435.0, 45, 0.5,   1535, 45,  0.5,  5e3, 100.0, 5.0, 2.0, 0.5, 0.5  ])


# Parameters' stepping behavior:
pstep = 0.01* params

# Parameter prior probability distributions:
# TODO Figure out how to use the Prior function better and how it differs from params. 
prior   =  params
priorlow = np.abs(params*0.5)
priorup  = np.abs(params*0.5)

# Parameter names:
# TODO Rename these to be better names for figures and shorten with Latex
pnames   =  ['Avg_BsL', "PCA_1", "PCA_2", "PCA_3", "PCA_4",'peak_G1430', 'std_G1430','G1430_amplitude', 'peak_G1515', 'std_G1515', 'G1515_amplitude', 'slope', 'offset', 'Average_1635Peak', '1635PeakPCA1', '1635PeakPCA2', '1635PeakPCA3']
texnames = ['Avg_BsL', "PCA_1", "PCA_2", "PCA_3", "PCA_4",'peak_G1430', 'std_G1430', 'G1430_amplitude', 'peak_G1515', 'std_G1515', 'G1515_amplitude', 'slope', 'offset', 'Average_1635Peak', '1635PeakPCA1', '1635PeakPCA2', '1635PeakPCA3']

# Sampler algorithm, choose from: 'snooker', 'demc' or 'mrw'.
sampler =   'snooker'

# MCMC setup:
nsamples =  1e5
burnin   = 1000
nchains  =   9
ncpu     =    3
thinning =    1

# MCMC initial draw, choose from: 'normal' or 'uniform'
kickoff = 'normal'
# DEMC snooker pre-MCMC sample size:
hsize   = 10

# Optimization before MCMC, choose from: 'lm' or 'trf':
leastsq    = 'lm'
chisqscale = False

# MCMC Convergence:
grtest  = True
grbreak = 1.01
grnmin  = 0.5

# Logging:
log = 'MCMC_tutorial.log'

# File outputs:
savefile = 'MCMC_tutorial.npz'
plots    =  False
rms      = True

# Carter & Winn (2009) Wavelet-likelihood method:
wlike = False

# Run the MCMC:
# Priors: pmin=pmin, pmax=pmax, arent working fix this..


mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
     indparams=indparams, pstep=pstep, 
     pmin=pmin, pmax=pmax,
     #priorlow=priorlow, priorup=priorup, 
     pnames=pnames, texnames=texnames,
     sampler=sampler, nsamples=nsamples,  nchains=nchains,
     ncpu=ncpu, burnin=burnin, thinning=thinning,
     leastsq=leastsq, chisqscale=chisqscale,
     grtest=grtest, grbreak=grbreak, grnmin=grnmin,
     hsize=hsize, kickoff=kickoff,
     wlike=wlike, log=log,
     plots=plots, savefile=savefile, rms=rms)


#%%
#

# %%
# plot baseline model results and data

fig, ax = plt.subplots(figsize= (12,6))
plt.plot(x[0,:],carbonate(mc3_output['meanp'], x, PCAmatrix, Peak_1635_PCAmatrix, Nvectors))
plt.plot(x[0,:],data)
ax.invert_xaxis()
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
#%%
# Prepare to plot all peaks
pca_results = np.array([mc3_output['meanp'][0:Nvectors]])
Baseline_Solve = pca_results * PCAmatrix.T
line_results = mc3_output['meanp'][-6:-4]
line = linear2(x, line_results[0], line_results[1])
Baseline_Solve = Baseline_Solve + line

peak_1635_weights = mc3_output['meanp'][-4:None]
peak_G1515, std_G1515, G1515_amplitude = mc3_output['meanp'][-9:-6]
peak_G1430, std_G1430, G1430_amplitude = mc3_output['meanp'][-12:-9]

Peak_1635 = peak_1635_weights * Peak_1635_PCAmatrix.T 
G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

# %%
#Plot Peaks
fig, ax = plt.subplots(figsize= (12,6))
plt.plot(x[0,:], data)
plt.plot(x[0,:], Peak_1635.T + Baseline_Solve.T )
plt.plot(x[0,:], G1515.T + Baseline_Solve.T)
plt.plot(x[0,:], G1430.T + Baseline_Solve.T)
plt.plot(x[0,:], carbonate(mc3_output['meanp'], x, PCAmatrix, Peak_1635_PCAmatrix, Nvectors))
plt.plot(x[0,:], Baseline_Solve.T)

plt.annotate(f"1635 cm $^{{- 1}}$  Peak Height: {Peak_1635.max():.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.95), xycoords = 'axes fraction')
plt.annotate(f"1515 cm $^{{- 1}}$  Peak Height: {G1515_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.90), xycoords = 'axes fraction')
plt.annotate(f"1430 cm $^{{- 1}}$  Peak Height: {G1430_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.85), xycoords = 'axes fraction')

plt.title(f'{Spec_name}')

ax.invert_xaxis()
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')


# %%
fig, ax = plt.subplots(figsize= (12,6))

plt.plot(x[0,:], Peak_1635.T)
plt.plot(x[0,:], G1515.T )
plt.plot(x[0,:], G1430.T )
#plt.plot(x[0,:], carbonate(mc3_output['meanp'], x, PCAmatrix, Peak_1635_PCAmatrix, Nvectors))
#plt.plot(x[0,:], Baseline_Solve.T)

plt.annotate(f"1635 cm $^{{- 1}}$  Peak Height: {Peak_1635.max():.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.95), xycoords = 'axes fraction')
plt.annotate(f"1515 cm $^{{- 1}}$  Peak Height: {G1515_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.90), xycoords = 'axes fraction')
plt.annotate(f"1430 cm $^{{- 1}}$  Peak Height: {G1430_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.85), xycoords = 'axes fraction')

plt.title(f'{Spec_name}')
plt.xlim(1250,1800)
ax.invert_xaxis()
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
# %%

"""

baseline =  PCAmatrix * PCA_Weights
    model_data = baseline + linear_offset + G1515 + G1430 

PCAmatrix is avg, pca1, pca2, pca3, pca4
Fit params come in this form. must be transformed to work as the input
Baseline_Matrix = pd.concat([Average_baseline, PCA_DF, offset, tilt, Peak2, Peak3, Peak1], axis=1)

What to do here??? WOuld be good to set up all of these as functions. 
I can string together these functions 
"""
# %%

# %%
