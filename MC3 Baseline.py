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
uncert = np.ones_like(spec)*0.01
error  = np.random.normal(0, uncert)

#%%
# Dans Spectrum 
Path_Dan_test = Path.cwd().joinpath("Dans_Examples")
all_files = Path_Dan_test.rglob("*.csv")
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=0, header=0, names=['Wavenumber', 'Absorbance']
                     )
    li.append(df)


Dan_FTIR = pd.concat(li, axis=1, )

Dan_FTIR_select = Dan_FTIR.loc[wn_low:wn_high]
Spectrum2 = Dan_FTIR_select.iloc[:, 1].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
data = Spectrum2[:,0]
Spec_name = 'Dan Cleveland Sample'
P_spect2 = [5.50728127e+00, -6.83405880e-01, -1.10914328e+00, 1.18137238e-01, -1.79234709e-01, 1430, 30, 0.2, 1515, 30, 0.2, 4.79900159e-04, 2.36431464e-01]
# %%
#P_spect2 = [5.50728127e+00, -.5, -.8, 0, 0, 1430, 30, 0.2, 1515, 30, 0.2, 0, 0.4]

dan_out = carbonate(P_spect2, x, PCAmatrix, Nvectors)
plt.plot(x[0,:],dan_out)
plt.plot(x[0,:],Spectrum2[:,0])

# %%
Nvectors = 5
#PCA_DF = pd.read_csv("PCA_Matrix.CSV", index_col= "Wavenumber")
PCA_DF = pd.read_csv("Baseline_Avg+PCA.csv", index_col= "Wavenumber")

Wavenumber = np.array([PCA_DF.index])

PCAmatrix = np.matrix(PCA_DF.to_numpy())#[:,0:Nvectors]
x = Wavenumber
# p1 = [PCA_Weights, peak_G1430, halfwidth_G1430, peak_G1515, halfwidth_G1515, amplitude, slope, offset]
#P = [7.73646458, -9.52374283e-01, -8.28969527e-01,1.72901441e-01, -2.32678933e-01, 1430, 30, 0.1, 1515, 30, 0.1, 9e-5, 0.65]

# peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, peak_L1635, width_L1635, L1635_amplitude, slope, offset = P[Nvectors:None]
# %%
def linear(x,m,b):
    x = x- x.min()
    line = x*m + b
    return line

def Gauss(x, mu, sd, A=1):
    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))
    return G

def Lorentzian(x, center, half_width, amp=1):
    L = amp* (half_width**2/(half_width**2 + (2*x - 2*center)**2))
    return L

#Peak1 = pd.Series( Lorentzian(x=Wavenumber, center=1635, half_width=55, amp=1),index=Wavenumber)
#%%
def linear2(x, tilt, offset):
    offset = np.ones_like(x) * offset
    tilt = np.arange(0, max(x.shape)) * tilt 
    tilt_offset = tilt + offset 
    return tilt_offset
# %%
tilt_offset =linear2(x[0,:], 1, 0)

lin_x = linear(x[0,:],1,0)
# %%
def carbonate(P, x, PCAmatrix, Nvectors=5): # add terms M and X, PCA_fit terms
    
    PCA_Weights = np.array( [P[0:Nvectors]] )

    #peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, slope, offset = P[Nvectors:None]
    peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, peak_L1635, width_L1635, L1635_amplitude, slope, offset = P[Nvectors:None]

    L1635 = Lorentzian(x, center=peak_L1635, half_width=width_L1635, amp=L1635_amplitude)
    G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = linear2(x,slope, offset) 

    baseline =  PCA_Weights * PCAmatrix.T
    model_data = baseline + linear_offset + G1515 + G1430 + L1635
    #dict = {'baseline': baseline,  'linear_offset': linear_offset, 'G1515': G1515, 'G1430': G1430}
    return np.array(model_data)[0,:]
    #return dict
# %%
# Synthetic data
#uncert = np.ones_like(Wavenumber[0,:])*0.01
#error  = np.random.normal(0, uncert)
#data   = carbonate(P, x, PCAmatrix, Nvectors) + error
# %%
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Define the modeling function as a callable:
func = carbonate

# List of additional arguments of func (if necessary):
indparams = [x, PCAmatrix, Nvectors]
# Average_bsl, PCA1,PCA2, PCA3, PCA4, peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, peak_L1635, width_L1635, L1635_amplitude, slope, offset


# Array of initial-guess values of fitting parameters:

#params = np.float64(np.array([7.73646458, -9.52374283e-01, -8.28969527e-01,1.72901441e-01, -2.32678933e-01, 1430, 30, 0.1, 1515, 30, 0.1, 7e-5, 0.65]))
#params = np.float64(np.array([3.73646458, -3.52374283e-01, -3.28969527e-01,0.72901441e-01, -0.32678933e-01, 1428, 25, 0.12, 1513, 25, 0.12, 7e-5, 0.65]))
# Lower and upper boundaries for the MCMC exploration:
P =           [1.0, 0.1 , 0.5, 0.5, 0.001, 1430.0, 30, 0.1, 1515, 30, 0.1, 1635.0, 55.0, 0.5, 7e-5, 0.65 ]
pmin = np.array([0.0, -10.0 , -5, -5, -5, 1425.0, 25, 0.001, 1510, 25, 0.01, 1630.0, 40.0, 0.01, 5e-10, 0.0 ])
pmax = np.array([50.0, 10.0 ,  5,  5,  5, 1435.0, 35, 0.5,   1520, 40,  0.5, 1640.0, 70,  1.0,  5e-2, 2.0 ])
params = np.float64(np.array(P))

#pmin = params - params*20
#pmax = params + params*20
# Parameters' stepping behavior:
pstep = 0.01* params
#np.abs(params*0.05)

# Parameter prior probability distributions:
prior   =  params
priorlow = np.abs(params*0.5)
priorup  = np.abs(params*0.5)

# Parameter names:
pnames   =  ['Avg_BsL', "PCA_1", "PCA_2", "PCA_3", "PCA_4",'peak_G1430', 'std_G1430','G1430_amplitude', 'peak_G1515', 'std_G1515', 'G1515_amplitude', 'peak_L1635', 'width_L1635', 'L1635_amplitude', 'slope', 'offset']
texnames = ['Avg_BsL', "PCA_1", "PCA_2", "PCA_3", "PCA_4",'peak_G1430', 'std_G1430', 'G1430_amplitude', 'peak_G1515', 'std_G1515', 'G1515_amplitude', 'peak_L1635', 'width_L1635', 'L1635_amplitude', 'slope', 'offset']

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
plots    = True # False
rms      = True

# Carter & Winn (2009) Wavelet-likelihood method:
wlike = False

# Run the MCMC:
# Priors: pmin=pmin, pmax=pmax, arent working fix this..


mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
     indparams=indparams, pstep=pstep, 
     pmin=pmin, pmax=pmax,
     priorlow=priorlow, priorup=priorup, 
     pnames=pnames, texnames=texnames,
     sampler=sampler, nsamples=nsamples,  nchains=nchains,
     ncpu=ncpu, burnin=burnin, thinning=thinning,
     leastsq=leastsq, chisqscale=chisqscale,
     grtest=grtest, grbreak=grbreak, grnmin=grnmin,
     hsize=hsize, kickoff=kickoff,
     wlike=wlike, log=log,
     plots=plots, savefile=savefile, rms=rms)

# pstep=pstep
# pmin=pmin, pmax=pmax,
#data = Spectrum.to_numpy()[:,0]
#uncert = np.ones_like(x)*0.0001
"""
# pstep=pstep
mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
     indparams=indparams, pmin=pmin, pmax=pmax, pstep=pstep,
     pnames=pnames, texnames=texnames, priorlow=priorlow, priorup=priorup,
     sampler=sampler, nsamples=nsamples,  nchains=nchains,
     ncpu=ncpu, burnin=burnin, thinning=thinning,
     leastsq=leastsq, chisqscale=chisqscale,
     grtest=grtest, grbreak=grbreak, grnmin=grnmin,
     hsize=hsize, kickoff=kickoff,
     wlike=wlike, log=log,
     plots=plots, savefile=savefile, rms=rms)
"""
#%%
#

# %%
# plot results
fig, ax = plt.subplots(figsize= (12,6))
plt.plot(x[0,:],carbonate(mc3_output['meanp'], x, PCAmatrix, Nvectors))
plt.plot(x[0,:],data)
ax.invert_xaxis()
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
#%%

pca_results = np.array([mc3_output['meanp'][0:Nvectors]])
Baseline_Solve = pca_results * PCAmatrix.T
line_results = mc3_output['meanp'][-2:None]
line = linear2(x, line_results[0], line_results[1])
Baseline_Solve = Baseline_Solve + line

peak_L1635, width_L1635, L1635_amplitude = mc3_output['meanp'][-5:-2]
peak_G1515, std_G1515, G1515_amplitude = mc3_output['meanp'][-8:-5]
peak_G1430, std_G1430, G1430_amplitude = mc3_output['meanp'][-11:-8]

L1635 =  Lorentzian(x, center=peak_L1635, half_width=width_L1635, amp=L1635_amplitude)
G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

# %%
fig, ax = plt.subplots(figsize= (12,6))
plt.plot(x[0,:], data)
plt.plot(x[0,:], L1635.T )
plt.plot(x[0,:], G1515.T)
plt.plot(x[0,:], G1430.T )
plt.plot(x[0,:], carbonate(mc3_output['meanp'], x, PCAmatrix, Nvectors))
plt.plot(x[0,:], Baseline_Solve.T)

plt.annotate(f"1635 cm $^{{- 1}}$  Peak Height: {L1635_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.95), xycoords = 'axes fraction')
plt.annotate(f"1515 cm $^{{- 1}}$  Peak Height: {G1515_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.90), xycoords = 'axes fraction')
plt.annotate(f"1430 cm $^{{- 1}}$  Peak Height: {G1430_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.85), xycoords = 'axes fraction')

plt.title(f'{Spec_name}')

ax.invert_xaxis()
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')

# %%

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