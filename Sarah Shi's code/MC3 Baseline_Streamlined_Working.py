# %%
import sys
import glob

import numpy as np
import pandas as pd

import mc3
from numpy.core.fromnumeric import shape
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from pathlib import Path

from matplotlib import pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import time 

# %% Load Spectra

wn_high = 2400
wn_low = 1250

path = 'SampleSpectra/'
files_all = glob.glob(path + "*")
# files_all.sort()

def load_csv(paths, wn_high = wn_high, wn_low = wn_low): 
    dfs = []    
    files = []
    for path in paths:
        head_tail = os.path.split(path)
        file = head_tail[1][0:-4]

        df = pd.read_csv(path, index_col = 0, header = 0, names = ['Wavenumber', 'Absorbance'])
        spec = df.loc[wn_low:wn_high]
        dfs.append(spec)
        files.append(file)
        
    zipobj = zip(files, dfs)
    dfs_dict = dict(zipobj)

    return dfs, dfs_dict, files, 
    
dfs, dfs_dict, files = load_csv(files_all, wn_high, wn_low)

# Sarah_PATH = Path.cwd().joinpath("SampleSpectra/OL50/")
# files_all = glob.glob(path + "*")
# files_all.sort()
# dfs, dfs_dict, files = load_csv(Sarah_PATH, wn_high, wn_low)



# %% PCA Vectors 

Nvectors = 5
PCA_DF = pd.read_csv("Baseline_Avg+PCA.csv", index_col= "Wavenumber")

Wavenumber = np.array([PCA_DF.index])
PCAmatrix = np.matrix(PCA_DF.to_numpy())
x = Wavenumber

# %% Define Functions. 

def Gauss(x, mu, sd, A=1):
    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))
    return G

def Lorentzian(x, center, half_width, amp=1):
    L = amp* (half_width**2/(half_width**2 + (2*x - 2*center)**2))
    return L

def linear2(x, tilt, offset):
    offset = np.ones_like(x) * offset
    tilt = np.arange(0, max(x.shape)) * tilt 
    tilt_offset = tilt + offset 
    return tilt_offset

def carbonate(P, x, PCAmatrix, Nvectors=5): 
    PCA_Weights = np.array( [P[0:Nvectors]] )

    peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude, peak_L1635, width_L1635, L1635_amplitude, slope, offset = P[Nvectors:None]

    L1635 = Lorentzian(x, center=peak_L1635, half_width=width_L1635, amp=L1635_amplitude)
    G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = linear2(x,slope, offset) 

    baseline =  PCA_Weights * PCAmatrix.T
    model_data = np.array(baseline + linear_offset + G1515 + G1430 + L1635)[0, :]

    return model_data
    

# %%

# Define the modeling function as a callable:
func = carbonate

# List of additional arguments of func (if necessary):
indparams = [x, PCAmatrix, Nvectors]

# Array of initial-guess values of fitting parameters:

P = [1.0, 0.1 , 0.5, 0.5, 0.001, 1430.0, 30, 0.1, 1515, 30, 0.1, 1635.0, 55.0, 0.5, 7e-5, 0.65]
params = np.float64(np.array(P))
# Lower and upper boundaries for the MCMC exploration:
pmin = np.array([0.0, -10.0 , -5, -5, -5, 1425.0, 25, 0.001, 1510, 25, 0.01, 1630.0, 40.0, 0.01, 5e-10, 0.0 ])
pmax = np.array([50.0, 10.0 ,  5,  5,  5, 1435.0, 35, 0.5,   1520, 40,  0.5, 1640.0, 70,  1.0,  5e-2, 2.0 ])


# Parameters' stepping behavior:
pstep = 0.01* params

# Parameter prior probability distributions:
# TODO Figure out how to use the Prior function better and how it differs from params. 
prior   =  params
priorlow = np.abs(params*0.5)
priorup  = np.abs(params*0.5)

# Parameter names:
# TODO Rename these to be better names for figures and shorten with Latex
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
plots    = False # False
rms      = True

# Carter & Winn (2009) Wavelet-likelihood method:
wlike = False

# # Parameter names:
# pnames   =  ['$\bar{BsL}$', "$PCA_1$", "$PCA_2$", "$PCA_3$", "$PCA_4$", '$G_{1430pk}$', '$$G_{1430std}$','$G_{1430amp}$', 'G_{1515pk}$', '$$G_{1515std}$', '$$G_{1515amp}$', 'L_{1635pk}', 'L_{1635wid}', 'L_{1635amp}', 'm', 'offset']
# texnames = ['$\bar{BsL}$', "$PCA_1$", "$PCA_2$", "$PCA_3$", "$PCA_4$",'$$G_{1430pk}$', '$G_{1430std}$', '$G_{1430amp}$', 'G_{1515pk}$', '$$G_{1515std}$', '$$G_{1515amp}$', 'L_{1635pk}', 'L_{1635wid}', 'L_{1635wid}', 'm', 'offset']


def MCMC(dfs_dict):
    
    # dfoutput = pd.DataFrame(columns = ['peak_L1635','width_L1635','L1635_amplitude','peak_G1515',
    #     'std_G1515','G1515_amplitude','peak_G1430', 'std_G1430', 'std_G1430'])

    # Run the MCMC:
    for files, data in dfs_dict.items():
        
        spec = data['Absorbance'].to_numpy()
        uncert = np.ones_like(spec)*0.01
        
        mc3_output = mc3.sample(data=spec, uncert=uncert, func=func, params=params,
            indparams=indparams, pstep=pstep, pmin=pmin, pmax=pmax,
            priorlow=priorlow, priorup=priorup, pnames=pnames, texnames=texnames,
            sampler=sampler, nsamples=nsamples,  nchains=nchains,
            ncpu=ncpu, burnin=burnin, thinning=thinning,
            leastsq=leastsq, chisqscale=chisqscale,
            grtest=grtest, grbreak=grbreak, grnmin=grnmin,
            hsize=hsize, kickoff=kickoff, wlike=wlike, log=log,
            plots=plots, savefile=savefile, rms=rms)

        # meanp = [mc3_output['meanp'][0:Nvectors]]
        # dfoutput.loc[files] = pd.Series({'peak_L1635': meanp[-5], 'width_L1635': meanp[-4], 'L1635_amplitude': meanp[-3],
        #     'peak_G1515': meanp[-8], 'std_G1515': meanp[-7], 'G1515_amplitude': meanp[-6],
        #     'peak_G1430': meanp[-11], 'std_G1430': meanp[-10], 'std_G1430': meanp[-9]})

    # return mc3_output


# %% 

dfoutput = MCMC(dfs_dict)


# %% PLOTTING - Prepare to plot all peaks

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

# Plot Peaks
fig, ax = plt.subplots(figsize= (12,8))
plt.plot(x[0,:], data, label = 'FTIR Spectrum')
plt.plot(x[0,:], L1635.T + Baseline_Solve.T , label = '1635')
plt.plot(x[0,:], G1515.T + Baseline_Solve.T, label = '1515')
plt.plot(x[0,:], G1430.T + Baseline_Solve.T, label = '1430')
plt.plot(x[0,:], carbonate(mc3_output['meanp'], x, PCAmatrix, Nvectors), label = 'MC3 Fit')
plt.plot(x[0,:], Baseline_Solve.T, label = 'Baseline')

plt.annotate(f"1635 cm $^{{- 1}}$  Peak Height: {L1635_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.95), xycoords = 'axes fraction')
plt.annotate(f"1515 cm $^{{- 1}}$  Peak Height: {G1515_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.925), xycoords = 'axes fraction')
plt.annotate(f"1430 cm $^{{- 1}}$  Peak Height: {G1430_amplitude:.2f} $\ cm ^{{- 1}} $ ", (0.01, 0.90), xycoords = 'axes fraction')

plt.title(f'{Spec_name}')

ax.invert_xaxis()
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
plt.legend(loc = 'lower right')
plt.show()


#%% 
import time

# %%

    for files, data in dfs_dict.items():
        
        spec = data['Absorbance'].to_numpy()
        uncert = np.ones_like(spec)*0.01
        print(np.size(spec))
        time.sleep(2)
        # print(uncert)
# %%


Sarah_PATH  = Path.cwd().joinpath("SampleSpectra/OL49_H2O_b.CSV")
li=[]
# sample=[]

def open_spectrum(path, wn_high=wn_high, wn_low=wn_low):
    df = pd.read_csv(path, index_col=0, header=0,
                     names=['Wavenumber', 'Absorbance'])
    spec = df.loc[wn_low:wn_high]
    return spec

Spectrum = open_spectrum(Sarah_PATH)

Spec_name = 'AC4_OL49'
# %%



# def MCMC(dfs_dict, nsamples, burnin, nchains, ncpu, thinning, log, savefile, plots):
    
#     dfoutput = pd.DataFrame(columns = ['peak_L1635','width_L1635','L1635_amplitude','peak_G1515',
#         'std_G1515','G1515_amplitude','peak_G1430', 'std_G1430', 'std_G1430'])

#     for files, data in dfs_dict.items():
#         data = data.to_numpy().reshape(1, 596)
#         mc3_output = mc3.sample(data=data, uncert=uncert, func=carbonate, params=params,
#             indparams=[x, PCAmatrix, Nvectors], pstep= 0.005 * params, 
#             pmin=pmin, pmax=pmax,
#             priorlow=np.abs(params*0.5), priorup=np.abs(params*0.5), 
#             pnames=pnames, texnames=texnames,
#             sampler='snooker', nsamples=nsamples,  nchains=nchains,
#             ncpu=ncpu, burnin=burnin, thinning=thinning,
#             leastsq='lm', chisqscale=False,
#             grtest=True, grbreak=1.01, grnmin=0.5,
#             hsize=10, kickoff='normal',
#             wlike=False, log=log,
#             plots=plots, savefile=savefile, rms=True)

#         meanp = mc3_output['meanp'] 
#         dfoutput.loc[files] = pd.Series({'peak_L1635': meanp[-5], 'width_L1635': meanp[-4], 'L1635_amplitude': meanp[-3],
#             'peak_G1515': meanp[-8], 'std_G1515': meanp[-7], 'G1515_amplitude': meanp[-6],
#             'peak_G1430': meanp[-11], 'std_G1430': meanp[-10], 'std_G1430': meanp[-9]})

#     return dfoutput
