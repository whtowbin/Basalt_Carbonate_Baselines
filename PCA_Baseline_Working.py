# %%

from matplotlib import pyplot as plt
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import glob

# Monte Carlo Markov Chain
# import mc3

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from scipy import signal
from scipy import interpolate
from pathlib import Path

# %% List of the baseline databases. 

# Raw Data that has been manually filtered to remove bad spectra H2O peaks are present
# From my experience this is the best maybe we can figure out how to remove the water in this. 
RawCleaned2 = 'Dans_Data_raw_cleaned2.csv' 

# Decarbonated Spectra normalized to a 100 micron thickness. Cleaned indicates removal of nonideal spectra. 
ThicknessNormalized = 'Thickness100.csv' # 
ThicknessCleanedNormalized = 'Thickness100_Cleaned.csv' 
# Cleaned removes WD33MI03, SH55MI02, SH01MI05, FI38MII01C, SH63MI43, FI45MI05, WD70MI01A, OK14MI08, FI38MI05A, SH63MI31, SH63MI24, SH63MI53

# Select the database used to make the baselines.
database = ThicknessCleanedNormalized
database_path = Path.cwd().joinpath(database)
df_cleaned = pd.read_csv(database_path, index_col='Wavenumber')
frame = df_cleaned
wavenumber = df_cleaned.index
devolatilized = list(frame.columns)

# %%
# Defining the Peaks shapes

def linear(x,m,b):
    line = x*m +b
    return line

def Gauss(x, mu, sd, A=1):
    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))
    return G

def select_PCA(PCAvectors, Nvectors):
    # function to make PCA vectors and output them as parameters
    return None

p1 = [PCA_Weights, peak_G1430, halfwidth_G1430, peak_G1515, halfwidth_G1515, amplitude, slope, offset]

# Starting Guess peak positions
# mu=1430, sd=30
# mu=1515, sd=30
# Input Order from Least squares
# [PCA_DF, offset, tilt, amplitude, amplitude]
# [PCA_Weights, offset, slope, amplitude, amplitude, peak_G1430, halfwidth_G1430, peak_G1515, halfwidth_G1515, ]


#PCA vectors needs to be properly unpacked since it is multidimensional. 
# Maybe it would be easiest to use the least squares minimization to fit PCA baselines after randomly doing peaks?
# This isnt ideal but could work. we want to know the vectors that best explain variance. 
# If program can take multiple length inputs we are fine,
# If not first version might be best to solve for first 6 PCA components. 
# MC3 must be 1D. 
# Maybe function could be wrapped inanother funtion that takes a 1D array. 
# Emcee we could write our own.  

# another idea is that one of the nonsampled parameters tells  the number of PCA vectors weights in P.
# 
# wavenumber = x

# Baseline Maxtrix should be input not vectors. Everything should be in proper format for matrix multiplication and addition
# Make input the exact same as the least squares outputs. And initial guess will be those outputs
# use the default peak positions and widths. 
def carbonate(x, PCAmatrix, Nvectors, P): # add terms M and X, PCA_fit terms
    
    PCA_Weights = np.array( P[0:Nvectors] )
    peak_G1430, std_G1430, peak_G1515, std_G1515, amplitude, slope, offset = P[Nvectors,None]

    G1515 = Gauss(x, peak_G1515, std_G1515, A=amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=amplitude)

    linear_offset = linear(wavenumber,slope, offset) 
   
    baseline =  PCAmatrix * PCA_Weights
    model_data = baseline + linear_offset + G1515 + G1430 

    return model_data


def baselineplotting(spectrum, titles, baselinematrix, fitparameter, wavenumber):
    modeled_baseline = np.matrix(baselinematrix[:, 0:-2]) * fitparameter[0][0:-2]  # Ignores peaks in fit.

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(wavenumber, spectrum, label = 'Spectrum')
    plt.plot(wavenumber, np.matrix(baselinematrix)*fitparameter[0], label = 'Modeled Fit')
    plt.plot(wavenumber, modeled_baseline, label = 'Baseline')
    ax.set_title('%s' %titles)
    ax.set_xlabel('Wavenumber (cm$^-1$')
    ax.set_ylabel('Absorbance')
    ax.invert_xaxis()
    ax.legend()
    return ax

# %% Select database subset by wavenumber.

wn_high = 2400
wn_low = 1250
wavenumber = frame.loc[wn_low:wn_high].index

frame_select = frame.loc[wn_low:wn_high]
Data = frame_select.values

#%% Subtract the mean from each column.

Data = Data - Data.mean(axis = 0)

# %% Plots the database.

fig, ax = plt.subplots(figsize = (12,8))
plt.plot(wavenumber, Data)
ax.invert_xaxis()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
ax.set_ylim(-0.4, 1.2)
plt.show()

# row1, col1 = np.shape(Data)

# for jj in range(col1):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     plt.plot(wavenumber, Data[:, jj])
#     ax.set_title('%s' %devolatilized[jj])
#     ax.invert_xaxis()
#     ax.set_xlabel('Wavenumber')
#     ax.set_ylabel('Absorbance')
#     ax.set_ylim(-0.6, 1.2)
#     plt.show()

# %% Calculates the Principle Components then plots the fraction of variance explained by each PCA vector. 

pca = PCA(10, ) # Number of PCA vectors to calculate 
principalcomponents = pca.fit(Data.T) #everything appears to work best with the raw data or raw data scaled

fig, ax = plt.subplots(figsize=(12,8))
variance = pca.explained_variance_
variance_norm = variance[0: -1]/np.sum(variance[0: -1])
plt.plot(variance_norm*100, marker='o', linestyle = 'None' )
ax.set_xlabel('Principle Component')
ax.set_ylabel('%Variance in CO$_2$ Free Baselines Selection')
pca.singular_values_
PCA_vectors = pca.components_
plt.show()
# plt.savefig('PCA_Variance_plot.png')

# %% Plots several Principle Components.

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wavenumber, PCA_vectors[0], label="PCA:1")
plt.plot(wavenumber, PCA_vectors[1], label="PCA:2")
plt.plot(wavenumber, PCA_vectors[2], label="PCA:3")
plt.plot(wavenumber, PCA_vectors[3], label="PCA:4")
# plt.plot(Wavenumber, PCA_vectors[4], label="PCA:5")
# plt.plot(Wavenumber, PCA_vectors[5], label="PCA:6")
# plt.plot(Wavenumber, PCA_vectors[6], label="PCA:7")
# plt.plot(Wavenumber, PCA_vectors[7], label="PCA:8")
plt.legend()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
ax.invert_xaxis()
ax.legend()
plt.show()
# plt.savefig('component_plot.png')


# %% Test spectrum

# testspec = Path.cwd().joinpath("Sarah_FTIRSpectra/Fuego2018FTIRSpectra_Transmission/OL49_H2O_b.CSV")

# def open_spectrum(path, wn_high=wn_high, wn_low=wn_low):
#     df = pd.read_csv(path, index_col=0, header=0,
#                      names=['Wavenumber', 'Absorbance'])
#     spec = df.loc[wn_low:wn_high]
#     return spec

# test = open_spectrum(testspec)
# baselinematrix, fitparameter = carbonate(x=wavenumber, peak_L = 1635, halfwidth_L = 55, peak_G1430 = 1430, halfwidth_G1430 = 30, peak_G1515 = 1515, halfwidth_G1515 = 30, amplitude = 1, spectrum = test, PCAvectors = PCA_vectors, nPCAvectors = 6)
# baselineplotting(test, baselinematrix = baselinematrix, fitparameter = fitparameter, wavenumber = wavenumber)


# #This line subtracts the mean from your data
# sarahFTIR = StandardScaler(with_std=False).fit_transform(Spectrum)
# sarahFTIR = Spectrum
# Baseline_Matrix, fit_param = Carbonate_baseline_fit(
#     Spec=sarahFTIR, n_PCA_vectors=6, PCA_vectors=PCA_vectors)


# baselineplotting(sarahFTIR, baselinematrix=baselinematrix,
#                       fitparameter=fitparameter, wavenumber=wavenumber)
# plt.title('AC4_OL49')
# plt.savefig('AC4_OL49_baselinefit.png')

# %%
# Loads Sarah's spectra. 

Sarah_PATH = Path.cwd().joinpath('SampleSpectra')
allfiles = Sarah_PATH.rglob('*.CSV')
li = []
samples = []

for filename in allfiles:
    df = pd.read_csv(filename, index_col=0, header=0, names=['Wavenumber', 'Absorbance']
                     )
    li.append(df)
    samples.append(filename)

Sarah_FTIR = pd.concat(li, axis=1, )
Sarah_FTIR_Select = Sarah_FTIR.loc[wn_low:wn_high]

row, col = np.shape(Sarah_FTIR_Select)

for ii in range(col): 
    S = Sarah_FTIR_Select.iloc[:, ii].values.reshape(len(Sarah_FTIR_Select.iloc[:, 0]), 1)
    S = StandardScaler(with_std=False).fit_transform(S)

    baselinematrix, fitparameter = carbonate(x=wavenumber, peak_L = 1635, halfwidth_L = 55, peak_G1430 = 1430, halfwidth_G1430 = 30, peak_G1515 = 1515, halfwidth_G1515 = 30, amplitude = 1, spectrum = S, PCAvectors = PCA_vectors, nPCAvectors = 6)
    baselineplotting(S, samples[ii], baselinematrix = baselinematrix, fitparameter = fitparameter, wavenumber = wavenumber)
    

# %%

# %%

# %%

# %%

# %%
