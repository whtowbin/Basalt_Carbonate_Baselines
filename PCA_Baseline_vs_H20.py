# %%
from Users.henry..vscode.extensions.ms-python.vscode-pylance-2020.8.3.server.bundled-stubs.matplotlib.pyplot import colorbar
from Users.henry.Python Files.Basalt Carbonate Baselines.MC3 Baseline import wn_low
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import glob
from scipy import signal
from scipy import interpolate
from pathlib import Path
# %%
# This is a list of the baseline databases to choose from. We will probably change this as we go. 

original = 'Dans_Data.csv'
# Water Peak removed by Dan and smoothed for noise
NoH2O_Path_smoothed= 'Dans_smoothed_no_H2O.csv'

# Water Peak removed by Dan, smoothed for noise, and the data has been offset 
# and scaled to make the spectra fit through the same point at the starting and ending wavenumber 
NoH2O_Path_smoothed_scaled= 'Dans_smoothed_no_H2O_scaled.csv'

# Scaled Data that has been manually filtered to remove bad spectra H2O peaks are present
Path_Scaled_cleaned2 = 'Dans_Data_scaled_cleaned2.csv'

# Raw Data that has been manually filtered to remove bad spectra H2O peaks are present
Path_Raw_cleaned2 = 'Dans_Data_raw_cleaned2.csv' 
# From my experience this is the best maybe we can figureout how to remove the water in this. 

# This is where I select which database is used to make the baselines.
DB_Name = original #NoH2O_Path_smoothed #Path_Raw_cleaned2
DB_Path = Path.cwd().joinpath(DB_Name)
df_cleaned = pd.read_csv(DB_Path, index_col='Wavenumber')
frame=df_cleaned

Wavenumber = df_cleaned.index

# %%
# Defining the Peaks shapes
def Lorentzian(x, center, half_width, amp=1):
    L = amp* (half_width**2/(half_width**2 + (2*x - 2*center)**2))
    return L

def Gauss(x, mu, sd, A=1):
    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))
    return G

def linear(x,m):
    line = x*m
    line = line - np.mean(line)
    return line
# %%

#Select subset of the database spectra by wavenumber
#wn_high = 2400

wn_high = 1500
wn_low = 1250

#wn_low = 1800
#wn_low = 1500
Wavenumber = frame.loc[wn_low:wn_high].index

frame_select = frame.loc[wn_low:wn_high]
Data = frame_select.values


#%%
# Subtract the mean from each column
Data = Data - Data.mean(axis =0)
# %%
#Normalize Data for scaling 


data_range= (Data[0,:]-Data[-1,:])
scaled_data = Data / data_range
#Data = scaled_data + scaled_data.mean(axis =0) 
Data = scaled_data - scaled_data[0,:] +0.5
Mean_baseline= Data.mean(axis=1) 
Data = Data - np.array([Mean_baseline]).T
# %%
# Plots the database
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Wavenumber, Data)

ax.invert_xaxis()

ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')

# %%
# Calculates the Principle components
pca = PCA(10, ) # Number of PCA vectors to calculate 

principalComponents = pca.fit(Data.T) #everything appears to work best with the raw data or raw data scaled

reduced_data = pca.fit(Data.T).transform(Data.T)
# %%
#plots the fraction of the variance explained by each PCA vector 
fig, ax = plt.subplots(figsize=(12,6))
variance = pca.explained_variance_
variance_norm = variance[0: -1]/np.sum(variance[0: -1])
plt.plot(variance_norm*100, marker='o', linestyle = 'None' )
ax.set_xlabel("Principle Component")
ax.set_ylabel(r"%Variance in CO_2 Free Baselines Selection")
pca.singular_values_
PCA_vectors = pca.components_
plt.savefig('PCA_Variance_plot.png')


# %%
#Plots the first several principle components

fig, ax = plt.subplots(figsize=(12,6))
plt.plot(Wavenumber, PCA_vectors[0], label="PCA:1")
plt.plot(Wavenumber, PCA_vectors[1], label="PCA:2")
plt.plot(Wavenumber, PCA_vectors[2], label="PCA:3")
#plt.plot(Wavenumber, PCA_vectors[3], label="PCA:4")
#plt.plot(Wavenumber, PCA_vectors[4], label="PCA:5")
#plt.plot(Wavenumber, PCA_vectors[5], label="PCA:6")
#plt.plot(Wavenumber, PCA_vectors[6], label="PCA:7")
#plt.plot(Wavenumber, PCA_vectors[7], label="PCA:8")
plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
plt.savefig('component_plot.png')

#%%
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(reduced_data[:,0],reduced_data[:,1])
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
# %%

fig, ax = plt.subplots(figsize=(12,6))

plt.plot(Wavenumber, Mean_baseline, label = 'Average Baseline' ,linewidth=3)
plt.plot(Wavenumber, Mean_baseline+PCA_vectors[0]*1, label = 'Average Baseline + PCA1')
plt.plot(Wavenumber, Mean_baseline-PCA_vectors[0]*1, label = 'Average Baseline - PCA1')
#plt.plot(Wavenumber, Mean_baseline-PCA_vectors[0]*2, label = 'Average Baseline - PCA1 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
plt.savefig('PCA1_plot.png')

#%%
fig, ax = plt.subplots(figsize=(12,6))

plt.plot(Wavenumber, Mean_baseline, label = 'Average Baseline', linewidth=3)
plt.plot(Wavenumber, Mean_baseline+PCA_vectors[1]*1, label = 'Average Baseline + PCA2')
plt.plot(Wavenumber, Mean_baseline-PCA_vectors[1]*1, label = 'Average Baseline - PCA2')
#plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
plt.savefig('PCA1_plot.png')

# %%
columns = ["PCA "+ str(x) for x in range(1,11)]
PCA_reduced_data_df = pd.DataFrame(reduced_data, index = frame.columns,columns=columns )

chem_data_path = '/Users/henry/Python Files/Basalt Carbonate Baselines/Dans_Baselinefiles/Databasefiles/mi_data_select.csv'

chem_data_df = pd.read_csv(chem_data_path, index_col=['Sample'])

chem_data_df = chem_data_df.loc[ chem_data_df.index.intersection(PCA_reduced_data_df.index)]


result = pd.concat([PCA_reduced_data_df, chem_data_df], axis=1, sort=False)

Mol_MgO = result['MGO']/40.3044 
Mol_FeO = result['FEO']/71.844
Fo = Mol_MgO/ (Mol_MgO+Mol_FeO)

#%%
fig, ax = plt.subplots(figsize =(10,8))
cm = plt.cm.get_cmap('RdYlBu')
x = result['PCA 1']
y = result['PCA 2']
z = result['H2O']
sc = plt.scatter(x, y, c=z, cmap=cm)
cbar= plt.colorbar(sc)

ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
cbar.set_label(z.name)


#%%
fig, ax = plt.subplots(figsize =(10,8))
cm = plt.cm.get_cmap('RdYlBu')
x = result['PCA 1']
y =result['CAO'] 
z = result['Thickness']
sc = plt.scatter(x, y, c=z, cmap=cm)
cbar= plt.colorbar(sc)

ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
cbar.set_label(z.name)




# %%
# Synthetic Peaks Choose peak shape, position and width. In the future these will be fit parameters
Peak1 = pd.Series( Lorentzian(x=Wavenumber, center=1635, half_width=55, amp=1),index=Wavenumber)
Peak2 = pd.Series(Gauss(x=Wavenumber, mu=1430, sd=30, A=1), index=Wavenumber)
Peak3 = pd.Series(Gauss(x=Wavenumber, mu=1515, sd=30, A=1), index=Wavenumber)

# %%
# Function to fit the baselines: 
# uses the PCA components and the synthetic peaks to mad elinear combinations that fit the data. T


def Carbonate_baseline_fit(Spec, PCA_vectors, n_PCA_vectors=4, Peak1=Peak1, Peak2=Peak2, Peak3=Peak3):

    PCA_DF = pd.DataFrame(PCA_vectors[0:n_PCA_vectors].T, index=Wavenumber)
    
    offset = pd.Series(np.ones(len(Peak2)), index= Wavenumber)
    tilt = pd.Series(np.arange(0, len(Peak2)), index=Wavenumber)

    
    Baseline_Matrix = pd.concat([PCA_DF, offset, tilt, Peak2, Peak3], axis=1)

    # This line is only used if we are fitting the Water peak with the CO2 peak. 
    #Baseline_Matrix = pd.concat([PCA_DF, offset, tilt, Peak2, Peak3, Peak1], axis=1)

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, Spec)  

    return Baseline_Matrix, fit_param

# %%
# plots the results

def plot_Baseline_results(Spectrum, Baseline_Matrix, fit_param, Wavenumber):
    modeled_basline = np.matrix(
        Baseline_Matrix[:, 0:-2])*fit_param[0][0:-2]  # Ignores the Peaks in fit.

    fig, ax = plt.subplots(figsize=(12, 6))
    #plt.plot(Wavenumber,Spectrum.values, label = "Spectrum") for Pandas
    plt.plot(Wavenumber, Spectrum, label="Spectrum")

    plt.plot(Wavenumber,np.matrix(Baseline_Matrix)*fit_param[0], label = 'Modeled Fit')
    plt.plot(Wavenumber,modeled_basline, label='Baseline')

    #Peak1_amp = fit_param[0][-1]
    #Peak2_amp = fit_param[0][-3]
    #Peak3_amp = fit_param[0][-2]

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Absorbance')
    return ax


# %%
# Loads One of Sarah's Spectra. We can easily modify this to load them all like I have done with Dan's below

Sarah_path = Path.cwd().joinpath("Sarah's FTIR Spectra/Fuego2018FTIRSpectra_Transmission/AC4_OL49_021920_30x30_H2O_b.CSV")

def open_spectrum(path, wn_high=wn_high, wn_low=wn_low):
    df = pd.read_csv(path, index_col=0, header=0,
                     names=['Wavenumber', 'Absorbance'])
    spec = df.loc[wn_low:wn_high]
    return spec

Spectrum = open_spectrum(Sarah_path)
# %%

#This line subtracts the mean from your data
sarahFTIR = StandardScaler(with_std=False).fit_transform(Spectrum)
sarahFTIR = Spectrum
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=sarahFTIR, n_PCA_vectors=5, PCA_vectors=PCA_vectors)


plot_Baseline_results(sarahFTIR, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('AC4_OL49')
plt.savefig('AC4_OL49_baselinefit.png')
# %%

# Loads All of Dan's data to fit

Path_Dan_test = Path.cwd().joinpath("Dans_Examples")
all_files = Path_Dan_test.rglob("*.csv")
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=0, header=0, names=['Wavenumber', 'Absorbance']
                     )
    li.append(df)


Dan_FTIR = pd.concat(li, axis=1, )

Dan_FTIR_select = Dan_FTIR.loc[wn_low:wn_high]
# %%
#Processing Dan's data

Spectrum1 = Dan_FTIR_select.iloc[:, 0].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Spectrum1 = StandardScaler(with_std=False).fit_transform(Spectrum1)

Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spectrum1, n_PCA_vectors=9, PCA_vectors=PCA_vectors)
plot_Baseline_results(Spectrum1, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('CL05MI01.csv')
plt.savefig('CL05MI01_baselinefit.png')

# %%


Spectrum2 = Dan_FTIR_select.iloc[:, 1].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
#Spectrum2 = StandardScaler(with_std=False).fit_transform(Spectrum2)
Baseline_Matrix1, fit_param1 = Carbonate_baseline_fit(
    Spectrum2, n_PCA_vectors=5, PCA_vectors=PCA_vectors)
plot_Baseline_results(Spectrum2, Baseline_Matrix=Baseline_Matrix1,
                      fit_param=fit_param1, Wavenumber=Wavenumber)
plt.title('CL05MI02.csv')
plt.savefig('CL05MI02_baselinefit.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 2].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spectrum, n_PCA_vectors=7, PCA_vectors=PCA_vectors)
plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('CL05MI08.csv')
plt.savefig('CL05MI08_baselinefit.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 3].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spectrum, n_PCA_vectors=7, PCA_vectors=PCA_vectors)
plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('CV04MI03.csv')
plt.savefig('CV04MI03_baselinefit.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 4].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spectrum, n_PCA_vectors=7, PCA_vectors=PCA_vectors)
plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('FR02MI01.csv')
plt.savefig('FR02MI01_baselinefit_.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 5].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spectrum, n_PCA_vectors=7, PCA_vectors=PCA_vectors)
plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.savefig('FR04MI01_baselinefit.png')
# %%
# %%
