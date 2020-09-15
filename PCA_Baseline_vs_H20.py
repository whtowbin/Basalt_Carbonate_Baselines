# %%
from PIL.Image import merge
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import glob
from scipy import signal
from scipy import interpolate
from pathlib import Path
# %%
# This is a list of the baseline databases to choose from. We will probably change this as we go. 


original_cleaned = 'Dans_Data_raw_cleaned.csv'
# Water Peak removed by Dan and smoothed for noise
NoH2O_Path_smoothed= 'Dans_smoothed_no_H2O2.csv'

# Water Peak removed by Dan, smoothed for noise, and the data has been offset 
# and scaled to make the spectra fit through the same point at the starting and ending wavenumber 
NoH2O_Path_smoothed_scaled= 'Dans_smoothed_no_H2O_scaled.csv'

# Scaled Data that has been manually filtered to remove bad spectra H2O peaks are present
Path_Scaled_cleaned2 = 'Dans_Data_scaled_cleaned2.csv'

# Raw Data that has been manually filtered to remove bad spectra H2O peaks are present
Path_Raw_cleaned2 = 'Dans_Data_raw_cleaned2.csv' 
# From my experience this is the best maybe we can figureout how to remove the water in this. 

NoH2O_rough= 'NoH2O_rough.csv'
# This is where I select which database is used to make the baselines.
DB_Name = NoH2O_rough #NoH2O_Path_smoothed #original #NoH2O_Path_smoothed #Path_Raw_cleaned2
DB_Path = Path.cwd().joinpath(DB_Name)
df_cleaned = pd.read_csv(DB_Path, index_col='Wavenumber')
frame=df_cleaned
Wavenumber = df_cleaned.index

# %%
H2O_DB_Name = Path_Raw_cleaned2 #original_cleaned
H2O_DB_Path = Path.cwd().joinpath(H2O_DB_Name)
H2O_df_cleaned = pd.read_csv(H2O_DB_Path, index_col='Wavenumber')
H2O_frame=H2O_df_cleaned
H2O_Wavenumber = H2O_df_cleaned.index

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


#wn_high = 1500
wn_low = 1250
wn_high = 2400

#wn_low = 1800
#wn_low = 1500
Wavenumber = frame.loc[wn_low:wn_high].index

frame_select = frame.loc[wn_low:wn_high]
Data_init = frame_select.values

H2O_frame_select = H2O_frame.loc[wn_low:wn_high]
H2O_Data_init = H2O_frame_select.values

#%%
# Subtract the mean from each column
"""
Mean_baseline= Data.mean(axis=1) 
Data = Data - Data.mean(axis=0)
Data = Data - np.array([Mean_baseline]).T
"""


# %%
def savgol_filter(x, smooth_width, poly_order):
    return signal.savgol_filter(x, smooth_width, poly_order)

#Smoothed= np.apply_along_axis(savgol_filter,0,Data, smooth_width= 101, poly_order= 3)

section_idx = np.where(np.round(Wavenumber) == 1500)[0][0]
Smooth_section1 = np.apply_along_axis(savgol_filter,0,Data_init[ 0:section_idx + 50, :], smooth_width= 51, poly_order= 3)
Smooth_section2 = np.apply_along_axis(savgol_filter,0,Data_init[ section_idx-50 :None, :], smooth_width= 301, poly_order= 3)

#%%
section_1 = Smooth_section1[0:-50, :]
section_2 = Smooth_section2[50:None, :]

offset_sections = section_1[-1] - section_2[0]
section_1  = section_1 - offset_sections

Smoothed = np.concatenate( [section_1, section_2], axis = 0)

#%%
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Wavenumber,Smoothed)
ax.invert_xaxis()

# Consider breaking up the spectra into overlapping segments and fitting. That way the overlapping regions wont cause a sharp transition when stitched. 

# %%
# Plots the database
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Wavenumber, Data)
plt.plot(Wavenumber, Smoothed)
ax.invert_xaxis()

ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
#%%
Data_start = Smoothed


# %%
#Normalize Data for scaling 
def scale_data(Data, Wavenumber):
    """
    Scales the data and subtracts the mean spectrum 
    """
    #Data = Data - Data.mean(axis =0)
    data_range= (Data[0,:]-Data[-1,:])
    scaled_data = Data / data_range
    
    Data = scaled_data - scaled_data[0,:] +0.5
    Mean_baseline= Data.mean(axis=1) 
    Data = Data - np.array([Mean_baseline]).T
    return Data, Mean_baseline

Data, Mean_baseline = scale_data(Data_start, Wavenumber)

#H2O_Data, H2O_Mean_baseline = scale_data(H2O_Data, Wavenumber)

# %%
# Calculates the Principle components
pca = PCA(6, ) # Number of PCA vectors to calculate 

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
plt.plot(Wavenumber, Mean_baseline+PCA_vectors[2]*1, label = 'Average Baseline + PCA2')
plt.plot(Wavenumber, Mean_baseline-PCA_vectors[2]*1, label = 'Average Baseline - PCA2')
#plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
plt.savefig('PCA1_plot.png')

# %%
# This needs to be rewritten to work with water removed peaks
columns = ["PCA "+ str(x) for x in range(1,4)]
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
# Make Dataframes without Water peak for improved to fit 
def H2O_peak_cut(df, wn_cut_low, wn_cut_high, return_DF = False):
    No_Peaks_Frame = df.drop(df[wn_cut_low:wn_cut_high].index)

    if return_DF == True:
        return No_Peaks_Frame
    
    No_Peaks_Wn = No_Peaks_Frame.index
    No_Peaks_Data = No_Peaks_Frame.values
    return  No_Peaks_Wn, No_Peaks_Data

# %%
wn_cut_low, wn_cut_high = (1500, 1800)

Full_No_Peaks_Wn, Full_No_Peaks_Values = H2O_peak_cut(H2O_frame_select, wn_cut_low, wn_cut_high)

H2O_free_PCA_DF = pd.DataFrame(PCA_vectors[0:2].T, index = Wavenumber)
PCA_No_Peaks_DF = H2O_peak_cut(H2O_free_PCA_DF, wn_cut_low, wn_cut_high, return_DF=True)

Average_baseline= pd.Series( Mean_baseline, index= Wavenumber)
Avg_BSL_no_peaks = H2O_peak_cut(Average_baseline, wn_cut_low, wn_cut_high, return_DF=True)
Avg_BSL_no_peaks_Wn = Avg_BSL_no_peaks.index

tilt = pd.Series(np.arange(0, len(Average_baseline)), index=Wavenumber)
tilt_cut = H2O_peak_cut(tilt, wn_cut_low, wn_cut_high, return_DF=True)

# %%
# Synthetic Peaks Choose peak shape, position and width. In the future these will be fit parameters
Peak1 = pd.Series( Lorentzian(x=Wavenumber, center=1635, half_width=55, amp=1),index=Wavenumber)
Peak2 = pd.Series(Gauss(x=Wavenumber, mu=1430, sd=30, A=1), index=Wavenumber)
Peak3 = pd.Series(Gauss(x=Wavenumber, mu=1515, sd=30, A=1), index=Wavenumber)

# %%
# Function to fit Water free baselines 
# uses the PCA components and the synthetic peaks to mad elinear combinations that fit the data. T


def No_H2O_fit(Spec, Average_baseline, PCA_DF,  tilt, Wavenumber = Wavenumber): 
    
    offset = pd.Series(np.ones(len(Average_baseline)), index= Wavenumber)
    #tilt = pd.Series(np.arange(0, len(Average_baseline)), index=Wavenumber)
    
    Baseline_Matrix = pd.concat([Average_baseline, PCA_DF, offset, tilt,], axis=1)

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, Spec)  

    return Baseline_Matrix, fit_param
# %%
def plot_NoH2O_results(Spectrum, Baseline_Matrix, fit_param, Wavenumber):


    fig, ax = plt.subplots(figsize=(12, 6))
    #plt.plot(Wavenumber,Spectrum.values, label = "Spectrum") for Pandas
    plt.plot(Wavenumber, Spectrum, label="Spectrum")

    plt.plot(Wavenumber,np.matrix(Baseline_Matrix)*np.matrix(fit_param[0]).T, label = 'Modeled Fit')

    #Peak1_amp = fit_param[0][-1]
    #Peak2_amp = fit_param[0][-3]
    #Peak3_amp = fit_param[0][-2]

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Absorbance')
    return ax

# %%
def tilt_fit(Spectrum, Baseline_Matrix, fit_param, Wavenumber):

    offset = pd.Series(np.ones(len(Wavenumber)), index= Wavenumber)
    tilt = pd.Series(np.arange(0, len(Wavenumber)), index=Wavenumber)


    Tilt_Matrix = pd.concat([offset, tilt], axis=1)

    Tilt_Matrix = np.matrix(Tilt_Matrix)

    fit_param = np.linalg.lstsq(Tilt_Matrix, Spec)  

# %%
#This line subtracts the mean from your data

Test_Spectrum = Full_No_Peaks_Values[:,19]
Baseline_Matrix, fit_param1 = No_H2O_fit(
    Spec=Test_Spectrum, Average_baseline=Avg_BSL_no_peaks, PCA_DF=PCA_No_Peaks_DF,
    Wavenumber = Avg_BSL_no_peaks_Wn, tilt = tilt_cut)



plot_NoH2O_results(Test_Spectrum , Baseline_Matrix=Baseline_Matrix,
                    fit_param=fit_param1, Wavenumber=Avg_BSL_no_peaks_Wn)

# %% Baseline Matrix without peaks removed
Average_baseline= pd.Series( Mean_baseline, index= Wavenumber)

Full_Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Data[:,0] , Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)

Full_Baseline_Matrix= Full_Baseline_Matrix[:, 0:-3]
#%%
# make list of all fit parameters to use to replace data. 
Fits = []
No_H2O_baseline = []
 

for spec in Full_No_Peaks_Values.T:
    Baseline_Matrix, fit_param = No_H2O_fit(
    Spec=spec, Average_baseline=Avg_BSL_no_peaks, PCA_DF=PCA_No_Peaks_DF,
    Wavenumber = Avg_BSL_no_peaks_Wn, tilt = tilt_cut)

    Fits.append(fit_param)

    base_full = Full_Baseline_Matrix*np.matrix(fit_param[0]).T
    No_H2O_baseline.append(base_full)

# %%

spec_idx = 10

fig, ax = plt.subplots(figsize = (12,6))
plt.plot(Wavenumber, No_H2O_baseline[spec_idx], label= "No Water Baseline")
plt.plot(Wavenumber, H2O_Data_init[:,spec_idx], label= "Spectrum" )

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')

# %%

No_H2O_baseline_df = pd.DataFrame(np.array(No_H2O_baseline)[:,:,0].T, index = Wavenumber, columns= H2O_frame_select.columns)

H2O_Peaks = H2O_frame_select - No_H2O_baseline_df

cutout = No_H2O_baseline_df[wn_cut_low:wn_cut_high]

#smooth_cutout = np.apply_along_axis(savgol_filter,0,cutout)

smooth_cutout_df = pd.DataFrame(cutout, index =cutout.index, columns= cutout.columns)

fig, ax = plt.subplots(figsize=(12,6))
plt.plot(cutout.index, smooth_cutout)
plt.plot(cutout)
#%%


offset_H2O_frame = H2O_frame_select.loc[wn_cut_high-1:wn_cut_high].values - H2O_frame_select.loc[wn_cut_low:wn_cut_low+1].values 

offset_smooth_cutout =  smooth_cutout_df.loc[wn_cut_high-1:wn_cut_high].values  - smooth_cutout_df.loc[wn_cut_low:wn_cut_low+1].values 

Scaled_data_cut = smooth_cutout_df* (offset_H2O_frame/ offset_smooth_cutout)

offset_1500 = H2O_frame_select.loc[wn_cut_low:wn_cut_low+1].values - Scaled_data_cut.loc[wn_cut_low:wn_cut_low+1].values

offset_1800 = H2O_frame_select.loc[wn_cut_high-1:wn_cut_high].values - Scaled_data_cut.loc[wn_cut_high-1:wn_cut_high].values

#slope = (offset_1800.values - offset_1500.values)/ (wn_cut_high-wn_cut_low)
#cutout_tilt = np.tile(np.arange(0, len(cutout.index)) - len(cutout.index)/2,(len(cutout.columns),1))

#stitch_adjust = pd.DataFrame(np.multiply(slope, cutout_tilt.T ) + offset_1500.values, index = cutout.index, columns = cutout.columns)

#cut_adjusted = smooth_cutout_df + stitch_adjust
cut_adjusted = Scaled_data_cut +offset_1800

# I need to stretch not tilt!!!

fig, ax = plt.subplots(figsize = (12,6))
#plt.plot(cut_adjusted)
plt.plot(cut_adjusted)

#%%
Peaks_removed_full_DF = H2O_frame_select.drop(H2O_frame_select.loc[wn_cut_low:wn_cut_high].index).append(cut_adjusted,)
Peaks_removed_full_DF.sort_index(inplace=True)
plt.plot(Peaks_removed_full_DF)
#%%

def savgol_filter_short(x):
    return signal.savgol_filter(x, 51, 3)

Peaks_removed_full = Peaks_removed_full_DF.apply( func = signal.savgol_filter, args = (31,3))
Peaks_removed_full = pd.DataFrame(Peaks_removed_full, index = Peaks_removed_full_DF.index, columns = Peaks_removed_full_DF.columns) 
Data_start = Peaks_removed_full.values

#%%
fig, ax = plt.subplots(figsize = (12,6))
plt.plot(Peaks_removed_full_DF, marker='o', markersize= 3, linestyle=None)
ax.set_xlim(1750, 1850)
ax.set_ylim(0.3,0.5)

fig, ax = plt.subplots(figsize = (12,6))
plt.plot(Peaks_removed_full_DF, marker='o', markersize= 1, linestyle=None)
ax.set_xlim(1450, 1550)
ax.set_ylim(0.3,0.5)
# In order to stitch I need to have the yint from offset_1500 and add 
#gh = cutout_tilt * slope + smooth_cutout
#Data = H2O_Peaks.values

#%%
 # %%
# Function to fit the baselines: 
# uses the PCA components and the synthetic peaks to mad elinear combinations that fit the data. T

Average_baseline= pd.Series( Mean_baseline, index= Wavenumber)

def Carbonate_baseline_fit(Spec, Average_baseline, PCA_vectors, n_PCA_vectors=2, Peak1=Peak1, Peak2=Peak2, Peak3=Peak3):

    PCA_DF = pd.DataFrame(PCA_vectors[0:n_PCA_vectors].T, index=Wavenumber)
    
    offset = pd.Series(np.ones(len(Peak2)), index= Wavenumber)
    tilt = pd.Series(np.arange(0, len(Peak2)), index=Wavenumber)

    
    #Baseline_Matrix = pd.concat([ Average_baseline, PCA_DF, offset, tilt, Peak2, Peak3], axis=1)

    # This line is only used if we are fitting the Water peak with the CO2 peak. 
    Baseline_Matrix = pd.concat([Average_baseline, PCA_DF, offset, tilt, Peak2, Peak3, Peak1], axis=1)

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, Spec)  

    return Baseline_Matrix, fit_param

# %%
# plots the results

def plot_Baseline_results(Spectrum, Baseline_Matrix, fit_param, Wavenumber):
    modeled_basline = np.matrix(
        Baseline_Matrix[:, 0:-3])*fit_param[0][0:-3]  # Ignores the Peaks in fit.

    #modeled_basline = np.matrix(
    #   Baseline_Matrix[:, 0:-2])*fit_param[0][0:-2]  # Ignores the Peaks in fit. Only for CO2 peaks


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
    Spec=sarahFTIR, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)


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
    Spec=Spectrum1, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)


plot_Baseline_results(Spectrum1, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('CL05MI01.csv')
plt.savefig('CL05MI01_baselinefit.png')

# %%


Spectrum2 = Dan_FTIR_select.iloc[:, 1].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
#Spectrum2 = StandardScaler(with_std=False).fit_transform(Spectrum2)

Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum2, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)

   
plot_Baseline_results(Spectrum2, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param1, Wavenumber=Wavenumber)
plt.title('CL05MI02.csv')
plt.savefig('CL05MI02_baselinefit.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 2].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)

Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)


plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('CL05MI08.csv')
plt.savefig('CL05MI08_baselinefit.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 3].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)


plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('CV04MI03.csv')
plt.savefig('CV04MI03_baselinefit.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 4].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)


plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.title('FR02MI01.csv')
plt.savefig('FR02MI01_baselinefit_.png')
# %%
Spectrum = Dan_FTIR_select.iloc[:, 5].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum, Average_baseline=Average_baseline, n_PCA_vectors=2, PCA_vectors=PCA_vectors)


plot_Baseline_results(Spectrum, Baseline_Matrix=Baseline_Matrix,
                      fit_param=fit_param, Wavenumber=Wavenumber)
plt.savefig('FR04MI01_baselinefit.png')
# %%
# %%
