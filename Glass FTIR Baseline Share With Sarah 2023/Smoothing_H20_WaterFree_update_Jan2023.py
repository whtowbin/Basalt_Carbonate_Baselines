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


original_cleaned = "Dans_Data_raw_cleaned.csv"
# Water Peak removed by Dan and smoothed for noise
NoH2O_Path_smoothed = "Dans_smoothed_no_H2O2.csv"

# Water Peak removed by Dan, smoothed for noise, and the data has been offset
# and scaled to make the spectra fit through the same point at the starting and ending wavenumber
NoH2O_Path_smoothed_scaled = "Dans_smoothed_no_H2O_scaled.]csv"

# Scaled Data that has been manually filtered to remove bad spectra H2O peaks are present
Path_Scaled_cleaned2 = "Dans_Data_scaled_cleaned2.csv"

# Raw Data that has been manually filtered to remove bad spectra H2O peaks are present
Path_Raw_cleaned2 = "Dans_Data_raw_cleaned2.csv"
# From my experience this is the best maybe we can figureout how to remove the water in this.

NoH2O_rough = "NoH2O_rough.csv"
NoH2O_rough_select = "NoH2O_rough_select.csv"
NoH2O_rough_select2 = "NoH2O_rough_select2.csv"  # removes Newman Baseline which doenst go higher than 2400
# This is where I select which database is used to make the baselines.
DB_Name = NoH2O_rough_select2  # NoH2O_Path_smoothed #original #NoH2O_Path_smoothed #Path_Raw_cleaned2
DB_Path = Path.cwd().joinpath(DB_Name)
df_cleaned = pd.read_csv(DB_Path, index_col="Wavenumber")
frame = df_cleaned
Wavenumber = df_cleaned.index

# %%
H2O_DB_Name = original_cleaned  # Path_Raw_cleaned2 #original_cleaned
H2O_DB_Path = Path.cwd().joinpath(H2O_DB_Name)
H2O_df_cleaned = pd.read_csv(H2O_DB_Path, index_col="Wavenumber")
H2O_frame = H2O_df_cleaned
H2O_Wavenumber = H2O_df_cleaned.index

# %%
# Defining the Peaks shapes
def Lorentzian(x, center, half_width, amp=1):
    # L = amp * (half_width ** 2 / (half_width ** 2 + (2 * x - 2 * center) ** 2))
    V = (x - center) / (half_width / 2)
    L = amp * 1 / (1 + V**2)
    return L


def Gauss(x, mu, sd, A=1):
    G = A * np.exp(-((x - mu) ** 2) / (2 * sd**2))
    return G


def linear(x, m):
    line = x * m
    line = line - np.mean(line)
    return line


#%%


def basic_scale_data(Data, Wavenumber):
    """
    Scales the data only
    """
    # Data = Data - Data.mean(axis =0)
    data_range = Data[0, :] - Data[-1, :]
    scaled_data = Data / np.abs(data_range)

    Data = scaled_data - scaled_data[0, :] + 0.5
    Mean_baseline = Data.mean(axis=1)

    return Data, Mean_baseline

    Data = scaled_data - scaled_data[0, :] + 0.5


# Normalize Data for scaling
def scale_data(Data, Wavenumber):
    """
    Scales the data and subtracts the mean spectrum
    """
    # Data = Data - Data.mean(axis =0)
    data_range = Data[0, :] - Data[-1, :]
    scaled_data = Data / np.abs(data_range)

    Data = scaled_data - scaled_data[0, :] + 0.5
    Mean_baseline = Data.mean(axis=1)
    Data = Data - np.array([Mean_baseline]).T
    return Data, Mean_baseline


# %%

# Select subset of the database spectra by wavenumber

# wn_high = 1500
wn_low = 1275  # 1250
wn_low_init = wn_low

wn_high = 2400  # 2200  # 2400
wn_high_init = wn_high

# wn_low = 1800
# wn_low = 1500
Wavenumber = frame.loc[wn_low:wn_high].index
Wavenumber_full = Wavenumber
frame_select = frame.loc[wn_low:wn_high]
Data_init = frame_select.values

H2O_frame_select = H2O_frame.loc[wn_low:wn_high]
H2O_Data_init = H2O_frame_select.values
H2O_Data__scale, H2O_Mean_baseline_scale = basic_scale_data(H2O_Data_init, Wavenumber)


#%%
def savgol_filter(x, smooth_width, poly_order):
    return signal.savgol_filter(x, smooth_width, poly_order)


# Smooth_diff = np.apply_along_axis(savgol_filter,0,diff, smooth_width= 71, poly_order= 4)
def interp_smooth(spectrum, Wavenumber=Wavenumber):
    w = np.ones_like(Wavenumber)
    w[0] = 100
    w[-1] = 100
    interp = interpolate.UnivariateSpline(
        x=Wavenumber, y=spectrum, k=5, s=0.01, ext=0, w=w
    )
    return interp(Wavenumber)


#%%
def smoothing_protocol(Data_values, Wavenumber=Wavenumber):
    section_idx = np.where(np.round(Wavenumber) == 1500)[0][
        0
    ]  # Select the divide into segments for smoothing

    # Smooth low wavenumbers
    Smooth_section1 = np.apply_along_axis(
        savgol_filter,
        0,
        Data_values[0 : section_idx + 50, :],
        smooth_width=91,  # Filter smoothing width
        poly_order=4,
    )
    diff = Smooth_section1 - Data_values[0 : section_idx + 50, :]
    Smooth_diff = np.apply_along_axis(
        interp_smooth, 0, diff, Wavenumber=Wavenumber[0 : section_idx + 50]
    )

    # Smooth high wavenumbers
    Smooth_section1 = Smooth_section1 - Smooth_diff
    Smooth_section2 = np.apply_along_axis(
        savgol_filter,
        0,
        Data_values[section_idx - 50 : None, :],
        smooth_width=131,  # 121
        poly_order=4,
    )

    # Cut and Stitch Smoothed Sections
    section_1 = Smooth_section1[0:-50, :]
    section_2 = Smooth_section2[50:None, :]

    offset_sections = section_1[-1] - section_2[0]
    section_1 = section_1 - offset_sections

    Smoothed_start = np.concatenate([section_1, section_2], axis=0)

    w = np.ones_like(Wavenumber)
    w[0:5] = 10
    w[-1:-5] = 10

    interp_data = []

    for array in Smoothed_start.T:
        smoothed_function = interpolate.UnivariateSpline(
            x=Wavenumber, y=array, k=5, w=w, s=0.005
        )
        smoothed_array = smoothed_function(Wavenumber)
        interp_data.append(smoothed_array)

    return Smoothed_start  # np.array(interp_data).T


Smoothed = smoothing_protocol(Data_init)

Smoothed_DF = pd.DataFrame(Smoothed, index=Wavenumber_full.T)

H2O_Data_smooth = smoothing_protocol(H2O_Data_init)
#%%
# for idx, spec in enumerate(H2O_Data_smooth[0, :]):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     plt.plot(Wavenumber, H2O_Data_smooth[:, idx])
#     plt.plot(Wavenumber, H2O_Data_init[:, idx])
#     ax.invert_xaxis()
#%%

Data_raw, Mean_baseline_raw = basic_scale_data(Data_init, Wavenumber)
Data_raw_smoothed, Mean_baseline_raw_smoothed = basic_scale_data(Smoothed, Wavenumber)

#%%

Scaled_Data = basic_scale_data(Data_init, Wavenumber)
Scaled_Smoothed = basic_scale_data(Smoothed, Wavenumber)

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Wavenumber, Scaled_Data[0])
plt.plot(Wavenumber, Scaled_Smoothed[0])
ax.invert_xaxis()

#%%
# Data_start = Smoothed
Data_start = Data_init

# %%

Data, Mean_baseline = scale_data(Data_start, Wavenumber)
Smooth_Data, Smooth_Mean_baseline = scale_data(Smoothed, Wavenumber)

#%%
# idx = 2
# Scaled_Data = basic_scale_data(Data_init, Wavenumber)
# Scaled_Smoothed = basic_scale_data(Smoothed, Wavenumber)

# fig, ax = plt.subplots(figsize=(12, 6))
# plt.plot(Wavenumber, Scaled_Data[0][:,idx])
# plt.plot(Wavenumber, Scaled_Smoothed[0][:,idx])
# ax.invert_xaxis()

# plt.ylim(-0.5,0)

# Consider breaking up the spectra into overlapping segments and fitting. That way the overlapping regions wont cause a sharp transition when stitched.

# %%

# idx = 0
# base = baseline_als(-Smooth_Data[30:None, idx], lam=1e2, p=1e-10, niter=20)
# plt.plot(Wavenumber, Data[:, idx])
# plt.plot(Wavenumber[30:None], -base)
# plt.plot(Wavenumber, Smooth_Data[:, idx])
# plt.plot((1635, 1635), (-0.075, 0.1))
# plt.plot((1275, 1275), (-0.075, 0.1))
# plt.plot((1515, 1515), (-0.075, 0.1))
# plt.plot((1430, 1430), (-0.075, 0.1))
# plt.ylim(-0.1, 0.2)

# Spectra to modify
# Smooth more: 9, 14, 15, 16
# Remove Peak: 0, 2, 10, 11,12
# Cut?:  3, 6 try to Smooth first because original spectra are nice-ish

# looking at unmodified
# Maybe Cut 10
# remove peak 11,12

# %%

# Good:1, 15, 16,
# 2, 4 smooth lam=1e1, p=1e-5,
# Baseline Smoothing Parameters
# idx 0: negative spec lam=1e5, p=5e-2
# idx 3: negative spec lam=1e5, p=5e-2
# idx 11: lam=5e5, p=3e-1, niter=20
# idx 12: lam=1e5, p=5e-2,
# idx 14: lam=1e5, p=5e-2
# %%

# Plotting all data  to make sure All CO2 Peaks are removed
plt.close("all")
# H2O_Data_raw, H2O_Mean_baseline_raw = basic_scale_data(H2O_Data_init, Wavenumber)
# for idx in range(H2O_Data_raw.shape[1]):

#     fig, ax = plt.subplots(figsize=(12, 6))
#     plt.plot((1430, 1430), (-0.5, 0))
#     plt.plot((1515, 1515), (-0.5, 0))
#     plt.plot((1630, 1630), (-0.5, 0))
#     plt.plot(Wavenumber, H2O_Data_raw[:, idx])

#     plt.title(f"idx:{idx}")
#     plt.ylim(-0.35, 0)
#     ax.invert_xaxis()
# Has CO2: 29,30,6, 23, 36,37
# Maybe:2, 33, 47
#%%
# idx = 35
# fig, ax = plt.subplots(figsize=(12, 6))
# plt.plot((1430, 1430), (-0.5, 0))
# plt.plot((1515, 1515), (-0.5, 0))
# plt.plot((1630, 1630), (-0.5, 0))
# plt.plot(Wavenumber, H2O_Data_raw[:, idx])

# plt.title(f"idx:{idx}")
# plt.ylim(-0.5, 0)
# ax.invert_xaxis()
# %%
plt.plot((1515, 1515), (-0.5, 0))

plt.plot((1630, 1630), (-0.5, 0))
plt.plot(Wavenumber, Data_raw_smoothed)
plt.ylim(-0.5, 0)


# H2O_Data, H2O_Mean_baseline = scale_data(H2O_Data, Wavenumber)
#%%
idx = 1

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Wavenumber, Data_raw[:, idx])
plt.plot(Wavenumber, Data_raw_smoothed[:, idx])
# plt.plot(Wavenumber, np.polyfit(Wavenumber, Data_raw_smoothed[:, idx], 6))
plt.plot((1635, 1635), (-0.5, 0.1))
plt.plot((1275, 1275), (-0.5, 0.1))
plt.plot((1515, 1515), (-0.5, 0.1))
plt.plot((1430, 1430), (-0.5, 0.1))
plt.ylim(-0.4, 0)
ax.invert_xaxis()

ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")

# %%
Smooth_Data, Smooth_Mean_baseline = scale_data(Smoothed, Wavenumber)

# Data, Mean_baseline = Data_smoothed, Mean_baseline_smoothed #Smooth_Data, Smooth_Mean_baseline
Data, Mean_baseline = Smooth_Data, Smooth_Mean_baseline
#%%
# Calculates the Principle components
pca = PCA(
    12,
)  # Number of PCA vectors to calculate

principalComponents = pca.fit(
    Data.T
)  # everything appears to work best with the raw data or raw data scaled

reduced_data = pca.fit(Data.T).transform(Data.T)
# %%
# plots the fraction of the variance explained by each PCA vector
fig, ax = plt.subplots(figsize=(12, 6))
variance = pca.explained_variance_
variance_norm = variance[0:-1] / np.sum(variance[0:-1])
plt.plot(variance_norm * 100, marker="o", linestyle="None")
ax.set_xlabel("Principle Component")
ax.set_ylabel(r"%Variance in CO_2 Free Baselines Selection")
pca.singular_values_
PCA_vectors = pca.components_
plt.savefig("PCA_Variance_plot.png")


# %%
# Plots the first several principle components

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
plt.plot(Wavenumber, PCA_vectors[0], label="PCA:1")
plt.plot(Wavenumber, PCA_vectors[1], label="PCA:2")
plt.plot(Wavenumber, PCA_vectors[2], label="PCA:3")
plt.plot(Wavenumber, PCA_vectors[3], label="PCA:4")
# plt.plot(Wavenumber, PCA_vectors[4], label="PCA:5")
# plt.plot(Wavenumber, PCA_vectors[5], label="PCA:6")
# plt.plot(Wavenumber, PCA_vectors[6], label="PCA:7")
# plt.plot(Wavenumber, PCA_vectors[7], label="PCA:8")

PCA3_Smooth = signal.savgol_filter(PCA_vectors[2], 65, 3)
plt.plot(Wavenumber, PCA3_Smooth, label="PCA3_Smooth")

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("component_plot.png")


# %%
# Save PCA database with removed 1630 water peak

Mean_base = pd.Series(Mean_baseline, index=Wavenumber, name="Average_Baseline")
Baseline_DF = pd.DataFrame(
    PCA_vectors[0:4].T, index=Wavenumber, columns=("PCA_1", "PCA_2", "PCA_3", "PCA_4")
)
Baseline_Database = pd.concat([Mean_base, Baseline_DF], axis=1)
Baseline_Database.to_csv("H2O_Free_Avg+PCA.csv")

# Baseline_Database.to_csv("Devol_Baseline_Avg+PCA.csv")
# Baseline_Database.to_csv("Smoothed_Baselines_H2O_Free_Avg+PCA.csv")


#%%
# Plots the baseline database in terms of PCA component.
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

#%%
# PCA vector ranges
PCA_max_Prior = reduced_data.max(axis=0)
PCA_min_Prior = reduced_data.min(axis=0)
# %%

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(Wavenumber, Mean_baseline + PCA_vectors[0], label="Average Baseline + PCA1")
plt.plot(Wavenumber, Mean_baseline - PCA_vectors[0], label="Average Baseline - PCA1")
# plt.plot(Wavenumber, Mean_baseline-PCA_vectors[0]*2, label = 'Average Baseline - PCA1 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA1+mean_plot_.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(Wavenumber, Mean_baseline + PCA_vectors[1], label="Average Baseline + PCA2")
plt.plot(Wavenumber, Mean_baseline - PCA_vectors[1], label="Average Baseline - PCA2")
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA2+mean_plot.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(Wavenumber, Mean_baseline + PCA_vectors[2], label="Average Baseline + PCA3")
plt.plot(Wavenumber, Mean_baseline - PCA_vectors[2], label="Average Baseline - PCA3")
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA3+mean_plot.png")
#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber, Mean_baseline + PCA_vectors[3] * 1, label="Average Baseline + PCA4"
)
plt.plot(
    Wavenumber, Mean_baseline - PCA_vectors[3] * 1, label="Average Baseline - PCA4"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA4+mean_plot.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber, Mean_baseline + PCA_vectors[4] * 1, label="Average Baseline + PCA5"
)
plt.plot(
    Wavenumber, Mean_baseline - PCA_vectors[4] * 1, label="Average Baseline - PCA5"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA5+mean_plot.png")
# %%
# Make Dataframes without Water peak for improved to fit
def H2O_peak_cut(df, wn_cut_low, wn_cut_high, return_DF=False):
    No_Peaks_Frame = df.drop(df[wn_cut_low:wn_cut_high].index)

    if return_DF == True:
        return No_Peaks_Frame

    No_Peaks_Wn = No_Peaks_Frame.index
    No_Peaks_Data = No_Peaks_Frame.values
    return No_Peaks_Wn, No_Peaks_Data


# %%
# wn_cut_low, wn_cut_high = (1521, 1730) # Before 3/22/21
# wn_cut_low, wn_cut_high = (1450, 1800) #after 3/22/21
# wn_cut_low, wn_cut_high = (1500, 1771)  # after 4/24/21  (1490, 1771)
wn_cut_low, wn_cut_high = (1510, 1771)  # after 5/12/21
Full_No_Peaks_Wn, Full_No_Peaks_Values = H2O_peak_cut(
    H2O_frame_select, wn_cut_low, wn_cut_high
)

n_PCA_vectors = 4  # 4 but after 5/10/21 2
H2O_free_PCA_DF = pd.DataFrame(PCA_vectors[0:n_PCA_vectors].T, index=Wavenumber)
PCA_No_Peaks_DF = H2O_peak_cut(H2O_free_PCA_DF, wn_cut_low, wn_cut_high, return_DF=True)

Average_baseline = pd.Series(Mean_baseline, index=Wavenumber)
Avg_BSL_no_peaks = H2O_peak_cut(
    Average_baseline, wn_cut_low, wn_cut_high, return_DF=True
)
Avg_BSL_no_peaks_Wn = Avg_BSL_no_peaks.index

tilt = pd.Series(np.arange(0, len(Average_baseline)), index=Wavenumber)
tilt_cut = H2O_peak_cut(tilt, wn_cut_low, wn_cut_high, return_DF=True)

# %%
# Synthetic Peaks Choose peak shape, position and width. In the future these will be fit parameters
Peak1 = pd.Series(
    Lorentzian(x=Wavenumber, center=1635, half_width=55, amp=1), index=Wavenumber
)
Peak2 = pd.Series(Gauss(x=Wavenumber, mu=1430, sd=30, A=1), index=Wavenumber)
Peak3 = pd.Series(Gauss(x=Wavenumber, mu=1515, sd=30, A=1), index=Wavenumber)
# %%
# Function to fit the baselines:
# uses the PCA components and the synthetic peaks to made linear combinations that fit the data. T


def Carbonate_baseline_fit(
    Spec,
    Average_baseline,
    PCA_vectors,
    n_PCA_vectors=2,
    Peak1=Peak1,
    Peak2=Peak2,
    Peak3=Peak3,
):

    PCA_DF = pd.DataFrame(PCA_vectors[0:n_PCA_vectors].T, index=Wavenumber)

    offset = pd.Series(np.ones(len(Peak2)), index=Wavenumber)
    tilt = pd.Series(np.arange(0, len(Peak2)), index=Wavenumber)

    # Baseline_Matrix = pd.concat([ Average_baseline, PCA_DF, offset, tilt, Peak2, Peak3], axis=1)

    # This line is only used if we are fitting the Water peak with the CO2 peak.
    Baseline_Matrix = pd.concat(
        [Average_baseline, PCA_DF, offset, tilt, Peak2, Peak3, Peak1], axis=1
    )

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, Spec)

    return Baseline_Matrix, fit_param


# %%
# plots the results


def plot_Baseline_results(Spectrum, Baseline_Matrix, fit_param, Wavenumber):
    modeled_basline = (
        np.matrix(Baseline_Matrix[:, 0:-3]) * fit_param[0][0:-3]
    )  # Ignores the Peaks in fit.

    # modeled_basline = np.matrix(
    #   Baseline_Matrix[:, 0:-2])*fit_param[0][0:-2]  # Ignores the Peaks in fit. Only for CO2 peaks

    fig, ax = plt.subplots(figsize=(12, 6))
    # plt.plot(Wavenumber,Spectrum.values, label = "Spectrum") for Pandas
    plt.plot(Wavenumber, Spectrum, label="Spectrum")

    plt.plot(Wavenumber, np.matrix(Baseline_Matrix) * fit_param[0], label="Modeled Fit")
    plt.plot(Wavenumber, modeled_basline, label="Baseline")

    # Peak1_amp = fit_param[0][-1]
    # Peak2_amp = fit_param[0][-3]
    # Peak3_amp = fit_param[0][-2]

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Absorbance")
    return ax


# %%
# Function to fit Water free baselines
# uses the PCA components and the synthetic peaks to mad elinear combinations that fit the data. T


def No_H2O_fit(Spec, Average_baseline, PCA_DF, tilt, Wavenumber=Wavenumber):

    offset = pd.Series(np.ones(len(Average_baseline)), index=Wavenumber)
    # tilt = pd.Series(np.arange(0, len(Average_baseline)), index=Wavenumber)

    Baseline_Matrix = pd.concat(
        [
            Average_baseline,
            PCA_DF,
            offset,
            tilt,
        ],
        axis=1,
    )

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, Spec)

    return Baseline_Matrix, fit_param


# %%
def plot_NoH2O_results(Spectrum, Baseline_Matrix, fit_param, Wavenumber):

    fig, ax = plt.subplots(figsize=(12, 6))
    # plt.plot(Wavenumber,Spectrum.values, label = "Spectrum") for Pandas
    plt.plot(Wavenumber, Spectrum, label="Spectrum")

    plt.plot(
        Wavenumber,
        np.matrix(Baseline_Matrix) * np.matrix(fit_param[0]).T,
        label="Modeled Fit",
    )

    # Peak1_amp = fit_param[0][-1]
    # Peak2_amp = fit_param[0][-3]
    # Peak3_amp = fit_param[0][-2]

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Absorbance")
    return ax


# %%
def tilt_fit(Spectrum, Baseline_Matrix, fit_param, Wavenumber):

    offset = pd.Series(np.ones(len(Wavenumber)), index=Wavenumber)
    tilt = pd.Series(np.arange(0, len(Wavenumber)), index=Wavenumber)

    Tilt_Matrix = pd.concat([offset, tilt], axis=1)

    Tilt_Matrix = np.matrix(Tilt_Matrix)

    fit_param = np.linalg.lstsq(Tilt_Matrix, Spec)


# %%
# This line subtracts the mean from your data

Test_Spectrum = Full_No_Peaks_Values[:, 19]
Baseline_Matrix, fit_param1 = No_H2O_fit(
    Spec=Test_Spectrum,
    Average_baseline=Avg_BSL_no_peaks,
    PCA_DF=PCA_No_Peaks_DF,
    Wavenumber=Avg_BSL_no_peaks_Wn,
    tilt=tilt_cut,
)


plot_NoH2O_results(
    Test_Spectrum,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param1,
    Wavenumber=Avg_BSL_no_peaks_Wn,
)

# %% Baseline Matrix without peaks removed
Average_baseline = pd.Series(Mean_baseline, index=Wavenumber)

Full_Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Data[:, 0],
    Average_baseline=Average_baseline,
    n_PCA_vectors=n_PCA_vectors,
    PCA_vectors=PCA_vectors,
)

Full_Baseline_Matrix = Full_Baseline_Matrix[:, 0:-3]
#%%
# make list of all fit parameters to use to replace data.
Fits = []
No_H2O_baseline = []


for spec in Full_No_Peaks_Values.T:
    Baseline_Matrix, fit_param = No_H2O_fit(
        Spec=spec,
        Average_baseline=Avg_BSL_no_peaks,
        PCA_DF=PCA_No_Peaks_DF,
        Wavenumber=Avg_BSL_no_peaks_Wn,
        tilt=tilt_cut,
    )

    Fits.append(fit_param)

    base_full = Full_Baseline_Matrix * np.matrix(fit_param[0]).T
    No_H2O_baseline.append(base_full)

# Scale Data
# No_H2O_baseline_Scaled, No_H2O_baseline_average = basic_scale_data(No_H2O_baseline,)
# %%

spec_idx = 14

fig, ax = plt.subplots(figsize=(12, 6))
H2O_init_range = H2O_Data_init[:, spec_idx][0] - H2O_Data_init[:, spec_idx][-1]
scale_data_No_H2O = No_H2O_baseline[spec_idx] / H2O_init_range
scale_H2O_Data = H2O_Data_init[:, spec_idx] / H2O_init_range

plt.plot(
    Wavenumber_full, scale_data_No_H2O - scale_H2O_Data.min(), label="No Water Baseline"
)
plt.plot(Wavenumber_full, scale_H2O_Data - scale_H2O_Data.min(), label="Spectrum")

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")

ax.set_ylim(0.2, 0.4)

plt.plot((1515, 1515), (0, 1))
plt.plot((1430, 1430), (0, 1))
plt.plot((1770, 1770), (0, 1))
# Exclude 2, 6, 23, 29, 30, 33, 36, 37


# %%
# Decisions For selection the best water only peaks 5/12
# Good: 0,
# Bad: 2, 3
# Maybe:1, 6?, 7,
# Notes
# look at 14, 35

plt.close("all")
# spec_idx = 1
for spec_idx in range(len(No_H2O_baseline)):
    fig, ax = plt.subplots(figsize=(12, 6))
    H2O_init_range = H2O_Data_init[:, spec_idx][0] - H2O_Data_init[:, spec_idx][-1]
    scale_data_No_H2O = No_H2O_baseline[spec_idx] / H2O_init_range
    scale_H2O_Data = H2O_Data_init[:, spec_idx] / H2O_init_range

    plt.plot(
        Wavenumber_full,
        scale_data_No_H2O - scale_H2O_Data.min(),
        label="No Water Baseline",
    )

    plt.plot(Wavenumber_full, scale_H2O_Data - scale_H2O_Data.min(), label="Spectrum")

    plt.legend()
    ax.invert_xaxis()
    ax.legend()
    plt.title(str(spec_idx))

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Absorbance")
    # ax.set_xlim(1800,1250)
    # ax.set_xlim(1750,1500)

    # ax.set_ylim(0.1, 0.4)

    plt.plot((1515, 1515), (0, 1))
    plt.plot((1430, 1430), (0, 1))
    plt.plot((1770, 1770), (0, 1))
# %%

No_H2O_baseline_df = pd.DataFrame(
    np.array(No_H2O_baseline)[:, :, 0].T,
    index=Wavenumber_full,
    columns=H2O_frame_select.columns,
)

H2O_Peaks = H2O_frame_select - No_H2O_baseline_df

H2O_select_smooth = pd.DataFrame(
    smoothing_protocol(H2O_frame_select.values),
    index=Wavenumber,
    columns=H2O_frame_select.columns,
)

# TODO stitch at multiple points at repeat this to make it smooth.
wn_stitch_low = 1451
wn_stitch_high = 1820

# Apply smooth junction. Maybe expand and reappend a wider series
cutout = No_H2O_baseline_df[wn_stitch_low:wn_stitch_high]
#%%
# smooth_cutout = np.apply_along_axis(savgol_filter,0,cutout, smooth_width = 31, poly_order = 3)

# smooth_cutout_df = pd.DataFrame(cutout, index =cutout.index, columns= cutout.columns)
smooth_cutout_df = cutout
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(smooth_cutout_df.index, smooth_cutout_df)
plt.plot(cutout)
#%%
# Stitching together data where peaks have been removed


offset_H2O_frame = (
    H2O_select_smooth.loc[wn_stitch_high - 2 : wn_stitch_high].values
    - H2O_select_smooth.loc[wn_stitch_low : wn_stitch_low + 2].values
)

offset_smooth_cutout = (
    smooth_cutout_df.loc[wn_stitch_high - 2 : wn_stitch_high].values
    - smooth_cutout_df.loc[wn_stitch_low : wn_stitch_low + 2].values
)
#%%
#%%
Scaled_data_cut = smooth_cutout_df * (
    offset_H2O_frame / offset_smooth_cutout
)  # Strech segment slightly to line up with edges being cut

offset_1500 = (
    H2O_select_smooth.loc[wn_stitch_low : wn_stitch_low + 2].values
    - Scaled_data_cut.loc[wn_stitch_low : wn_stitch_low + 2].values
)

offset_1800 = (
    H2O_select_smooth.loc[wn_stitch_high - 2 : wn_stitch_high].values
    - Scaled_data_cut.loc[wn_stitch_high - 2 : wn_stitch_high].values
)

# slope = (offset_1800.values - offset_1500.values)/ (wn_cut_high-wn_cut_low)
# cutout_tilt = np.tile(np.arange(0, len(cutout.index)) - len(cutout.index)/2,(len(cutout.columns),1))

# stitch_adjust = pd.DataFrame(np.multiply(slope, cutout_tilt.T ) + offset_1500.values, index = cutout.index, columns = cutout.columns)

# cut_adjusted = smooth_cutout_df + stitch_adjust
cut_adjusted = Scaled_data_cut + offset_1800

# I need to stretch not tilt!!!

fig, ax = plt.subplots(figsize=(12, 6))
# plt.plot(cut_adjusted)
plt.plot(cut_adjusted)

#%%

# TODO Repeat stitching on 2 sets of data cut at different points and average
# Smooth Original data before stitch
Peaks_removed_full_DF = H2O_select_smooth.drop(
    H2O_select_smooth.loc[wn_stitch_low:wn_stitch_high].index
).append(
    cut_adjusted,
)
Peaks_removed_full_DF.sort_index(inplace=True)
plt.plot(Peaks_removed_full_DF)

Peaks_removed_full_DF.to_csv("CO2_free_baselines_Water_removed_5-19-21.csv")
#%%


for idx, spec in enumerate(Peaks_removed_full_DF):  # Peaks_removed_full_DF.iteritems():
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(Peaks_removed_full_DF.iloc[:, idx])
    plt.plot(H2O_frame_select.iloc[:, idx])
    plt.ylim(0.2, 0.6)
#%%
# Devol Baselines saved


# def savgol_filter_short(x):
#     return signal.savgol_filter(x, 51, 3)

# TODO decide if the original water free spectra need to be added back in.
Peaks_removed_full = pd.concat((Peaks_removed_full_DF, Smoothed_DF), axis=1)

# Mayeb comment out this next section
# Peaks_removed_full = Peaks_removed_full_DF.apply(
#     func=signal.savgol_filter, args=(31, 3)
# )
Peaks_removed_full = pd.DataFrame(
    Peaks_removed_full,
    index=Peaks_removed_full_DF.index,
    # columns=Peaks_removed_full_DF.columns,
)


Data_start = Peaks_removed_full.values

# Peaks_removed_full.to_csv('CO2_free_baselines_Water_removed_5-4-21.csv')

#%%
# Smooth_Data, Smooth_Mean_baseline = scale_data(Peaks_removed_full_DF.values, Wavenumber)
Smooth_Data, Smooth_Mean_baseline = scale_data(Peaks_removed_full.values, Wavenumber)
# Data, Mean_baseline = Data_smoothed, Mean_baseline_smoothed #Smooth_Data, Smooth_Mean_baseline
Data, Mean_baseline = Smooth_Data, Smooth_Mean_baseline
#%%
# Calculates the Principle components
pca = PCA(
    12,
)  # Number of PCA vectors to calculate

principalComponents = pca.fit(
    Data.T
)  # everything appears to work best with the raw data or raw data scaled

reduced_data = pca.fit(Data.T).transform(Data.T)
# %%
# plots the fraction of the variance explained by each PCA vector
fig, ax = plt.subplots(figsize=(12, 6))
variance = pca.explained_variance_
variance_norm = variance[0:-1] / np.sum(variance[0:-1])
plt.plot(variance_norm * 100, marker="o", linestyle="None")
ax.set_xlabel("Principle Component")
ax.set_ylabel(r"%Variance in CO_2 Free Baselines Selection")
pca.singular_values_
PCA_vectors = pca.components_
plt.savefig("PCA_Variance_plot.png")

# TODO Smooth PCA before final output
# %%
# Plots the first several principle components

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
plt.plot(Wavenumber_full, PCA_vectors[0], label="PCA:1")
plt.plot(Wavenumber_full, PCA_vectors[1], label="PCA:2")
plt.plot(Wavenumber_full, PCA_vectors[2], label="PCA:3")
plt.plot(Wavenumber_full, PCA_vectors[3], label="PCA:4")
# plt.plot(Wavenumber, PCA_vectors[4], label="PCA:5")
# plt.plot(Wavenumber, PCA_vectors[5], label="PCA:6")
# plt.plot(Wavenumber, PCA_vectors[6], label="PCA:7")
# plt.plot(Wavenumber, PCA_vectors[7], label="PCA:8")


plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("component_plot.png")

#%%


# %%
# Save PCA database with removed 1630 water peak

# Mean_base = pd.Series(Mean_baseline, index = Wavenumber_full, name = 'Average_Baseline')
# Baseline_DF = pd.DataFrame(PCA_vectors[0:4].T, index = Wavenumber_full, columns = ('PCA_1','PCA_2','PCA_3','PCA_4'))
# Baseline_Database = pd.concat([Mean_base, Baseline_DF], axis = 1)
# Baseline_Database.to_csv("Devol_Baseline_Avg+PCA.csv")
# Baseline_Database.to_csv("Smoothed_Baselines_H2O_Free_Avg+PCA.csv")


#%%
# Plots the baseline database in terms of PCA component.
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

#%%
# PCA vector ranges
PCA_max_Prior = reduced_data.max(axis=0)
PCA_min_Prior = reduced_data.min(axis=0)
# %%

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber_full, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber_full, Mean_baseline + PCA_vectors[0], label="Average Baseline + PCA1"
)
plt.plot(
    Wavenumber_full, Mean_baseline - PCA_vectors[0], label="Average Baseline - PCA1"
)
# plt.plot(Wavenumber, Mean_baseline-PCA_vectors[0]*2, label = 'Average Baseline - PCA1 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA1+mean_plot_.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber_full, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber_full, Mean_baseline + PCA_vectors[1], label="Average Baseline + PCA2"
)
plt.plot(
    Wavenumber_full, Mean_baseline - PCA_vectors[1], label="Average Baseline - PCA2"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA2+mean_plot.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber_full, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber_full, Mean_baseline + PCA_vectors[2], label="Average Baseline + PCA3"
)
plt.plot(
    Wavenumber_full, Mean_baseline - PCA_vectors[2], label="Average Baseline - PCA3"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA3+mean_plot.png")
#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber_full, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber_full, Mean_baseline + PCA_vectors[3] * 1, label="Average Baseline + PCA4"
)
plt.plot(
    Wavenumber_full, Mean_baseline - PCA_vectors[3] * 1, label="Average Baseline - PCA4"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("PCA4+mean_plot.png")


# %%
idx = 2

fig, ax = plt.subplots(figsize=(12, 6))
# plt.plot(Peaks_removed_full.iloc[:,idx])
# plt.plot(H2O_frame_select.iloc[:,idx])
water_peak = H2O_frame_select.iloc[:, idx] - Peaks_removed_full.iloc[:, idx]
plt.plot(water_peak / water_peak.max())
plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")

#%%


fig, ax = plt.subplots(figsize=(12, 6))

# plt.plot(Peaks_removed_full.iloc[:,idx])
# plt.plot(H2O_frame_select.iloc[:,idx])
Peak_only = H2O_frame_select - Peaks_removed_full_DF

plt.plot(Peak_only / Peak_only.max(axis=0))

plt.xlim(1500, 1750)

ax.invert_xaxis()

ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")

#%%


fig, ax = plt.subplots(figsize=(12, 6))

# plt.plot(Peaks_removed_full_DF.iloc[:,idx])
# plt.plot(H2O_frame_select.iloc[:,idx])
Peak_only = H2O_frame_select - Peaks_removed_full_DF

plt.plot(Peak_only / Peak_only.max(axis=0))

plt.xlim(1500, 1750)

ax.invert_xaxis()

ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
# %%
# PCA Peaks

wn_low = 1400
wn_high = 1820  # 1771  # 1750 #this might be were the issue is

# wn_low = 1800
# wn_low = 1500
# Wavenumber = frame.loc[wn_low:wn_high].index

Peaks_Only_full = H2O_frame_select - Peaks_removed_full_DF

# Exclude 2, 6, 23, 29, 30, 33, 36, 37

bad_idx = [3, 4, 9, 21, 28, 41, 45, 47, 49, 39, 52, 53, 2, 6, 23, 29, 30, 33, 36, 37]
Good_peaks = Peaks_Only_full.drop(Peaks_Only_full.iloc[:, bad_idx], axis=1)

# bad_idx2 = [15, 21, 30] # old numbers
bad_idx2 = [13, 18, 24]  # 33?
Good_peaks = Good_peaks.drop(Good_peaks.iloc[:, bad_idx2], axis=1)

bad_idx3 = [30]  # 33?
Good_peaks = Good_peaks.drop(Good_peaks.iloc[:, bad_idx3], axis=1)

Peaks_Only = Good_peaks.loc[wn_low:wn_high]
Wavenumber = Peaks_Only.index


Peak_1630_start = Peaks_Only.values

Peaks_Only.to_csv("1630_Peaks_Baseline_removed.csv")
# %%
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(Good_peaks / Good_peaks.max(axis=0))

# plt.plot(Wavenumber, Lorentzian(Wavenumber, 1635, 40, amp=1), color="g", linewidth=3)

# plt.plot(Wavenumber, Gauss(Wavenumber, 1635, 20, A=1), color = 'r', linewidth = 3)
plt.xlim(1500, 1800)  # was 1750

ax.invert_xaxis()

ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")


# %%
fig, ax = plt.subplots(figsize=(12, 8))
idx = 19
plt.plot(Good_peaks.iloc[:, idx] / Good_peaks.iloc[:, idx].max(axis=0))

# bad_indexes = 15, 21,30
idx = 7
plt.plot(Good_peaks.iloc[:, idx] / Good_peaks.iloc[:, idx].max(axis=0))

# plt.plot(Wavenumber, Lorentzian(Wavenumber, 1635, 50, amp=1), color = 'g', linewidth = 3)

# plt.plot(Wavenumber, Gauss(Wavenumber, 1635, 20, A=1), color = 'r', linewidth = 3)
plt.xlim(1500, 1800)  # was 1750

ax.invert_xaxis()

ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")


# %%
for idx, spec in enumerate(Good_peaks):
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.plot(Good_peaks.iloc[:, idx] / Good_peaks.iloc[:, idx].max(axis=0))
    plt.title(idx)

    # bad_indexes = 15, 21,30
    idx = 7
    plt.plot(Good_peaks.iloc[:, idx] / Good_peaks.iloc[:, idx].max(axis=0))

    # plt.plot(Wavenumber, Lorentzian(Wavenumber, 1635, 50, amp=1), color = 'g', linewidth = 3)

    # plt.plot(Wavenumber, Gauss(Wavenumber, 1635, 20, A=1), color = 'r', linewidth = 3)
    plt.xlim(1500, 1800)  # was 1750

    ax.invert_xaxis()

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Absorbance")
# %%
def Subtract_mean(Data, Wavenumber):
    """
    subtracts the mean spectrum from the data
    """
    data_range = Peak_1630_start.max(axis=0) - Peak_1630_start.min(axis=0)
    Data = Data / data_range
    Mean_baseline = Data.mean(axis=1)

    Data = Data - np.array([Mean_baseline]).T
    return Data, Mean_baseline


Data, Mean_Peak = Subtract_mean(Peak_1630_start, Wavenumber)

# H2O_Data, H2O_Mean_baseline = scale_data(H2O_Data, Wavenumber)

# %%
# Calculates the Principle components
pca = PCA(
    12,
)  # Number of PCA vectors to calculate

principalComponents = pca.fit(
    Data.T
)  # everything appears to work best with the raw data or raw data scaled

reduced_data = pca.fit(Data.T).transform(Data.T)
# %%
# plots the fraction of the variance explained by each PCA vector
fig, ax = plt.subplots(figsize=(12, 6))
variance = pca.explained_variance_
variance_norm = variance[0:-1] / np.sum(variance[0:-1])
plt.plot(variance_norm * 100, marker="o", linestyle="None")
ax.set_xlabel("Principle Component")
ax.set_ylabel(r"%Variance in CO_2 Free Baselines Selection")
pca.singular_values_
PCA_vectors = pca.components_
# plt.savefig("PCA_Variance_plot.png")

# save PCA
# %%
# Save PCA database with removed 1630 water peak

# Mean_base = pd.Series(Mean_Peak, index = Wavenumber, name = 'Average_Baseline')
# Baseline_DF = pd.DataFrame(PCA_vectors[0:3].T, index = Wavenumber, columns = ('PCA_1','PCA_2','PCA_3','PCA_4'))
# Baseline_Database = pd.concat([Mean_base, Baseline_DF], axis = 1)
# Baseline_Database.to_csv("Devol_Baseline_Avg+PCA.csv")
# Baseline_Database.to_csv("Water_Peak_1630_Free_Avg+PCA.csv")

# %%
# Plots the first several principle components

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
plt.plot(Wavenumber, PCA_vectors[0], label="PCA:1")
plt.plot(Wavenumber, PCA_vectors[1], label="PCA:2")
plt.plot(Wavenumber, PCA_vectors[2], label="PCA:3")
# plt.plot(Wavenumber, PCA_vectors[3], label="PCA:3")
# plt.plot(Wavenumber, PCA_vectors[4], label="PCA:5")
# plt.plot(Wavenumber, PCA_vectors[5], label="PCA:6")
# plt.plot(Wavenumber, PCA_vectors[6], label="PCA:7")
# plt.plot(Wavenumber, PCA_vectors[7], label="PCA:8")

PeakL = pd.Series(
    Lorentzian(x=Wavenumber, center=1638, half_width=35, amp=1), index=Wavenumber
)

# plt.plot(PeakL*.21, label= "Lorentzian Peak")

# PeakG = pd.Series(Gauss(x=Wavenumber, mu=1638, sd=29, A=1), index=Wavenumber)

# plt.plot(PeakG*.21, label= "Gaussian Peak")

# plt.plot((PeakG + PeakL)*.105, label= "Gaussian Peak")
# deriv = (PCA_vectors[0] - np.roll(PCA_vectors[0],1)) / (Wavenumber[1]-Wavenumber[0])
# plt.plot(Wavenumber, deriv*40)
# deriv2 = (deriv - np.roll(deriv,1)) / (Wavenumber[1]-Wavenumber[0])
# plt.plot(Wavenumber, deriv2*400)

# np.where( Wavenumber>1400 & Wavenumber< 1500,PCA1_Smooth,np.nan())
PCA1_Smooth = signal.savgol_filter(PCA_vectors[0], 55, 6)
plt.plot(Wavenumber, PCA1_Smooth, label="PCA1_Smooth")


PCA2_Smooth = signal.savgol_filter(PCA_vectors[1], 55, 6)
plt.plot(Wavenumber, PCA2_Smooth, label="PCA2_Smooth")


PCA3_Smooth = signal.savgol_filter(PCA_vectors[2], 55, 6)
plt.plot(Wavenumber, PCA3_Smooth, label="PCA3_Smooth")

plt.plot(Wavenumber, Mean_Peak, label="Average")

plt.xlim(1250, 1800)

# PeakL = pd.Series(
#     Lorentzian(x=Wavenumber, center=1638, half_width=50, amp=1), index=Wavenumber
# )

# plt.plot(PeakL, label="Lorentzian Peak")

# PeakG = pd.Series(Gauss(x=Wavenumber, mu=1638, sd=23, A=1), index=Wavenumber)

# plt.plot(PeakG, label="Gaussian Peak")

# PeakG2 = pd.Series(Gauss(x=Wavenumber, mu=1638, sd=30, A=1), index=Wavenumber)

# plt.plot(PeakG2, label="Gaussian Peak")
#
# plt.plot(Wavenumber, PCA_vectors[0], label="PCA:1")
plt.legend()

ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
ax.invert_xaxis()

ax.plot((1515, 1515), (-0.2, 0.2))

plt.savefig("1630_Peak_shape_+Gaussian_Full.png")

# %%
# Smoothed_cut_PCA1 = np.where( ((Wavenumber>1550) & (Wavenumber< 1800)),PCA1_Smooth,0)

# plt.plot(Wavenumber,Smoothed_cut_PCA1)
# Smoothed_cut_PCA1 = np.where( ((Wavenumber>1400) & (Wavenumber< 1500)),PCA1_Smooth,np.nan())
Peak_Components = np.array([Mean_Peak, PCA1_Smooth, PCA2_Smooth, PCA3_Smooth])

Peak_comps_df = pd.DataFrame(
    Peak_Components.T,
    index=Wavenumber,
    columns=[
        "Average_1630_Peak",
        "1630_Peak_PCA_1",
        "1630_Peak_PCA_2",
        "1630_Peak_PCA_3",
    ],
)

Zeros = pd.DataFrame(
    np.zeros_like(Wavenumber_full, shape=(len(Wavenumber_full), 4)),
    index=Wavenumber_full,
    columns=[
        "Average_1630_Peak",
        "1630_Peak_PCA_1",
        "1630_Peak_PCA_2",
        "1630_Peak_PCA_3",
    ],
)
# TODO this might be the place to change the boundaries to make it a bigger
# wn_cut_low, wn_cut_high
Peak_comps_df_full = pd.concat(
    [
        Zeros.loc[wn_low_init:wn_stitch_low],
        Peak_comps_df.loc[wn_stitch_low:wn_stitch_high],
        Zeros.loc[wn_stitch_high:wn_high_init],
    ],
)
Peak_comps_df_full.to_csv("Water_Peak_1635.csv")

# %%


def Water_Peak_fit(Spec, PCA_vectors=Peak_comps_df_full, n_PCA_vectors=2):

    PCA_DF = PCA_vectors.iloc[:, 0:n_PCA_vectors]

    Peak_Matrix = np.matrix(PCA_DF)

    fit_param = np.linalg.lstsq(Peak_Matrix, Spec)

    return Peak_Matrix, fit_param


# %%
def plot_peak_results(Spectrum, Peak_Matrix, fit_param, Wavenumber):

    plt.plot(Wavenumber, Peak_Matrix * np.matrix(fit_param[0]).T, label="Modeled Fit")
    plt.plot(Wavenumber, spec, label="Spectrum")

    # plt.annotate(fit_param[0][0:n_PCA_vectors], xy= (1,0))

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Absorbance")
    return ax


# %%

spec = Good_peaks.values.T[28]
Peak_Matrix, fit_param = Water_Peak_fit(
    spec, PCA_vectors=Peak_comps_df_full, n_PCA_vectors=2
)
plot_peak_results(
    spec,
    Peak_Matrix,
    fit_param,
    Wavenumber_full,
)

# %%

# TODO write these to a PDF with info about number of PCA's
fig.clear("all")
n_PCA_vectors = 2
fit_param_list = []
for idx, spec in enumerate(Good_peaks.values.T):
    fig, ax = plt.subplots(figsize=(12, 6))
    Peak_Matrix, fit_param = Water_Peak_fit(
        spec, PCA_vectors=Peak_comps_df_full, n_PCA_vectors=n_PCA_vectors
    )
    plot_peak_results(spec, Peak_Matrix, fit_param, Wavenumber_full)
    plt.plot((1515, 1515), (0, 0.1))
    plt.title(str(idx))
    fit_param_list.append(fit_param[0])

# %%

# TODO write these to a PDF with info about number of PCA's
fig.clear("all")
n_PCA_vectors = 2
fit_param_list = []
for idx, spec in enumerate(Peaks_Only_full.values.T):
    fig, ax = plt.subplots(figsize=(12, 6))
    Peak_Matrix, fit_param = Water_Peak_fit(
        spec, PCA_vectors=Peak_comps_df_full, n_PCA_vectors=n_PCA_vectors
    )
    plot_peak_results(spec, Peak_Matrix, fit_param, Wavenumber_full)
    plt.plot((1515, 1515), (0, 0.1))
    plt.title(str(idx))
    fit_param_list.append(fit_param[0])
#%%
pd.DataFrame(fit_param_list)
# %%
#


# %%
# Save PCA database with removed 1630 water peak
"""
Mean_base = pd.Series(Mean_baseline, index = Wavenumber, name = 'Average_Baseline')
Baseline_DF = pd.DataFrame(PCA_vectors[0:4].T, index = Wavenumber, columns = ('PCA_1','PCA_2','PCA_3','PCA_4'))
Baseline_Database = pd.concat([Mean_base, Baseline_DF], axis = 1)
Baseline_Database.to_csv("Baseline_Avg+PCA.csv")
"""
#%%
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

#%%
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(reduced_data[:, 1], reduced_data[:, 2])
ax.set_xlabel("PCA 2")
ax.set_ylabel("PCA 3")
# %%

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(Wavenumber, Mean_baseline + PCA_vectors[0], label="Average Baseline + PCA1")
plt.plot(Wavenumber, Mean_baseline - PCA_vectors[0], label="Average Baseline - PCA1")
# plt.plot(Wavenumber, Mean_baseline-PCA_vectors[0]*2, label = 'Average Baseline - PCA1 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("1630_PCA1+mean_plot_.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber, Mean_baseline + PCA_vectors[1] * 0.05, label="Average Baseline + PCA2"
)
plt.plot(
    Wavenumber, Mean_baseline - PCA_vectors[1] * 0.05, label="Average Baseline - PCA2"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("1630_PCA2+mean_plot.png")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber, Mean_baseline + PCA_vectors[2] * 0.04, label="Average Baseline + PCA2"
)
plt.plot(
    Wavenumber, Mean_baseline - PCA_vectors[2] * 0.04, label="Average Baseline - PCA2"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("1630_PCA4+mean_plot.png")
#%%

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(Wavenumber, Mean_baseline, label="Average Baseline", linewidth=3)
plt.plot(
    Wavenumber, Mean_baseline + PCA_vectors[3] * 0.02, label="Average Baseline + PCA2"
)
plt.plot(
    Wavenumber, Mean_baseline - PCA_vectors[3] * 0.02, label="Average Baseline - PCA2"
)
# plt.plot(Mean_baseline-PCA_vectors[1]*2, label = 'Average Baseline + PCA2 *2')

plt.legend()
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Absorbance")
plt.savefig("1630_PCA3+mean_plot.png")
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Peaks_removed_full_DF, marker="o", markersize=3, linestyle=None)
ax.set_xlim(1750, 1850)
ax.set_ylim(0.3, 0.5)


#%%
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(Peaks_removed_full_DF, marker="o", markersize=1, linestyle=None)
ax.set_xlim(1450, 1550)
ax.set_ylim(0.3, 0.5)
# In order to stitch I need to have the yint from offset_1500 and add
# gh = cutout_tilt * slope + smooth_cutout
# Data = H2O_Peaks.values

#%%


# %%
# Loads One of Sarah's Spectra. We can easily modify this to load them all like I have done with Dan's below

Sarah_path = Path.cwd().joinpath(
    "Sarah's FTIR Spectra/Fuego2018FTIRSpectra_Transmission/AC4_OL49_021920_30x30_H2O_b.CSV"
)


def open_spectrum(path, wn_high=wn_high, wn_low=wn_low):
    df = pd.read_csv(path, index_col=0, header=0, names=["Wavenumber", "Absorbance"])
    spec = df.loc[wn_low:wn_high]
    return spec


Spectrum = open_spectrum(Sarah_path)
# %%

# This line subtracts the mean from your data
sarahFTIR = StandardScaler(with_std=False).fit_transform(Spectrum)
sarahFTIR = Spectrum
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=sarahFTIR,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    sarahFTIR,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param,
    Wavenumber=Wavenumber,
)
plt.title("AC4_OL49")
plt.savefig("AC4_OL49_baselinefit.png")
# %%

# Loads All of Dan's data to fit

Path_Dan_test = Path.cwd().joinpath("Dans_Examples")
all_files = Path_Dan_test.rglob("*.csv")
li = []

for filename in all_files:
    df = pd.read_csv(
        filename, index_col=0, header=0, names=["Wavenumber", "Absorbance"]
    )
    li.append(df)


Dan_FTIR = pd.concat(
    li,
    axis=1,
)

Dan_FTIR_select = Dan_FTIR.loc[wn_low:wn_high]
# %%
# Processing Dan's data

Spectrum1 = Dan_FTIR_select.iloc[:, 0].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1
)
Spectrum1 = StandardScaler(with_std=False).fit_transform(Spectrum1)

Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum1,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    Spectrum1,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param,
    Wavenumber=Wavenumber,
)
plt.title("CL05MI01.csv")
plt.savefig("CL05MI01_baselinefit.png")

# %%


Spectrum2 = Dan_FTIR_select.iloc[:, 1].values.reshape(
    len(Dan_FTIR_select.iloc[:, 0]), 1
)
# Spectrum2 = StandardScaler(with_std=False).fit_transform(Spectrum2)

Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum2,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    Spectrum2,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param1,
    Wavenumber=Wavenumber,
)
plt.title("CL05MI02.csv")
plt.savefig("CL05MI02_baselinefit.png")
# %%
Spectrum = Dan_FTIR_select.iloc[:, 2].values.reshape(len(Dan_FTIR_select.iloc[:, 0]), 1)

Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    Spectrum,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param,
    Wavenumber=Wavenumber,
)
plt.title("CL05MI08.csv")
plt.savefig("CL05MI08_baselinefit.png")
# %%
Spectrum = Dan_FTIR_select.iloc[:, 3].values.reshape(len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    Spectrum,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param,
    Wavenumber=Wavenumber,
)
plt.title("CV04MI03.csv")
plt.savefig("CV04MI03_baselinefit.png")
# %%
Spectrum = Dan_FTIR_select.iloc[:, 4].values.reshape(len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    Spectrum,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param,
    Wavenumber=Wavenumber,
)
plt.title("FR02MI01.csv")
plt.savefig("FR02MI01_baselinefit_.png")
# %%
Spectrum = Dan_FTIR_select.iloc[:, 5].values.reshape(len(Dan_FTIR_select.iloc[:, 0]), 1)
Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    Spec=Spectrum,
    Average_baseline=Average_baseline,
    n_PCA_vectors=2,
    PCA_vectors=PCA_vectors,
)


plot_Baseline_results(
    Spectrum,
    Baseline_Matrix=Baseline_Matrix,
    fit_param=fit_param,
    Wavenumber=Wavenumber,
)
plt.savefig("FR04MI01_baselinefit.png")
# %%
# %%

# %%
