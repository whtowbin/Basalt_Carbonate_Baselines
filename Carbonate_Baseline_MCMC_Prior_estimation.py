def Prior_fit(
    Spec,
    PCAmatrix,
    n_PCA_vectors=4,
    Peak_1635_PCA
):
"""
Function to supply inital parameter guesses and priors for Carbonate Baseline mcmc fits. 

"""
    Peak2 = pd.Series(Gauss(x=Wavenumber, mu=1430, sd=30, A=1), index=Wavenumber)
    Peak3 = pd.Series(Gauss(x=Wavenumber, mu=1515, sd=30, A=1), index=Wavenumber)

    PCA_DF = pd.DataFrame(PCAmatrix[0:n_PCA_vectors].T, index=Wavenumber)

    offset = pd.Series(np.ones(len(Peak2)), index=Wavenumber)
    tilt = pd.Series(np.arange(0, len(Peak2)), index=Wavenumber)

    # This line is only used if we are fitting the Water peak with the CO2 peak.
    Baseline_Matrix = pd.concat(
        [PCA_DF, peak_G1430, peak_G1515, tilt, offset, Peak_1635_PCA[:,0:-1]], axis=1
    )

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, Spec)

    Baseline_Params = fit_param[0:n_PCA_vectors]
    Peak1430_height = fit_param[n_PCA_vectors+1]
    Peak15150_height = fit_param[n_PCA_vectors+2]
    tilt_fit = fit_param[n_PCA_vectors+3]
    offset_fit = fit_param[n_PCA_vectors+4]
    Peak_1635_fit = fit_param[-3:None]
    ['Avg_BsL', "PCA_1", "PCA_2", "PCA_3", "PCA_4",'peak_G1430', 'std_G1430','G1430_amplitude', 'peak_G1515', 'std_G1515', 'G1515_amplitude', 'slope', 'offset', 'Average_1635Peak', '1635PeakPCA1', '1635PeakPCA2', '1635PeakPCA3']
    Params = np.concat(Baseline_Params, 1430, 30, Peak1430_height, 1515, 30, Peak15150_height,  tilt_fit, offset_fit, Peak_1635_fit )

    return Params

