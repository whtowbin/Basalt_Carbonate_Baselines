# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pykrige import OrdinaryKriging

# %%
def Krige_Interpolate(X, Y, new_X):
    uk = OrdinaryKriging(
        X,
        np.zeros(X.shape),
        Y,
        pseudo_inv=True,
        weight=True,
        # nlags=3,
        # exact_values=False,
        variogram_model="linear",
        variogram_parameters={"slope": 3e-5, "nugget": 0.0002}
        # variogram_model="gaussian",
        # variogram_parameters={"sill": 1e2, "range": 1e2, "nugget": 0.0006},
    )

    y_pred, y_std = uk.execute("grid", new_X, np.array([0.0]), backend="loop")
    y_pred = np.squeeze(y_pred)
    y_std = np.squeeze(y_std)

    return new_X, y_pred, y_std


# %%

newman_raw = pd.read_csv("Newman_Baseline_raw.csv", index_col=0)
NMT = pd.read_csv("STD_NMT80-1-3_021720_256s_40x40_a.CSV", index_col=0)
# %%

NMT[1249:2200]
# %%
newman_interp = Krige_Interpolate(
    X=newman_raw.index, Y=newman_raw.values, new_X=NMT[1249:2200].index
)
# %%

# %%
interp_DF = pd.DataFrame(newman_interp).T
interp_DF.to_csv("Newman_Baseline_interp.csv")
# %%
