#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:36:19 2025

@author: qcu21637
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cycle annuel moyen des précipitations mensuelles (1990-2014) - Région lyonnaise
- CPRCM en vert, 3 RCM en nuances de bleu, SIM2 en rouge
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Répertoires et fichiers
# -------------------------------
OUT_DIR = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/Figures/Cycles_annuels_moyens/RAIN/"
os.makedirs(OUT_DIR, exist_ok=True)

FN_SIM   = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/SIM2_1990_2014_rl_converted.nc"
FN_CPRCM = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/CPRCM_AROME_8km_rl_converted.nc"
FN_RCMs = [
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_ALADIN64E1_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_HCLIM43ALADIN_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_RACMO32E_rl_converted.nc"
]

# -------------------------------
# Lecture datasets
# -------------------------------
ds_sim   = xr.open_dataset(FN_SIM)
ds_cprcm = xr.open_dataset(FN_CPRCM)
ds_rcms  = [xr.open_dataset(f) for f in FN_RCMs]

datasets = [ds_cprcm] + ds_rcms + [ds_sim]
labels   = ["CPRCM_AROME", "RCM_ALADIN64E1", "RCM_HCLIM43ALADIN", "RCM_RACMO32E", "SIM2"]
colors   = ["green", "#a6cee3", "#1f78b4", "#08306b", "red"]

# -------------------------------
# Paramètres temporels
# -------------------------------
start_date = "1990-01-01"
end_date   = "2014-12-31"

# -------------------------------
# Fonction utilitaire
# -------------------------------
def extract_monthly_precip(ds, var_name="RAINTOT"):
    ds_sel = ds.sel(time=slice(start_date, end_date))
    monthly = ds_sel[var_name].resample(time="MS").sum()
    return monthly

def apply_corrections(var_name, arr, ds_label):
    """Pour RCM ou CPRCM si nécessaire"""
    if isinstance(arr, xr.DataArray):
        arr = arr.values
    return arr  # RAINTOT déjà en mm

# -------------------------------
# Calcul et tracé du cycle annuel moyen
# -------------------------------
plt.figure(figsize=(12, 6))

for ds, label, color in zip(datasets, labels, colors):
    monthly_precip = extract_monthly_precip(ds)
    monthly_precip = apply_corrections("RAINTOT", monthly_precip, label)

    # Moyenne spatiale sur la région
    monthly_mean = np.nanmean(monthly_precip, axis=(1,2))

    # Calcul moyenne par mois
    months = np.arange(1,13)
    # reshape en années x 12
    n_years = monthly_mean.size // 12
    monthly_mean = monthly_mean[:n_years*12].reshape(n_years,12)
    climatology = np.mean(monthly_mean, axis=0)

    plt.plot(months, climatology, color=color, label=label, linewidth=2)

plt.xlabel("Mois")
plt.ylabel("Précipitations moyennes (mm/mois)")
plt.title("Cycle annuel moyen des précipitations (1990-2014) - Région lyonnaise", fontsize=14, fontweight="bold")
plt.grid(alpha=0.3)
plt.xticks(months)
plt.legend()
plt.tight_layout()

out_file = os.path.join(OUT_DIR, "CycleAnnuelMoyen_RAIN_1990_2014.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f" → sauvegardé : {out_file}")
