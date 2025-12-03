#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:41:06 2025

@author: qcu21637
"""

### Période d'étude : 1990-2014
### A partir du meme GCM : 3 x RCM => en nuances de bleu
###                         1 CPRCM => en vert
### Obs historiques = SIM2 


########################### 1 - Distribution pluie mensuelle 1990-2014 région lyonnaise totale ##############################

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

# --- Répertoires et fichiers ---

OUT_FIG = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/Figures/Distrib_mensuelle_variable_zone/"
FN_SIM   = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/SIM2_1990_2014_rl_converted.nc"
FN_CPRCM = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/CPRCM_AROME_8km_rl_converted.nc"
FN_RCMs = [
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_ALADIN64E1_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_HCLIM43ALADIN_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_RACMO32E_rl_converted.nc"
]

# --- Lecture datasets ---
ds_sim   = xr.open_dataset(FN_SIM)
ds_cprcm = xr.open_dataset(FN_CPRCM)
ds_rcms  = [xr.open_dataset(f) for f in FN_RCMs]

# --- Paramètres temporels ---
start_date = "1990-01-01"
end_date   = "2014-12-31"

# --- Extraction précipitations mensuelles ---
def extract_monthly_precip(ds, var_name="RAINTOT"):
    ds_sel = ds.sel(time=slice(start_date, end_date))
    monthly = ds_sel[var_name].resample(time="MS").sum()
    return monthly

def flatten_zone(arr):
    """Mettre toutes les grilles en un seul vecteur 1D pour la distribution globale."""
    flat = arr.values.reshape(arr.shape[0], -1)
    flat = flat.flatten()
    flat = flat[~np.isnan(flat)]
    return flat

# --- Préparer les données ---
datasets = [ds_cprcm] + ds_rcms + [ds_sim]
labels = ["CPRCM_AROME", "RCM_ALADIN64E1", "RCM_HCLIM43ALADIN", "RCM_RACMO32E", "SIM2"]
colors = ["green", "#a6cee3", "#1f78b4", "#08306b", "red"] 

plt.figure(figsize=(12, 6))

for ds, label, color in zip(datasets, labels, colors):
    monthly_precip = extract_monthly_precip(ds, var_name="RAINTOT")
    data_flat = flatten_zone(monthly_precip)
    
    # échantillonnage si trop grand
    max_samples = 500_000
    if data_flat.size > max_samples:
        idx = np.random.choice(data_flat.size, max_samples, replace=False)
        data_flat = data_flat[idx]
    
    # KDE
    kde = gaussian_kde(data_flat)
    xs = np.linspace(np.min(data_flat), np.max(data_flat), 300)
    plt.plot(xs, kde(xs), color=color, label=label, linewidth=2)

plt.xlabel("Précipitations mensuelles (mm)")
plt.ylabel("Densité")
plt.title("Distribution mensuelle des précipitations (1990-2014) - Région lyonnaise")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()

out_file = os.path.join(OUT_FIG, "DistributionMensuelle_RAIN_1990_2014.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f" → sauvegardé : {out_file}")

#%%
########################### 2 - Distribution pluie mensuelle par zone ##############################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution mensuelle des précipitations par zone
pour CPRCM, RCMs et SIM (1990-2014)
"""

import xarray as xr
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

# -------------------------------
# Répertoires et fichiers
# -------------------------------
OUT_FIG = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/Figures/Distrib_mensuelle_variable_zone/"
os.makedirs(OUT_FIG, exist_ok=True)

FN_SIM   = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/SIM2_1990_2014_rl_converted.nc"
FN_CPRCM = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/CPRCM_AROME_8km_rl_converted.nc"
FN_RCMs = [
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_ALADIN64E1_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_HCLIM43ALADIN_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_RACMO32E_rl_converted.nc"
]

SHAPEFILE = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/shapefile_zones_region_lyonnaise/8km/grille_8km_centree_classif.shp"

# -------------------------------
# Lecture datasets
# -------------------------------
ds_sim   = xr.open_dataset(FN_SIM)
ds_cprcm = xr.open_dataset(FN_CPRCM)
ds_rcms  = [xr.open_dataset(f) for f in FN_RCMs]

datasets = [ds_cprcm] + ds_rcms + [ds_sim]
labels = ["CPRCM_AROME", "RCM_ALADIN64E1", "RCM_HCLIM43ALADIN", "RCM_RACMO32E", "SIM2"]
colors = ["green", "#a6cee3", "#1f78b4", "#08306b", "red"]

# -------------------------------
# Lecture shapefile zones
# -------------------------------
gdf = gpd.read_file(SHAPEFILE)

# Mapping zones numériques → noms explicites
zone_map = {
    1: "Artificielle",
    2: "Agricole",
    3: "Forêt"
}

# Liste des zones présentes dans le shapefile
zone_names_numeric = gdf["classif_cl"].unique()
zone_names = [zone_map.get(z, f"Zone_{z}") for z in zone_names_numeric]

# -------------------------------
# Extraction des indices de zones
# -------------------------------
lat_var = "latc" if "latc" in ds_sim else "lat"
lon_var = "lonc" if "lonc" in ds_sim else "lon"

# gérer coordonnées 1D ou 2D
if ds_sim[lat_var].ndim == 1:
    nlat = ds_sim[lat_var].shape[0]
    nlon = ds_sim[lon_var].shape[0]
else:
    nlat, nlon = ds_sim[lat_var].shape

# Créer la grille des zones
Zones_ind = np.empty((nlat, nlon), dtype=object)
k = 0
for i in range(nlat):
    for j in range(nlon):
        Zones_ind[i, j] = zone_map.get(gdf["classif_cl"].iloc[k], f"Zone_{gdf['classif_cl'].iloc[k]}")
        k += 1

# -------------------------------
# Paramètres temporels
# -------------------------------
start_date = "1990-01-01"
end_date   = "2014-12-31"

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def extract_monthly_precip(ds, var_name="RAINTOT"):
    ds_sel = ds.sel(time=slice(start_date, end_date))
    monthly = ds_sel[var_name].resample(time="MS").sum()
    return monthly

def extract_zone_data(arr, zones, zone_name):
    idx = np.where(zones == zone_name)
    return arr[:, idx[0], idx[1]].reshape(arr.shape[0], -1)

def apply_corrections(var_name, arr, ds_label):
    """Corrections pour les RCM/CPRCM si nécessaire"""
    if "RCM" in ds_label or "CPRCM" in ds_label:
        if var_name in ["T2MEAN", "T2MAX", "T2MIN"]:
            arr = arr - 273.15  # Kelvin → °C
        # RAINTOT déjà en mm/jour donc pas de correction
    return arr

# -------------------------------
# Boucle sur zones pour distribution
# -------------------------------
for zone in zone_names:
    plt.figure(figsize=(12, 6))
    for ds, label, color in zip(datasets, labels, colors):
        monthly_precip = extract_monthly_precip(ds)
        monthly_precip = apply_corrections("RAINTOT", monthly_precip.values, label)
        zone_arr = extract_zone_data(monthly_precip, Zones_ind, zone)
        zone_flat = zone_arr.flatten()
        zone_flat = zone_flat[~np.isnan(zone_flat)]

        # échantillonnage si trop grand
        max_samples = 200_000
        if zone_flat.size > max_samples:
            idx = np.random.choice(zone_flat.size, max_samples, replace=False)
            zone_flat = zone_flat[idx]

        # KDE
        kde = gaussian_kde(zone_flat)
        xs = np.linspace(np.min(zone_flat), np.max(zone_flat), 300)
        plt.plot(xs, kde(xs), color=color, label=label, linewidth=2)

    plt.xlabel("Précipitations mensuelles (mm)")
    plt.ylabel("Densité")
    plt.title(f"Distribution mensuelle - {zone} (1990-2014)", fontsize=14, fontweight="bold", color="black")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(OUT_FIG, f"DistributionMensuelle_RAIN_{zone}_1990_2014.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f" → sauvegardé : {out_file}")

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution mensuelle de T2MAX pour CPRCM, RCMs et SIM2 (1990-2014)
Région lyonnaise totale + par zone
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from scipy.stats import gaussian_kde

# -------------------------------
# Répertoires et fichiers
# -------------------------------
OUT_FIG = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/Figures/Distrib_mensuelle_variable_zone/T2MAX_total_and_per_zone/"
os.makedirs(OUT_FIG, exist_ok=True)

FN_SIM   = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/SIM2_1990_2014_rl_converted.nc"
FN_CPRCM = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/CPRCM_AROME_8km_rl_converted.nc"
FN_RCMs = [
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_ALADIN64E1_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_HCLIM43ALADIN_rl_converted.nc",
    "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/datasets_grilleSAFRAN_region_lyonnaise/RCM_RACMO32E_rl_converted.nc"
]

SHAPEFILE = "/work/crct/qcu21637/02122025_Hackathon_climat/DATA_VF/VF/shapefile_zones_region_lyonnaise/8km/grille_8km_centree_classif.shp"

# -------------------------------
# Lecture datasets
# -------------------------------
ds_sim   = xr.open_dataset(FN_SIM)
ds_cprcm = xr.open_dataset(FN_CPRCM)
ds_rcms  = [xr.open_dataset(f) for f in FN_RCMs]

datasets = [ds_cprcm] + ds_rcms + [ds_sim]
labels = ["CPRCM_AROME", "RCM_ALADIN64E1", "RCM_HCLIM43ALADIN", "RCM_RACMO32E", "SIM2"]
colors = ["green", "#a6cee3", "#1f78b4", "#08306b", "red"]

# -------------------------------
# Lecture shapefile zones
# -------------------------------
gdf = gpd.read_file(SHAPEFILE)

# Mapping zones numériques → noms explicites
zone_map = {1: "Artificielle", 2: "Agricole", 3: "Forêt"}
zone_names_numeric = gdf["classif_cl"].unique()
zone_names = [zone_map.get(z, f"Zone_{z}") for z in zone_names_numeric]

# -------------------------------
# Paramètres temporels
# -------------------------------
start_date = "1990-01-01"
end_date   = "2014-12-31"

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def extract_monthly_var(ds, var_name="T2MAX"):
    ds_sel = ds.sel(time=slice(start_date, end_date))
    monthly = ds_sel[var_name].resample(time="MS").mean()  # Température -> moyenne mensuelle
    return monthly

def flatten_zone(arr):
    flat = arr.values.reshape(arr.shape[0], -1)
    flat = flat.flatten()
    flat = flat[~np.isnan(flat)]
    return flat

def extract_zone_data(arr, zones, zone_name):
    idx = np.where(zones == zone_name)
    return arr[:, idx[0], idx[1]].reshape(arr.shape[0], -1)

def apply_corrections(var_name, arr, ds_label):
    """Conversion Kelvin -> °C pour RCM/CPRCM"""
    if "RCM" in ds_label or "CPRCM" in ds_label:
        if var_name in ["T2MAX", "T2MIN", "T2MEAN"]:
            arr = arr - 273.15
    return arr

# -------------------------------
# Extraction indices de zones
# -------------------------------
lat_var = "latc" if "latc" in ds_sim else "lat"
lon_var = "lonc" if "lonc" in ds_sim else "lon"

if ds_sim[lat_var].ndim == 1:
    nlat = ds_sim[lat_var].shape[0]
    nlon = ds_sim[lon_var].shape[0]
else:
    nlat, nlon = ds_sim[lat_var].shape

Zones_ind = np.empty((nlat, nlon), dtype=object)
k = 0
for i in range(nlat):
    for j in range(nlon):
        Zones_ind[i, j] = zone_map.get(gdf["classif_cl"].iloc[k], f"Zone_{gdf['classif_cl'].iloc[k]}")
        k += 1

# ================================
# 1 - Distribution région totale
# ================================
plt.figure(figsize=(12,6))
for ds, label, color in zip(datasets, labels, colors):
    monthly_var = extract_monthly_var(ds, var_name="T2MAX")
    monthly_var = apply_corrections("T2MAX", monthly_var.values, label)
    data_flat = flatten_zone(monthly_var)

    # échantillonnage
    max_samples = 500_000
    if data_flat.size > max_samples:
        idx = np.random.choice(data_flat.size, max_samples, replace=False)
        data_flat = data_flat[idx]

    kde = gaussian_kde(data_flat)
    xs = np.linspace(np.min(data_flat), np.max(data_flat), 300)
    plt.plot(xs, kde(xs), color=color, label=label, linewidth=2)

plt.xlabel("T2MAX mensuelle (°C)")
plt.ylabel("Densité")
plt.title("Distribution mensuelle T2MAX (1990-2014) - Région lyonnaise", fontsize=14, fontweight="bold", color="black")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()

out_file = os.path.join(OUT_FIG, "DistributionMensuelle_T2MAX_1990_2014_total.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f" → sauvegardé : {out_file}")

# ================================
# 2 - Distribution par zone
# ================================
for zone in zone_names:
    plt.figure(figsize=(12,6))
    for ds, label, color in zip(datasets, labels, colors):
        monthly_var = extract_monthly_var(ds, var_name="T2MAX")
        monthly_var = apply_corrections("T2MAX", monthly_var.values, label)
        zone_arr = extract_zone_data(monthly_var, Zones_ind, zone)
        zone_flat = zone_arr.flatten()
        zone_flat = zone_flat[~np.isnan(zone_flat)]

        # échantillonnage
        max_samples = 200_000
        if zone_flat.size > max_samples:
            idx = np.random.choice(zone_flat.size, max_samples, replace=False)
            zone_flat = zone_flat[idx]

        kde = gaussian_kde(zone_flat)
        xs = np.linspace(np.min(zone_flat), np.max(zone_flat), 300)
        plt.plot(xs, kde(xs), color=color, label=label, linewidth=2)

    plt.xlabel("T2MAX mensuelle (°C)")
    plt.ylabel("Densité")
    plt.title(f"Distribution mensuelle T2MAX (1990-2014) - Zone {zone}", fontsize=14, fontweight="bold", color="black")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(OUT_FIG, f"DistributionMensuelle_T2MAX_{zone}_1990_2014.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f" → sauvegardé : {out_file}")