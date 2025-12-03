#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:23:05 2025

@author: Quentin Cournault (groupe de travail OPTIMALES)
"""

"""
Script générique pour découper un fichier NetCDF météo France
selon une zone géographique définie par un shapefile.

Fonctionnalités :
- Supporte les fichiers 3D (time, y, x) ou 2D (lat/lon) type SIM.
- Calcul du masque basé sur le shapefile.
- Réduction automatique du domaine aux cellules valides.
- Sauvegarde dans un nouveau fichier NetCDF.
"""

import xarray as xr
import geopandas as gpd
import rioxarray
import shapely
import numpy as np
import os
import gc

# ======================================================================================
# --- CHEMINS DES FICHIERS 
# --- (1 fichier par variable -> fichiers déjà mergés avec nco - ncks) ---
# ======================================================================================
# Fichier NetCDF d'entrée 
nc_in = "[CHEMIN_VERS_FICHIER_NETCDF]"

# Fichier NetCDF de sortie
nc_out = "[CHEMIN_VERS_FICHIER_NETCDF_SORTIE]"

# Shapefile représentant la zone d'intérêt (dans notre cas : région lyonnaise)
shapefile_path = "[CHEMIN_VERS_SHAPEFILE]"

# ==============================
# --- SUPPRESSION DU FICHIER SORTIE EXISTANT ---
# ==============================
if os.path.exists(nc_out):
    os.remove(nc_out)

# ==============================
# --- CHARGEMENT DU DATASET ---
# ==============================
ds = xr.open_dataset(nc_in)

# ==============================
# --- CHARGEMENT DU SHAPEFILE ---
# ==============================
gdf = gpd.read_file(shapefile_path)

# Fusionner toutes les géométries en une seule (utile si shapefile contient plusieurs polygones)
zone = gdf.unary_union

# ==============================
# --- DÉTECTION DU TYPE DE GRILLE ---
# ==============================
# Deux cas principaux :
# 1) Grille régulière avec coordonnées lat/lon => cas de données observées SIM2 Météo France déjà post-processées (utilisées en base de comparaison sur l'histo)
# 2) Grille x/y avec coordonnées rioxarray (cas des données GCM/RCM et CPRCM mises à dispo pour le hackathon)

if "latc" in ds.coords and "lonc" in ds.coords:
    # Cas SIM / lat/lon centrées
    lat = ds["latc"].values
    lon = ds["lonc"].values
    
    # Création du masque (True si la cellule est dans la zone)
    mask_np = np.zeros(lat.shape, dtype=bool)
    print("Calcul du masque pour la grille lat/lon…")
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            mask_np[i, j] = zone.contains(shapely.geometry.Point(lon[i, j], lat[i, j]))
    
    mask = xr.DataArray(mask_np, dims=("lat", "lon"))
    
    # Réduction du domaine aux indices valides
    i_valid = mask_np.any(axis=1)
    j_valid = mask_np.any(axis=0)
    imin, imax = np.where(i_valid)[0][[0, -1]]
    jmin, jmax = np.where(j_valid)[0][[0, -1]]
    
    # Variables à traiter
    vars_to_mask = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
    
    # Écriture des coordonnées découpées
    coords_ds = xr.Dataset({
        "time": ds["time"],
        "latc": ds["latc"].isel(lat=slice(imin, imax+1), lon=slice(jmin, jmax+1)),
        "lonc": ds["lonc"].isel(lat=slice(imin, imax+1), lon=slice(jmin, jmax+1)),
    })
    coords_ds.to_netcdf(nc_out, mode="w")
    del coords_ds
    gc.collect()
    
    # Découpage et écriture des variables
    for var in vars_to_mask:
        print(f"Découpage et écriture de la variable {var}…")
        da = ds[var].isel(lat=slice(imin, imax+1), lon=slice(jmin, jmax+1))
        da = da.where(mask.isel(lat=slice(imin, imax+1), lon=slice(jmin, jmax+1)))
        da.to_netcdf(nc_out, mode="a")
        del da
        gc.collect()

else:
    # Cas rioxarray / grille x/y (CPRCM, RCM)
    print("Grille type x/y détectée, utilisation de rioxarray pour le clipping…")
    
    shapefile_proj = gdf.to_crs("EPSG:27572")
    
    # Filtrer les variables 3D (time, y, x)
    vars_3D = [v for v in ds.data_vars if set(ds[v].dims) == {"time", "y", "x"}]
    
    ds_masked = xr.Dataset(attrs=ds.attrs)
    
    for var in vars_3D:
        print(f"Découpage de la variable {var}…")
        da_rio = ds[var].rio.write_crs("EPSG:27572", inplace=False)
        da_masked = da_rio.rio.clip(shapefile_proj.geometry, shapefile_proj.crs, drop=True, invert=False)
        
        # Supprimer les attributs conflictuels
        attrs_clean = {k: v for k, v in da_masked.attrs.items() if k != "grid_mapping"}
        da_masked.attrs = attrs_clean
        
        # Réduire le domaine aux cellules valides
        mask_valid = ~da_masked.isnull().all(dim="time")
        y_sel = mask_valid.any(dim="x")
        x_sel = mask_valid.any(dim="y")
        da_masked = da_masked.sel(y=y_sel, x=x_sel)
        
        ds_masked[var] = da_masked
    
    # Copier les coordonnées réduites
    for coord in ["time", "y", "x"]:
        if coord not in ds_masked and coord in ds:
            ds_masked[coord] = ds[coord]
    
    # Sauvegarder
    ds_masked.to_netcdf(nc_out)
    
print(f"✔ Fichier découpé créé : {nc_out}")
