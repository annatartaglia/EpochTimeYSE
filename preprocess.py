'''
    Preprocessing YSE data for input into MCMC and/or NN. 
    
    The YSE DR1 directory must be imported from https://zenodo.org/records/7317476
    The py file for reading in YSE data is sourced from https://github.com/patrickaleo/ysedr1_data_demos
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import shutil
from scipy.optimize import minimize_scalar
import glob
from read_yse_ztf_snana_dir import read_YSE_ZTF_snana_dir

# Read in YSE DR1 data
full_snid_list, full_meta_list, full_df_list = read_YSE_ZTF_snana_dir(dir_name='yse_dr1_zenodo')


def isnondetection(index, df):
    '''
        return whether or not the point at given index is a nondetection
    '''
    SNR  = df['FLUX'][df.index[index]]/df['FLUXERR'][df.index[index]]
    if SNR>5:
        return False
    else:
        return True
    
def get_t_trigger(df):
    '''
        gives the MJD trigger time (first detection) of LC
    '''
    for i in range(len(df)):
        if not isnondetection(i, df):
            return df['MJD'][df.index[i]]
        continue
    print("uh oh")

def shift_times(df):
    '''
        set new column df['TIMES'] to the returned array (shift times so t = 0 is trigger)
    '''
    t_trig = get_t_trigger(df)
    return df['MJD'] - t_trig

def normalize(df, uncert="e_cts", light="cts"):
    e_cts = df[uncert]
    cts = df[light]
    max_cts = cts.max()
    min_cts = cts.min()
    normalized_cts = (cts - min_cts) / (max_cts - min_cts)
    normalized_ects = e_cts / (max_cts - min_cts)
    return normalized_cts, normalized_ects


def sigma_clip(df, col,  times=5, const=3):
    for _ in range(0, times):
        mean = df[col].mean()
        threshold = const * df[col].std()
        df = df[np.abs(df[col] - mean) <= threshold]
    return df

'''
def convert_to_bin_timescale(value, interval):
    return pd.Timedelta(value, unit="D").round(interval) / pd.to_timedelta(interval)
'''


def remove_duplicate_indices(data_col):
    duplicates = data_col.index.duplicated()
    return data_col.loc[~duplicates]




def create_params_obj(sub_bg_model=False, remove_extinction=True, median_filter=True, window_size="1.5D",
                      to_bin=False, norm=False, bin_interval="0.5D", time_scale="trigger",
                      convert_to_mag=False):
    return {
        "norm": norm,
        "to_bin": to_bin,
        "bin_interval": bin_interval,
        "time_scale": time_scale,
        "convert_to_mag": convert_to_mag,
        "sub_bg_model": sub_bg_model,
        "median_filter": median_filter,
        "window_size": window_size,
        "remove_extinction": remove_extinction,
        # ztf_tess alignment params
        "scale_factor": 180,
        "optimize_scale": True,
        "manual_diff_corr": 0
    }

def preprocess(curve, curve_meta, light="FLUX", uncert="FLUXERR", sub_bg_model=False, remove_extinction=True,
               to_bin=True, norm=False, bin_interval="0.5D", time_scale="trigger", convert_to_mag=False,
               median_filter=True, window_size="1.5D", pb_wavelength=7865):


    # sigma clipping by e_cts
    curve = sigma_clip(curve, uncert)

    median_filter = False # need to change the window size for this to work i think
    # median window filtering
    if median_filter:
        outliers = []
        filter_df = curve[[light, "relative_time"]].copy()
        filter_df.index = pd.TimedeltaIndex(curve["relative_time"], unit="D")

        def median_window_filtering(window):
            median = window[light].median()
            abl_deviations = (window[light] - median).abs()
            mad = abl_deviations.median()
            window_outliers = window[abl_deviations >= 3 * mad]
            if not window_outliers.empty:
                outliers.extend(window_outliers["relative_time"].tolist())
            return 0

        filter_df.resample(window_size, origin="start").apply(median_window_filtering)
        curve = curve[~curve["relative_time"].isin(outliers)]

    # skipping convert to mag / offset step since we want flux

    if remove_extinction:
        ra = curve_meta["ra"]
        dec = curve_meta["dec"]
        flux_in = curve[light]
        bandpass_wavelengths = np.array([pb_wavelength,])

        # Get Milky Way E(B-V) Extinction
        coo = coord.SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
        b = coo.galactic.b.value
        mwebv = curve_meta['mwebv']

        # Remove extinction from light curves
        # (Using negative a_v so that extinction.apply works in reverse and removes the extinction)
        extinction_per_passband = extinction.fitzpatrick99(wave=bandpass_wavelengths, a_v=-3.1 * mwebv, r_v=3.1, unit='aa')
        if convert_to_mag:
            flux_out = flux_in + extinction_per_passband[0]
        else:
            flux_out = extinction.apply(extinction_per_passband[0], flux_in, inplace=False)

        curve[light] = flux_out
        curve_meta["gal_lat"] = b
    if norm:
        curve[light], curve[uncert] = normalize(curve, uncert=uncert, light=light)

    if not to_bin:
        curve.index = curve['relative_time']
        curve_meta['interval'] = bin_interval

    return curve



def preprocess_yse(ind, parameters):

    curve = full_df_list[ind]
    #curve_name = full_snid_list[ind]
    curve_meta = full_meta_list[ind]
    yse_data = {}
    passbands = {'g': "g", 'r': "r", 'i': "i", 'z': "z", 'ZTF_r': "X", 'ZTF_g': "Y"}
    wavelengths = {'g': 4849, 'r': 6201, 'i': 7535, 'z': 8674, 'ZTF_r': 6215, 'ZTF_g': 4767}

    if parameters['time_scale'] == "trigger":
        curve['relative_time'] = shift_times(curve)
        curve_meta["trigger"] = 0.0

    if parameters['time_scale'] == "first":
        # convert time to relative to 1st observation
        curve['relative_time'] = curve['MJD'] - curve['MJD'].iloc[0]
        curve_meta["trigger"] = get_t_trigger(curve) - curve['MJD'].iloc[0]

    for pb, fid in passbands.items():
        pb_df = curve[curve['PASSBAND'] == fid]

        pb_df['FLUX']
        light_str = f'{pb}_flux' if not parameters['convert_to_mag'] else f'{pb}_mag'
        uncert_str = f'{pb}_uncert'
        if not pb_df.empty:
            pb_wavelength = wavelengths[pb]
            to_process_df = pd.DataFrame({"relative_time": pb_df['relative_time'], light_str: pb_df['FLUX'], uncert_str: pb_df["FLUXERR"]})
            processed_pb = preprocess(to_process_df, curve_meta, light=light_str, uncert=uncert_str,
                                        to_bin=parameters['to_bin'], bin_interval=parameters['bin_interval'],
                                        time_scale=parameters['time_scale'], norm=parameters['norm'],
                                        median_filter=False, remove_extinction=parameters['remove_extinction'],
                                        convert_to_mag=parameters['convert_to_mag'], pb_wavelength=pb_wavelength)
            if not parameters['to_bin']:
                yse_data[light_str] = remove_duplicate_indices(processed_pb[light_str])
                yse_data[uncert_str] = remove_duplicate_indices(processed_pb[uncert_str])
            else:
                yse_data[light_str] = processed_pb[light_str]
                yse_data[uncert_str] = processed_pb[uncert_str]
        else:
            yse_data[light_str] = pd.Series(dtype='float64')
            yse_data[uncert_str] = pd.Series(dtype='float64')

        yse_df = pd.DataFrame(yse_data)
        yse_df.index = yse_df.index.rename("relative_time")
    return yse_df, curve_meta


def save_processed_curves(params, to_process_lst=[],  enable_final_processing=True, meta_targets=[], fill_missing=True,
                          mask_val=0.0, curve_range=None, reset=False, directory="./bad_processed_curves/"):
    mwebv_outliers = []
    '''if reset:
        for f in os.listdir(directory):
            os.remove(os.path.join(directory, f))'''

    for SN in to_process_lst:
        ind = full_snid_list.index(SN)
        curve, meta = preprocess_yse(ind, params)

        if curve is not None and not curve.empty:
            passbands = ['g','r','i','z','ZTF_r','ZTF_g']
            light_unit = "mag" if params["convert_to_mag"] else "flux"
            targets = [f"{pb}_{light_unit}" for pb in passbands] + [f"{pb}_uncert" for pb in passbands]
            df = curve[targets]

            # limit data points to those in range
            if curve_range is not None:
                df = df.loc[curve_range[0]:curve_range[1], :]

            # filter out curves with mwebv above 0.5
            if meta['mwebv'] > 0.6:
                mwebv_outliers.append({"mwebv": meta['mwebv'], "class": meta['classification'], 'gal_lat': meta['gal_lat'], 'IAU_name': SN})
            else:
                if enable_final_processing:
                    # add additional meta information
                    for info in meta_targets:
                        valid_timesteps = (df != mask_val).any(axis=1)
                        df[info] = pd.Series(meta[info], index=df[valid_timesteps].index)
                        df.loc[~valid_timesteps, info] = mask_val
                    # fill in missing timesteps
                    if fill_missing:
                        df = df.fillna(mask_val)
                        interval_val = float(params["bin_interval"].split("D")[0])
                        if curve_range is None:
                            raise Exception("Needs range to be set")
                        for t in np.arange(curve_range[0], curve_range[1] + interval_val, interval_val):
                            if t not in df.index:
                                df.loc[t] = np.full(shape=len(df.columns), fill_value=mask_val)
                                df = df.sort_index()

                    #df["max_light"] = max_light
                    #df["max_uncert"] = uncert
                filename = f"lc_{SN}_processed.csv"
                df.to_csv(directory + filename)


if __name__ == "__main__":
    config = {
        "norm": False,
        "to_bin": False,
        "bin_interval": "0.5D",
        "time_scale": "trigger",
        'convert_to_mag': False,
        'sub_bg_model': False,
        'remove_extinction': True,
        "median_filter": True,
        "window_size": "2.5D",
        # ztf_tess alignment params
        "scale_factor": 180,
        "optimize_scale": True,
        "manual_diff_corr": 0
    }
    zip_name = "smallset"
    curve_lim = (-300, 200)
    fill = False
    maskval = 0.0
    meta_targets = ["mwebv"]
    ztf_tess = True
    label_formatting = True

    # to_process = to_process_list 

    to_process = ['2020awu', '2020aazr', '2020abjz', '2020acyt', '2021adnv', '2021avi', 
                  '2021ctn', '2021guc', '2021iww', '2021lel',  '2021pe', '2021pvw', '2021qxr', 
                  '2021shf', '2021tjk', '2021wlt', '2021zzv', '2020jur', '2021ctc']

    # # # process all
    # to_process = None
    labels = None
    save_processed_curves(config, to_process_lst=to_process, reset=True,
                          meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                          enable_final_processing=label_formatting)
    
    #print(f"Saved to:{zip_name}")