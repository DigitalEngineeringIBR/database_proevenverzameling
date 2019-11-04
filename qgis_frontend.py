import sys, os, shutil
os.chdir(r'D:\Documents\Python_Scripts\Parameter Database')
sys.path.append(r'D:\Documents\Python_Scripts\Parameter Database')
from qgis.utils import *
import qgis_backend as qb
import pandas as pd
import numpy as np

active_layer = iface.activeLayer()

loc_ids = qb.get_loc_ids(active_layer)

df_meetp = qb.get_meetpunten(loc_ids)
df_geod = qb.get_geo_dossiers( df_meetp.gds_id )
df_gm = qb.get_geotech_monsters(loc_ids)
df_gm_filt_on_z = qb.select_on_z_coord(df_gm,0,-5)

df_dict = {'BIS_Meetpunten': df_meetp, 'BIS_GEO_Dossiers':df_geod, 'BIS_Geotechnische_Monsters':df_gm}

df_sdp = qb.get_sdp(df_gm.gtm_id)
if df_sdp is not None:
    df_sdp_result = qb.get_sdp_result(df_gm.gtm_id)
    df_dict.update({'BIS_SDP_Proeven':df_sdp, 'BIS_SDP_Resultaten':df_sdp_result})

df_trx = qb.get_trx(df_gm.gtm_id, proef_type = ['CU','UU','CD'])
if df_trx is not None:
    
    df_trx_results = qb.get_trx_result(df_trx.gtm_id)
    df_trx_dlp = qb.get_trx_dlp(df_trx.gtm_id)
    df_trx_dlp_result = qb.get_trx_dlp_result(df_trx.gtm_id)

    df_dict.update({'BIS_TRX_Proeven':df_trx, 'BIS_TRX_Results':df_trx_results, 'BIS_TRX_DLP':df_trx_dlp, 'BIS_TRX_DLP_Results': df_trx_dlp_result})
    
    minvg, maxvg = min(df_trx.volumegewicht_nat), max(df_trx.volumegewicht_nat)
    N = round(len(df_trx.index)/5) + 1
    cutoff = 1
    if (maxvg-minvg)/N > cutoff:
        Vg_linspace = np.linspace(minvg, maxvg, N)
    else:
        Vg_linspace = np.linspace(minvg, maxvg, round((maxvg-minvg)/cutoff))
    Vgmax = Vg_linspace[1:]
    Vgmin = Vg_linspace[0:-1]

    df_vg_stat_dict = {}
    ls_list = []
    for vg_max, vg_min in zip(Vgmax, Vgmin):
        gtm_ids = qb.select_on_vg(df_trx, Vg_max = vg_max, Vg_min = vg_min, soort = 'nat')['gtm_id']
        if len(gtm_ids) > 0:
            df_trx_results_temp = qb.get_trx_result(gtm_ids)
            stat_df = qb.get_average_per_ea(df_trx_results_temp)
            stat_df.index.name = key = 'Vg: ' + str(round(vg_min,1)) + '-' + str(round(vg_max, 1)) + ' kN/m3'
            stat_df.columns.name = 'ea ='
            df_vg_stat_dict[key] = stat_df
            fi, coh, E, E_per_n, eps, N = qb.get_least_squares(
                qb.get_trx_dlp_result(gtm_ids), 
                ea = 2, 
                name = 'Least Squares Analysis, ea: 5\n' + key,
                make_plot = True
                )
            ls_list.append(pd.DataFrame(index = [key], data = [[fi,coh,E,E_per_n,eps,N]],\
                columns=['fi','coh','Abs. Sq. Error','Abs. Sq. Error/N','Rel Error %','N']))
    df_ls_stat = pd.concat(ls_list)
    df_dict.update({'Vg Least Squares Stat':df_ls_stat})

    df_bbn_stat_dict = {}
    for bbn_code in pd.unique(df_trx.bbn_kode):
        gtm_ids = df_trx[df_trx.bbn_kode == bbn_code].gtm_id
        if len(gtm_ids > 0):
            df_trx_results_temp = qb.get_trx_result(gtm_ids)
            stat_df = qb.get_average_per_ea(df_trx_results_temp)
            stat_df.index.name = key = bbn_code
            stat_df.columns.name = 'ea ='
            df_bbn_stat_dict[key] = stat_df
    
os.chdir(r'D:\Documents\Python_Scripts\Parameter Database')

output_file = 'TRX_Example.xlsx'

shutil.copyfile('NEN 9997.xlsx', output_file)

with pd.ExcelWriter(output_file,mode='a') as writer: #writer in append mode so that the NEN tables are kept
    for key in df_dict:
        df_dict[key].to_excel(writer, sheet_name = key)
        
    if df_trx is not None:
        row = 0
        for key in df_vg_stat_dict:
            df_vg_stat_dict[key].to_excel(writer, sheet_name = 'Vg statistieken', startrow = row)
            row = row + len(df_vg_stat_dict[key].index) + 2
        row=0
        for key in df_bbn_stat_dict:
            df_bbn_stat_dict[key].to_excel(writer, sheet_name = 'bbn Statistieken', startrow = row)
            row = row + len(df_bbn_stat_dict[key].index) + 2

os.startfile(output_file)

