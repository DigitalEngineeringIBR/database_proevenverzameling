

# Inputs:
# Locatie ids uit QGIS
# hoogte selectie voor ALLE monsters Zbot en Ztop
# Proef_types voor triaxiaal proeven: ['CU','UU', 'CD']
# Volume gewicht voor triaxiaal proeven Vgmin Vgmax
# Rek ea voor de LST-squares voor triaxiaal proeven: [2,5] or [2]
# Output file/dir: 'D:\documents\Proeven-Selectie.xlsx'
hoogte_selectie = [100, -10] # mNAP
proef_types = ['CU'] # Consolidated Undrained, Unconsolidated Undrained, Consolidated Drained ['CU','CD','UU']
volume_gewicht_selectie = [9, 22] # kN/m3
rek_selectie = [2] # Lijst met rek percentages waar statistieken van gemaakt worden
output_location = r'' # Output folder zoals: D:\Documents\geo_parameters\'
output_file = 'TRX_Example.xlsx' # Excel filename waar alle data in komt te staan
show_plot = True # Laat alle plotjes zien op je scherm
save_plot = False # Sla alle plotje automatisch op in output_location


# Begin van het script
import sys, os, shutil
os.chdir(r'D:\Documents\GitHub\proeven_verzameling')
sys.path.append(r'D:\Documents\GitHub\proeven_verzameling')
from qgis.utils import *
import qgis_backend as qb
import pandas as pd
import numpy as np

active_layer = iface.activeLayer()

loc_ids = qb.get_loc_ids(active_layer)
df_meetp = qb.get_meetpunten(loc_ids)
df_geod = qb.get_geo_dossiers( df_meetp.gds_id )
df_gm = qb.get_geotech_monsters(loc_ids)
df_gm_filt_on_z = qb.select_on_z_coord(df_gm, hoogte_selectie[0], hoogte_selectie[1])

df_dict = {'BIS_Meetpunten': df_meetp, 'BIS_GEO_Dossiers':df_geod, 'BIS_Geotechnische_Monsters':df_gm_filt_on_z}

df_sdp = qb.get_sdp(df_gm_filt_on_z.gtm_id)
if df_sdp is not None:
    df_sdp_result = qb.get_sdp_result(df_gm.gtm_id)
    df_dict.update({'BIS_SDP_Proeven':df_sdp, 'BIS_SDP_Resultaten':df_sdp_result})

df_trx = qb.get_trx(df_gm_filt_on_z.gtm_id, proef_type = proef_types)

if df_trx is not None:
    df_trx = qb.select_on_vg(df_trx, volume_gewicht_selectie[1], volume_gewicht_selectie[0])
    df_trx_results = qb.get_trx_result(df_trx.gtm_id)
    df_trx_dlp = qb.get_trx_dlp(df_trx.gtm_id)
    df_trx_dlp_result = qb.get_trx_dlp_result(df_trx.gtm_id)

    df_dict.update({'BIS_TRX_Proeven':df_trx, 'BIS_TRX_Results':df_trx_results, 'BIS_TRX_DLP':df_trx_dlp, 'BIS_TRX_DLP_Results': df_trx_dlp_result})
    
    # Statistieken
    if len(df_trx.index) > 1: # je kan geen statistieken doen op 1 proef
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
        for vg_max, vg_min in zip(Vgmax, Vgmin):
            gtm_ids = qb.select_on_vg(df_trx, Vg_max = vg_max, Vg_min = vg_min, soort = 'nat')['gtm_id']
            if len(gtm_ids) > 0:
                df_trx_results_temp = qb.get_trx_result(gtm_ids)
                stat_df = qb.get_average_per_ea(df_trx_results_temp, rek_selectie)
                stat_df.index.name = key = 'Vg: ' + str(round(vg_min,1)) + '-' + str(round(vg_max, 1)) + ' kN/m3'
                stat_df.columns.name = 'ea ='
                df_vg_stat_dict[key] = stat_df
        
        df_dict_lst_sqrs = {}
        for ea in  rek_selectie:
            ls_list = []
            for vg_max, vg_min in zip(Vgmax, Vgmin):
                gtm_ids = qb.select_on_vg(df_trx, Vg_max = vg_max, Vg_min = vg_min, soort = 'nat')['gtm_id']
                key = 'Vg: ' + str(round(vg_min,1)) + '-' + str(round(vg_max, 1)) + ' kN/m3'
                if len(gtm_ids) > 0:
                    fi, coh, E, E_per_n, eps, N = qb.get_least_squares(
                        qb.get_trx_dlp_result(gtm_ids), 
                        ea = ea, 
                        name = 'Least Squares Analysis, ea: ' + str(ea) + '\n' + key,
                        show_plot = show_plot, save_plot = save_plot
                        )
                    ls_list.append(pd.DataFrame(index = [key], data = [[vg_min, vg_max, fi, coh, E, E_per_n, eps, N]],\
                        columns=['min(Vg)', 'max(Vg)','fi','coh','Abs. Sq. Err.','Abs. Sq. Err./N','Mean Rel. Err. %','N']))
            if len(ls_list)>0:
                df_ls_stat = pd.concat(ls_list)
                df_ls_stat.index.name = 'ea: ' + str(ea)
                df_dict_lst_sqrs.update({ str(ea) + r'% rek least squares fit':df_ls_stat})

        df_bbn_stat_dict = {}
        for bbn_code in pd.unique(df_trx.bbn_kode):
            gtm_ids = df_trx[df_trx.bbn_kode == bbn_code].gtm_id
            if len(gtm_ids > 0):
                df_trx_results_temp = qb.get_trx_result(gtm_ids)
                stat_df = qb.get_average_per_ea(df_trx_results_temp, rek_selectie)
                stat_df.index.name = key = bbn_code
                stat_df.columns.name = 'ea ='
                df_bbn_stat_dict[key] = stat_df

#Make a copy of an excelsheet that already has the two important NEN 9997 tables in it
#Could be skipped but pd.ExcelWriter(..., mode = 'a') needs to be put on a write/overwrite mode = 'w' i think
shutil.copyfile('NEN 9997.xlsx', output_file)

with pd.ExcelWriter(output_file,engine='openpyxl',mode='a') as writer: #writer in append mode so that the NEN tables are kept
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
        row=0
        for key in df_dict_lst_sqrs:
            df_dict_lst_sqrs[key].to_excel(writer, sheet_name = 'Least Squares Statistieken', startrow = row)
            row = row + len(df_dict_lst_sqrs[key].index) + 2

os.startfile(output_file)

