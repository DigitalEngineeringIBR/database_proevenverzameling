'''
Script: qgis_frontend.py
Repository: https://github.com/KRS-dev/proeven_verzameling
Author: Kevin Schuurman
E-mail: kevinschuurman98@gmail.com
Summary: Frontend use in QGIS to query data from the proeven verzameling database.
'''
'''
Inputs:
hoogte selectie voor ALLE monsters Zbot en Ztop
Proef types voor triaxiaal proeven: ['CU','UU', 'CD']
Volume gewicht voor triaxiaal proeven Vgmin Vgmax
Rek ea voor selectie bij LST-squares voor triaxiaal proeven
Output file/dir: 'D:\documents\Proeven-Selectie.xlsx'
'''
#if __name__ == "__main__":
hoogte_selectie = [100, -100] # mNAP
proef_types = ['CU'] # ['CU','CD','UU']
volume_gewicht_selectie = [9, 22] # kN/m3
rek_selectie = [5] # %
output_location = r'D:\Documents\Projects\EXCEL BIS STAT' # Example: D:\Documents\geo_parameters\'
output_file = 'TRX_Example.xlsx' # Example: TRX_example.xlsx
show_plot = True # True/False
save_plot = False # True/False


import sys, os, shutil
# Change the directory so the script can open all files in this directory easily
os.chdir(r'D:\Documents\GitHub\proeven_verzameling')
# Adding the same path to the python system so that qgis_backend can be imported
sys.path.append(r'D:\Documents\GitHub\proeven_verzameling')
from qgis.utils import iface
import qgis_backend as qb
import pandas as pd
import numpy as np
import openpyxl

# Check if the directory still has to be made.
if os.path.isdir(output_location) == False:
    os.mkdir(output_location)

#Get the active layer from QGIS
active_layer = iface.activeLayer()
# Extract the loc ids from the selected points in the active layer
loc_ids = qb.get_loc_ids(active_layer)
# Get all meetpunten related to these loc_ids
df_meetp = qb.get_meetpunten(loc_ids)
df_geod = qb.get_geo_dossiers( df_meetp.gds_id )
df_gm = qb.get_geotech_monsters(loc_ids)
df_gm_filt_on_z = qb.select_on_z_coord(df_gm, hoogte_selectie[0], hoogte_selectie[1])
# Add the df_meetp, df_geod and df_gm_filt_on_z to a dataframe dictionary
df_dict = {'BIS_Meetpunten': df_meetp, 'BIS_GEO_Dossiers':df_geod, 'BIS_Geotechnische_Monsters':df_gm_filt_on_z}

df_sdp = qb.get_sdp( df_gm_filt_on_z.gtm_id )
if df_sdp is not None:
    df_sdp_result = qb.get_sdp_result(df_gm.gtm_id)
    df_dict.update({'BIS_SDP_Proeven':df_sdp, 'BIS_SDP_Resultaten':df_sdp_result})

df_trx = qb.get_trx(df_gm_filt_on_z.gtm_id, proef_type = proef_types)
df_trx = qb.select_on_vg(df_trx, volume_gewicht_selectie[1], volume_gewicht_selectie[0])
if df_trx is not None:
    # Get all TRX results, TRX deelproeven and TRX deelproef results
    df_trx_results = qb.get_trx_result(df_trx.gtm_id)
    df_trx_dlp = qb.get_trx_dlp(df_trx.gtm_id)
    df_trx_dlp_result = qb.get_trx_dlp_result(df_trx.gtm_id)
    df_dict.update({'BIS_TRX_Proeven':df_trx, 'BIS_TRX_Results':df_trx_results, 'BIS_TRX_DLP':df_trx_dlp, 'BIS_TRX_DLP_Results': df_trx_dlp_result})
    
    # Doing statistics on the select TRX proeven
    if len(df_trx.index) > 1:
        ## Create a linear space between de maximal volumetric weight and the minimal volumetric weight
        minvg, maxvg = min(df_trx.volumegewicht_nat), max(df_trx.volumegewicht_nat)
        N = round(len(df_trx.index)/5) + 1
        cutoff = 1 # The interval cant be lower than 1 kn/m3
        if (maxvg-minvg)/N > cutoff:
            Vg_linspace = np.linspace(minvg, maxvg, N)
        else:
            Vg_linspace = np.linspace(minvg, maxvg, round((maxvg-minvg)/cutoff))
        
        Vgmax = Vg_linspace[1:]
        Vgmin = Vg_linspace[0:-1]

        df_vg_stat_dict = {}
        df_lst_sqrs_dict = {}
        for ea in rek_selectie:
            ls_list = []
            avg_list = []
            for vg_max, vg_min in zip(Vgmax, Vgmin):
                # Make a selection for this volumetric weight interval
                gtm_ids = qb.select_on_vg(df_trx, Vg_max = vg_max, Vg_min = vg_min, soort = 'nat')['gtm_id']
                if len(gtm_ids) > 0:
                    # Create a tag for this particular volumetric weight interval
                    key = 'Vg: ' + str(round(vg_min,1)) + '-' + str(round(vg_max, 1)) + ' kN/m3'
                    # Get the related TRX results...
                    # 
                    ## Potentially the next line could be done without querying the database again 
                    ## for the data that is already availabe in the variable df_trx_results
                    ## but I have not found the right type of filter methods in Pandas which
                    ## can replicate the SQL filters
                    # 
                    df_trx_results_temp = qb.get_trx_result(gtm_ids)
                    # Calculate the averages and standard deviation of fi and coh for different strain types and add them to a dataframe list
                    mean_fi,std_fi,mean_coh,std_coh,N = qb.get_average_per_ea(df_trx_results_temp, ea)
                    df_avg_temp = pd.DataFrame(index = [key], data = [[vg_min, vg_max, mean_fi, mean_coh, std_fi, std_coh, N]],\
                        columns=['min(Vg)','max(Vg)','mean(fi)','mean(coh)','std(fi)','std(coh)','N'])                    
                    avg_list.append(df_avg_temp)
                    # Calculate the least squares estimate of the S en T and add them to a dataframe list
                    fi, coh, E, E_per_n, eps, N = qb.get_least_squares(
                        qb.get_trx_dlp_result(gtm_ids), 
                        ea = ea, 
                        plot_name = 'Least Squares Analysis, ea: ' + str(ea) + '\n' + key,
                        show_plot = show_plot, 
                        save_plot = save_plot
                        )
                    df_lst_temp = pd.DataFrame(index = [key], data = [[vg_min, vg_max, fi, coh, E, E_per_n, eps, N]],\
                        columns=['min(Vg)', 'max(Vg)','fi','coh','Abs. Sq. Err.','Abs. Sq. Err./N','Mean Rel. Err. %','N'])
                    ls_list.append(df_lst_temp)
            if len(ls_list)>0:
                df_ls_stat = pd.concat(ls_list)
                df_ls_stat.index.name = 'ea: ' + str(ea) +'%'
                df_lst_sqrs_dict.update({ str(ea) + r'% rek least squares fit':df_ls_stat})
            if len(avg_list)>0:
                df_avg_stat = pd.concat(avg_list)
                df_avg_stat.index.name = 'ea: ' + str(ea) +'%'
                df_vg_stat_dict.update({ str(ea) + r'% rek gemiddelde fit':df_avg_stat})

        df_bbn_stat_dict = {}
        for ea in rek_selectie:
            bbn_list = []
            for bbn_code in pd.unique(df_trx.bbn_kode):
                gtm_ids = df_trx[df_trx.bbn_kode == bbn_code].gtm_id
                if len(gtm_ids > 0):
                    df_trx_results_temp = qb.get_trx_result(gtm_ids)
                    mean_fi,std_fi,mean_coh,std_coh,N = qb.get_average_per_ea(df_trx_results_temp, ea)
                    bbn_list.append(pd.DataFrame(index = [bbn_code], data = [[mean_fi, mean_coh, std_fi, std_coh, N]],\
                        columns=['mean(fi)','mean(coh)','std(fi)','std(coh)','N']))
            if len(bbn_list)>0:        
                df_bbn_stat = pd.concat(bbn_list)
                df_bbn_stat.index.name = 'ea: ' + str(ea) +'%'
                df_bbn_stat_dict.update({ str(ea) + r'% rek per BBN code':df_bbn_stat})

# Check if the .xlsx file exists
output_file_dir = os.path.join(output_location, output_file)
if os.path.exists(output_file_dir) == False:
    book = openpyxl.Workbook()
    book.save(output_file_dir)
else:
    name, ext = output_file.split('.')
    i = 1
    while os.path.exists(os.path.join(output_location, name + '{}.'.format(i) + ext)):
        i += 1
    output_file_dir = os.path.join(output_location, name + '{}.'.format(i) + ext)
    book = openpyxl.Workbook()
    book.save(output_file_dir)

# At the end of the 'with' function it closes the excelwriter automatically, even if there was an error
with pd.ExcelWriter(output_file_dir,engine='openpyxl',mode='w') as writer: #writer in append mode so that the NEN tables are kept
    for key in df_dict:
        # Writing every dataframe in the dictionary to a different sheet
        df_dict[key].to_excel(writer, sheet_name = key)
        
    if df_trx is not None:
        # Write the multiple dataframes of the same statistical analysis for TRX to the same excel sheet by counting rows
        row = 0
        for key in df_vg_stat_dict:
            df_vg_stat_dict[key].to_excel(writer, sheet_name = 'Simpele Vg stat.', startrow = row)
            row = row + len(df_vg_stat_dict[key].index) + 2
        # Repeat...
        row=0
        for key in df_lst_sqrs_dict:
            df_lst_sqrs_dict[key].to_excel(writer, sheet_name = 'Least Squares Vg Stat.', startrow = row)
            row = row + len(df_lst_sqrs_dict[key].index) + 2
        row=0
        for key in df_bbn_stat_dict:
            df_bbn_stat_dict[key].to_excel(writer, sheet_name = 'bbn_kode Stat.', startrow = row)
            row = row + len(df_bbn_stat_dict[key].index) + 2
        

os.startfile(output_file_dir)

