import pandas as pd
import numpy as np
import psycopg2 as psy
import matplotlib.pyplot as plt

def get_loc_ids(QgisLayer):
    loc_ids = []
    features = QgisLayer.selectedFeatures()

    if len(features) > 0:
        print(str(len(features)) + ' were selected')
        for f in features:
            try:
                loc_ids.append(f.attribute('loc_id'))
            except KeyError:
                raise KeyError(
                    'This layer does not contain an attribute called loc_id')
            except:
                raise IOError(
                    'Something went wrong in selecting the column \'loc_id\'')
        return loc_ids
    else:
        raise KeyError('No features were selected in the layer')

def get_meetpunten( loc_ids ):
    if isinstance( loc_ids, ( list, tuple, pd.Series) ):
        if len( loc_ids ) > 0:
            if( all( isinstance(x, int) for x in loc_ids )):

                with psy.connect(
                    host = "localhost",
                    database = "bis",
                    user = "postgres",
                    password = "admin"
                ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple(loc_ids)
                    values_str = '(' + ','.join( str(i) for i in values ).strip(',') + ')'
                    query = 'SELECT * FROM graf_loc_aanduidingen '\
                        + 'INNER JOIN meetpunten ON meetpunten.mpt_id = graf_loc_aanduidingen.loc_id '\
                        + 'WHERE graf_loc_aanduidingen.loc_id IN ' + values_str
                    print('SQL QUERY:\n' + query)
                    cur.execute(query)
                    fetched = cur.fetchall()
                    if ( 0 < len( fetched )):
                        meetp_df = pd.DataFrame(fetched)
                        colnames = [ desc[0] for desc in cur.description ]
                        meetp_df.columns = colnames
                        meetp_df.gds_id = meetp_df.gds_id.fillna(0)
                        meetp_df.gds_id = pd.to_numeric(meetp_df.gds_id, downcast='integer')
                        return meetp_df
                    else:
                        raise ValueError('These selected geometry points do not contain valid loc_ids: ' + values_str)
            else:
                raise TypeError('not all inputs are integers')
        else:
            raise ValueError('No bor_ids were supplied.') 
    else:
        raise TypeError('Input is not a list or tuple')    

def get_geo_dossiers( gds_ids ):
    if isinstance( gds_ids, ( list, tuple, pd.Series) ):
        if len( gds_ids ) > 0:
            if( all( isinstance(x, int) for x in gds_ids )):

                with psy.connect(
                    host = "localhost",
                    database = "bis",
                    user = "postgres",
                    password = "admin"
                ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple( gds_ids )
                    values_str = '(' + ','.join( str(i) for i in values ).strip(',') + ')'
                    query = 'SELECT * FROM geo_dossiers WHERE gds_id IN ' + values_str
                    print('SQL QUERY:\n' + query)
                    cur.execute(query)
                    fetched = cur.fetchall()
                    if ( 0 < len( fetched )):
                        geod_df = pd.DataFrame(fetched)
                        colnames = [ desc[0] for desc in cur.description ]
                        geod_df.columns = colnames
                        return geod_df
                    else:
                        raise ValueError('The selected gds_ids: ' + values_str + \
                        ' do not contain any geodossiers.')
            else:
                raise TypeError('not all inputs are integers')
        else:
            raise ValueError('No bor_ids were supplied.') 
    else:
        raise TypeError('Input is not a list or tuple')    

def get_geotech_monsters( bor_ids ):
    if isinstance( bor_ids, ( list, tuple, pd.Series) ):
        if len( bor_ids ) > 0:
            if( all( isinstance( x, (int)) for x in bor_ids )):
                
                with psy.connect(
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin"
                ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple(bor_ids)
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT * FROM geotech_monsters WHERE bor_id IN ' + values_str
                    cur.execute( query )
                    print('SQL QUERY:\n' + query )
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        g_mon_df = pd.DataFrame( fetched )
                        colnames = [ desc[0] for desc in cur.description ]
                        g_mon_df.columns = colnames
                        g_mon_df['z_coordinaat_laag'] = pd.to_numeric( g_mon_df['z_coordinaat_laag'] )
                        return g_mon_df
                    else:
                        raise ValueError('These selected boring(en): ' + values_str + \
                        ' do not contain any triaxiaal proeven.')
            else:
                raise TypeError('not all inputs are integers')
        else:
            raise ValueError('No bor_ids were supplied.') 
    else:
        raise TypeError('Input is not a list or tuple')
        
def select_on_z_coord( g_mon_df, zmax, zmin ):
    if isinstance( g_mon_df, pd.DataFrame ):
        new_g_mon_df = g_mon_df[( zmax > g_mon_df.z_coordinaat_laag ) & ( g_mon_df.z_coordinaat_laag > zmin )]
        return new_g_mon_df
    else:
         raise TypeError('No pandas dataframe was supplied')

def get_trx( gtm_ids, proef_type = ('CD') ):
    if isinstance( gtm_ids, ( list, tuple, pd.Series ) ):
        if all( any( x == i for i in ('CU','CD','UU') ) for x in proef_type ):
            if len( gtm_ids ) > 0:
                if all( isinstance( x, ( int )) for x in gtm_ids ):
                    
                    with psy.connect( 
                        host = "localhost", 
                        database = "bis", 
                        user = "postgres", 
                        password = "admin" 
                        ) as dbcon:

                        cur = dbcon.cursor()
                        values = tuple(gtm_ids)
                        values_str = '(' + ','.join( str( i ) for i in values ).strip(',') + ')'
                        proef_type = tuple(proef_type)
                        proef_type_str = '(\'' + '\',\''.join( str( i ) for i in proef_type ).strip(',\'') + '\')'
                        query = 'SELECT * FROM trx WHERE proef_type IN ' + proef_type_str + \
                            ' AND gtm_id IN ' + values_str
                        cur.execute( query )
                        print('SQL QUERY: \n' + query )
                        fetched = cur.fetchall()
                        if( len( fetched ) > 0 ):
                            trx_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in cur.description]
                            trx_df.columns = colnames
                            trx_df[['volumegewicht_droog','volumegewicht_nat','watergehalte','terreinspanning','bezwijksnelheid']] = \
                            trx_df[['volumegewicht_droog','volumegewicht_nat','watergehalte','terreinspanning','bezwijksnelheid']].apply(pd.to_numeric)
                            trx_df.volumegewicht_nat = trx_df.volumegewicht_nat.astype(float)
                            return trx_df
                        else:
                            print('These selected boring(en): ' + values_str + \
                                ' do not contain any triaxiaal proeven with proef_type: ' + proef_type_str)
                else:
                    raise TypeError('not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Only CU, CD and UU or a combination are allowed as proef_type')
    else:
        raise TypeError('Input is not a list or tuple')
      
def select_on_vg( trx_df, Vg_max = 20, Vg_min = 17, soort ='nat' ):
    #Volume gewicht y in kN/m3
    if isinstance(trx_df, pd.DataFrame):
        if soort == 'nat':
            new_trx_df = trx_df[(Vg_max >= trx_df.volumegewicht_nat) & (trx_df.volumegewicht_nat >= Vg_min)]
        elif soort == 'droog':
            new_trx_df = trx_df[(Vg_max >= trx_df.volumegewicht_droog) & (trx_df.volumegewicht_droog >= Vg_min)]
        else:
            raise TypeError('\'' + soort + '\' is not allowed as argument for soort,\
                 only \'nat\' and \'droog\' are allowed.')
        return new_trx_df
    else:
        raise TypeError('No pandas dataframe was supplied')

def get_trx_result( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        
        if len( gtm_ids ) > 0:
            
            if all(isinstance(x, (int)) for x in gtm_ids):
            
                with psy.connect( 
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin" 
                    ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple( gtm_ids )
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT * FROM trx_result WHERE gtm_id IN ' + values_str  
                    cur.execute( query )
                    print('SQL QUERY: \n' + query )  
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        trx_result_df = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in cur.description]
                        trx_result_df.columns = colnames
                        trx_result_df[['ea','coh','fi']] = trx_result_df[['ea','coh','fi']].apply(pd.to_numeric)
                        return trx_result_df
                    else:
                        print('These selected boring(en): ' + values_str + \
                            ' do not contain any trx_results.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

def get_trx_dlp( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        
        if len( gtm_ids ) > 0:
            
            if all(isinstance(x, (int)) for x in gtm_ids):
            
                with psy.connect( 
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin" 
                    ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple( gtm_ids )
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT * FROM trx_dlp WHERE gtm_id IN ' + values_str  
                    cur.execute( query )
                    print('SQL QUERY: \n' + query )  
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        trx_dlp = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in cur.description]
                        trx_dlp.columns = colnames
                        trx_dlp.loc[:,'eps50':] = trx_dlp.loc[:,'eps50':].apply(pd.to_numeric)
                        return trx_dlp
                    else:
                        print('These selected geomonsters: ' + values_str + \
                            ' do not contain any trx_dlp.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

def get_trx_dlp_result( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        
        if len( gtm_ids ) > 0:
            
            if all(isinstance(x, (int)) for x in gtm_ids):
            
                with psy.connect( 
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin" 
                    ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple( gtm_ids )
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT * FROM trx_dlp_result WHERE gtm_id IN ' + values_str  
                    cur.execute( query )
                    print('SQL QUERY: \n' + query )  
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        trx_dlp_result = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in cur.description]
                        trx_dlp_result.columns = colnames
                        trx_dlp_result.rename(columns={'tpr_ea':'ea'},inplace=True)
                        trx_dlp_result.loc[:,'ea':] = trx_dlp_result.loc[:,'ea':].apply(pd.to_numeric)
                        return trx_dlp_result
                    else:
                        print('These selected boring(en): ' + values_str + \
                            ' do not contain any trx_results.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   
        
def select_on_ea( trx_result, ea = 2 ):
     if isinstance(trx_result, pd.DataFrame): 
         new_trx_result_ea = trx_result[ ea == trx_result.ea ]
         return new_trx_result_ea
     else:
         raise TypeError('No pandas dataframe was supplied')

def get_average_per_ea( df_trx_result ):
    if isinstance( df_trx_result, pd.DataFrame ):
        ea_list = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        mean_coh = []
        mean_fi = []
        nstd_coh = []
        nstd_fi = []
        N = []
        for i in ea_list:
            trx_temp = select_on_ea( df_trx_result , i)
            N.append(int(len(trx_temp['coh'])))
            mean_coh.append( round( np.mean( trx_temp['coh'] ), 1 ))
            mean_fi.append( round( np.mean( trx_temp['fi'] ), 1 ))
            nstd_coh.append( round( np.std( trx_temp['coh'] )/np.mean( trx_temp['coh'] ),2 ))
            nstd_fi.append( round( np.std( trx_temp['fi'] )/np.mean( trx_temp['fi'] ), 2 ))
        mux = pd.Index(['mean(coh)', 'nstd(coh)', 'mean(fi)', 'nstd(fi)','N elements'])
        stat_df = pd.DataFrame([mean_coh,nstd_coh,mean_fi,nstd_fi,N], index = mux, columns = ea_list)
        stat_df.columns.name = 'ea ='
        return stat_df
    else:
        raise TypeError('No pandas dataframe was supplied.')

def get_least_squares( df_trx_dlp_result, name = 'TRX_DLP', ea = 2, make_plot = True ):
    df = df_trx_dlp_result[df_trx_dlp_result.ea == ea]
    data_full = (df.p, df.q)
    ### Begin Least Squares fitting van een 'linear regression'
    x, y = data_full
    N = len(x)
    x_m = np.mean(x)
    x_quadm = np.sum(x*x)/N
    y_m = np.mean(y)
    yx_quadm = np.sum(x*y)/N

    a = (yx_quadm - y_m*x_m)/(x_quadm - x_m**2) # Hellings Coefficient
    b = coh =  y_m - a*x_m   # Start Coefficent/cohesie
    fi = np.arctan(a) # fi

    def func(a,b,x):
        return a*x + b
    
    E = np.sum((y - func(a,b,x))**2) # Abs. Squared Error
    eps = np.mean((y - func(a,b,x))**2/y**2) # Normalised/Relative Error average for all points
    E_per_n = E/N # Abs.Squared Error per n
    ### Einde Least Squares fitting

    
    if make_plot:
        dlp1, dlp2, dlp3 = df[(df.deelproef_nummer == 1)], df[(df.deelproef_nummer == 2)], df[(df.deelproef_nummer == 3)]
        data_colors = ((dlp1.p, dlp1.q), (dlp2.p, dlp2.q), (dlp3.p, dlp3.q))
        colors = ('red', 'green', 'blue')
        dlp_label = ('dlp 1', 'dlp 2', 'dlp 3')

        fig = plt.figure(figsize=(14,7))
        ax = fig.add_subplot(1,2,1)
        
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,4 )
        ## Plotten verschillende deelproeven
        for data, color, lab in zip(data_colors, colors, dlp_label):
            x2, y2 = data
            ax.scatter(x2, y2, alpha=0.8, c=color, edgecolors='none', s=30, label=lab)
        
        ax.plot([min(x), max(x)], [func(a,b,min(x)), func(a,b,max(x))], c='black', label='least squares fit')
        ax.text((max(x)-min(x))/2, min(y)+0.1*y_m, 'Eq: \u03A4 = \u03C3*tan(' + str(round(fi,3)) + ') + ' + str(round(b,2)) \
            + '\n' + 'Abs Squared Error: ' + str(round(E,1))\
            + '\n' + 'Abs Squared Error/N: ' + str(round(E_per_n,2))\
            + '\n' + 'Abs Error/N: ' + str(round(np.sqrt(E_per_n),2))\
            + '\n' + 'Mean Rel Error: ' + str(round(eps*100, 2)) + '%'\
            + '\n' + 'N: ' + str(N))
        ax.set_title(name)
        ax.legend(loc=2)
        ax.set_xlabel('\u03C3_n Normaalspanning')
        ax.set_ylabel('\u03A4 Schuifspanning ')

        ax2.hist(x,round(N/4))
        ax3.hist(y,round(N/4))
        ax2.set_title('Histogrammen van \u03C3_n en \u03A4')
        ax2.set_ylabel('N')
        ax3.set_ylabel('N')
        ax2.set_xlabel('\u03C3_n als (\u03C3_3 + \u03C3_2)/2')
        ax3.set_xlabel('\u03A4 als (\u03C3_3 - \u03C3_1)/2')

        plt.show()
    return round(fi,3), round(coh,1), round(E), round(E_per_n,1), round(eps*100,1), N

def get_sdp( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        
        if len( gtm_ids ) > 0:
            
            if all(isinstance(x, (int)) for x in gtm_ids):
            
                with psy.connect( 
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin" 
                    ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple( gtm_ids )
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT * FROM sdp WHERE gtm_id IN ' + values_str  
                    cur.execute( query )
                    print('SQL QUERY: \n' + query )  
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        sdp_df = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in cur.description]
                        sdp_df.columns = colnames
                        sdp_df.loc[:,'volumegewicht_droog':] = \
                            sdp_df.loc[:,'volumegewicht_droog':].apply(pd.to_numeric)
                        return sdp_df
                    else:
                        print('These selected boring(en): ' + values_str + \
                            ' do not contain any SDP_Proeven.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

def get_sdp_result( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        
        if len( gtm_ids ) > 0:
            
            if all(isinstance(x, (int)) for x in gtm_ids):
            
                with psy.connect( 
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin" 
                    ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple( gtm_ids )
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT * FROM sdp_result WHERE gtm_id IN ' + values_str  
                    cur.execute( query )
                    print('SQL QUERY: \n' + query )  
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        sdp_result_df = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in cur.description]
                        sdp_result_df.columns = colnames
                        sdp_result_df.loc[:,'load':] = \
                            sdp_result_df.loc[:,'load':].apply(pd.to_numeric)
                        return sdp_result_df
                    else:
                        print('These selected boring(en): ' + values_str + \
                            ' do not contain any SDP_results.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

def join_trx_with_trx_results( gtm_ids, proef_type = 'CD' ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        if len(gtm_ids) > 0:
            if all( isinstance( x, ( int )) for x in gtm_ids ):
                
                with psy.connect( 
                    host = "localhost", 
                    database = "bis", 
                    user = "postgres", 
                    password = "admin" 
                    ) as dbcon:

                    cur = dbcon.cursor()
                    values = tuple(gtm_ids)
                    values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                    query = 'SELECT trx.gtm_id, volumegewicht_droog, volumegewicht_nat, ' \
                        + 'watergehalte, terreinspanning, bezwijksnelheid, trx_result.trx_volgnr, ea, '\
                        + 'coh, fi FROM trx ' \
                        + 'INNER JOIN trx_result ON trx.gtm_id = trx_result.gtm_id '\
                        + 'AND trx.trx_volgnr = trx_result.trx_volgnr '\
                        + 'WHERE proef_type = \'' + proef_type + '\' AND trx.gtm_id IN ' + values_str
                    cur.execute( query )
                    print('SQL QUERY: ' + query )
                    fetched = cur.fetchall()
                    if( len( fetched ) > 0 ):
                        trx_df = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in cur.description]
                        trx_df.columns = colnames
                        trx_df.volumegewicht_nat = trx_df.volumegewicht_nat.astype(float)
                        return trx_df
                    else:
                        raise ValueError('These selected boring(en): ' + values_str + ' do not contain any trx + trx_result.')
            else:
                raise TypeError('not all inputs are ints')    
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:    
        raise TypeError('Input is not a list or tuple')     
