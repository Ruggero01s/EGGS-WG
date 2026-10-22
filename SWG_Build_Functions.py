#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint:disable=invalid-name
#pylint:disable=import-outside-toplevel

def Data_Extract(L_Directory,S_Directory,Lats,Lons,start_year=1950,end_year=2022):
    '''Converts an on disk monthly grib file into a variable netcdf file'''
    import xarray as xr
    import os
    Year_start=start_year
    Year_end=end_year
    Lat_max=Lats[0]
    Lat_min=Lats[1]
    Lon_max=Lons[0]
    Lon_min=Lons[1]
    for i in range(Year_start,Year_end+1):
        for j in range(1,13):
            if j>9:
                substr=str(i)+'_'+str(j)
                asubstr=str(i)+'-'+str(j)
            else:
                substr=str(i)+'_0'+str(j)
                asubstr=str(i)+'-0'+str(j)
            ERA5_name='ERA_Land_Multi_var_'
            fullname=L_Directory+ERA5_name+substr+'.grib'
            Savename=S_Directory+'ERA_Land_'+substr+'.nc'
            if not os.path.isfile(Savename):
                base_data = xr.load_dataset(fullname, engine="cfgrib",decode_timedelta=False)
                sub_data=base_data.sel(latitude=slice(Lat_max,Lat_min),
                                longitude=slice(Lon_min,Lon_max),time=asubstr)
                
                sub_data.to_netcdf(Savename)
                print(Savename+' Processed')
    return

def Variable_Extract(variable_str,L_Directory,S_Directory,Lats,Lons,start_year=1950,end_year=2022):
    '''Converts an on disk monthly grib file into a variable netcdf file'''
    import numpy as np
    import xarray as xr
    Year_start=start_year
    Year_end=end_year
    Lat_max=Lats[0]
    Lat_min=Lats[1]
    Lon_max=Lons[0]
    Lon_min=Lons[1]
    flag=0
    for i in range(Year_start,Year_end+1):
        print(variable_str+': year='+str(i))
        for j in range(1,13):
            if j>9:
                substr=str(i)+'_'+str(j)
                asubstr=str(i)+'-'+str(j)
            else:
                substr=str(i)+'_0'+str(j)
                asubstr=str(i)+'-0'+str(j)
            ERA5_name='ERA_Land_'
            fullname=L_Directory+ERA5_name+substr+'.nc'
            if flag==0:
                base_data = xr.load_dataset(fullname,decode_timedelta=False)
                flag=1
                sub_data=base_data.sel(latitude=slice(Lat_max,Lat_min),
                                longitude=slice(Lon_min,Lon_max),time=asubstr)
                var_data = sub_data[variable_str]
            else:
                var_time=var_data['time'].values[-1]
                tmp_data = xr.load_dataset(fullname,decode_timedelta=False)
                sub_data2=var_data.sel(latitude=slice(Lat_max,Lat_min),
                            longitude=slice(Lon_min,Lon_max),
                            time=var_data['time'].values[-1],
                            step=var_data['step'].values[-2])
                sub_data2=sub_data2.expand_dims('time')
                sub_data2=sub_data2.expand_dims('step')
                lat=sub_data2['latitude'].values
                lon=sub_data2['longitude'].values
                sub_data=tmp_data.sel(latitude=slice(Lat_max,Lat_min),
                                longitude=slice(Lon_min,Lon_max),time=asubstr)
                var_data=xr.concat([var_data,sub_data[variable_str]],'time')
                for lat_val in lat:
                    for lon_val in lon:
                        ilat=np.nonzero(np.equal(lat,lat_val))[0][0]
                        ilon=np.nonzero(np.equal(lon,lon_val))[0][0]
                        
                        var_data.loc[{
                            'time':var_time,'step':sub_data['step'].values[-1],
                            'latitude':lat_val,'longitude':lon_val}]=(
                                sub_data2.values[0,0,ilat,ilon]+sub_data[variable_str].values[0,0,ilat,ilon])/2               
    
    Savename=S_Directory+variable_str+'_' +str(Year_start)+'_'+str(Year_end)+'.nc'
    var_data.to_netcdf(Savename)
    return


def detrend_curve_fit(in_data, in_date):
    '''Curve fitting function for detrending process'''
    from scipy.optimize import curve_fit
    p0 = [0., 0., 0., 0., 0., 0., 0., 0.]
    coeff, var_matrix = curve_fit(detrend_curve_func, in_date, in_data, p0=p0, )
    return coeff

def detrend_curve_func(x, *p):
    '''Curve fitting function for detrending process'''
    import numpy as np
    #First 4 Fourier Terms
    a0, a1, a2, a3, a4, a5, a6, a7 = p
    year_len = 365.25
    d = x
    return a0 + a1 * d + a2 * np.sin(2 * np.pi * d / year_len) + a3 * np.cos(2 * np.pi * d / year_len) + a4 * np.sin(
        4 * np.pi * d / year_len) + a5 * np.cos(4 * np.pi * d / year_len) + a6 * np.sin(
        6 * np.pi * d / year_len) + a7 * np.cos(6 * np.pi * d / year_len)

def Line_based_correction(in_fit, in_data, in_d,ind):
    '''make adjustments to the data based on the line'''
    line_vals = in_fit[0] + in_fit[1] * in_d
    end_val = line_vals[ind]
    adjust_vals = end_val - line_vals
    out_data = in_data + adjust_vals
    return out_data

def Variable_Detrender(var,directory,start_year=1950,end_year=2022,reference_year=None):
    '''Full detrending process'''
    #Removes the trend over the observational time peiod 
    import xarray as xr
    import numpy as np
    import os
    #Load in the regular data
    init_name=directory+var+'_'+str(start_year)+'_'+str(end_year)+'.nc'
    var_name=var
    time_name='time'
    t_axis=(1,2,3)
    init_data=xr.open_dataset(init_name,decode_timedelta=False)
    tmp_data=init_data[var_name].values
    time=init_data[time_name].values
    d_values=time.astype('datetime64[D]').astype(int)-np.min(time.astype('datetime64[D]').astype(int))
    #Fit data
    regional_mean=np.nanmean(tmp_data,axis=t_axis)
    tmp_fit_coeff = detrend_curve_fit(regional_mean, d_values)
    #Next do the retroactive adjustments based on the linear fit
    reference_year = min(2020, end_year) if reference_year is None else reference_year
    reference = np.datetime64(f'{reference_year}-01-01')
    ind_2020 = int(np.argmin(np.abs(time - reference)))
    #Adj_value=Line_based_correction(tmp_fit_coeff, regional_mean, d_values,ind_2020)
    #Apply fit to data
    tmp_val=np.ones(np.shape(tmp_data))*np.nan
    for lat in range(0,len(tmp_data[0,0,:,0])):
        for lon in range(0,len(tmp_data[0,0,0,:])):
            for hour in range(0,24):
                tmp_val[:,hour,lat,lon]=Line_based_correction(tmp_fit_coeff, tmp_data[:,hour,lat,lon], d_values,ind_2020)
    ds=init_data
    ds[var_name]=(('time', 'step', 'latitude', 'longitude'),tmp_val)
    ds['Detrend Coefficients']=tmp_fit_coeff
    ds['Detrend Pivot']=ind_2020
    ds.attrs['detrend_reference_date'] = str(time[ind_2020].astype('datetime64[D]'))
    #Output the data to disk
    Save_name=directory+'Detrended_'+var+'_'+str(start_year)+'_'+str(end_year)+'.nc'
    if os.path.isfile(Save_name):
        os.remove(Save_name)
    ds.to_netcdf(Save_name)
    return


def Daily_Precipitation_Extract(Directory,start_year=1950,end_year=2022):
    '''Transform the hourly precipitation data into daily output'''
    import xarray as xr
    import numpy as np
    #Takes the raw downloaded ERA5 files and then processes them into a managable form
    Year_start=start_year
    Year_end=end_year
    loadname=Directory+'tp_' +str(Year_start)+'_'+str(Year_end)+'.nc'
    tp=xr.open_dataset(loadname,decode_timedelta=False)
    new_tp=tp.copy(deep=True)
    tp_values=new_tp['tp']
    #First need to handle the accumulation formatting in ERA5 Land data
    step_list=tp['step']
    nan_inds=np.isnan(tp_values[0,0,:,:].values)
    for i in range(0,23):
        t1=tp.sel(step=step_list[i])
        t2=tp.sel(step=step_list[i+1])
        tp_values.loc[{'step':step_list[i+1]}]=t2['tp']-t1['tp']
    tp=new_tp
    #Next need to transform hourly values to daily
    daily_pr=tp.sum(dim='step')
    daily_pr['tp'].data=daily_pr['tp'].data*1000
    grid=daily_pr['tp']
    for i in range(0,len(nan_inds[:,0])):
        for j in range(0,len(nan_inds[0,:])):
            if nan_inds[i,j]:
                grid[:,i,j]=np.nan
    mask=grid>=1
    daily_pr=daily_pr.assign(Precipitation_mask=mask)
    Savename=Directory+'daily_tp_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    daily_pr.to_netcdf(Savename)
    return

def Markov_Calculation(Directory,start_year=1950,end_year=2022):
    '''Calculating the probabilities of a tranisition in daily precip state'''
    import xarray as xr
    import numpy as np
    import os
    Year_start=start_year
    Year_end=end_year
    loadname=Directory+'daily_tp_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    tp=xr.open_dataset(loadname)
    #Just count instances and then can divide by sum to get the probabilities
    rain_counts=np.zeros([len(tp['tp'][0,:,0]),len(tp['tp'][0,0,:]),12,4])
    total_counts=np.zeros([len(tp['tp'][0,:,0]),len(tp['tp'][0,0,:]),12,4])
    #ordering is p00,p01,p10,p11 where p01 is rain yesterday dry before
    for i in range(0,len(tp['tp'][0,0,:])):
        for j in range(0,len(tp['tp'][0,:,0])):
            sub_data=tp['Precipitation_mask'].isel(latitude=j,longitude=i).values
            date_data=tp['time'].values
            for k in range(2,len(sub_data)):
                if date_data[k] - date_data[k-2] != np.timedelta64(2, 'D'):
                    continue
                ind=2*sub_data[k-2]+sub_data[k-1]
                mon_ind=date_data[k].astype('datetime64[M]').astype(int) % 12
                total_counts[j,i,mon_ind,ind]+=1
                if sub_data[k]:
                    rain_counts[j,i,mon_ind,ind]+=1
    # Undefined previous-state probabilities use the observed monthly wet rate.
    # This retains valid dry-state counts in arid months instead of erasing them.
    monthly_rate=np.zeros(total_counts.shape[:3])
    for month in range(12):
        selected=tp['time'].values.astype('datetime64[M]').astype(int) % 12 == month
        monthly_rate[:,:,month]=tp['Precipitation_mask'].values[selected].mean(axis=0)
    def supported_ratio(numerator, denominator):
        return np.divide(numerator,denominator,out=monthly_rate.copy(),where=denominator>0)
    p0=supported_ratio(rain_counts[:,:,:,0]+rain_counts[:,:,:,2],total_counts[:,:,:,0]+total_counts[:,:,:,2])
    p1=supported_ratio(rain_counts[:,:,:,1]+rain_counts[:,:,:,3],total_counts[:,:,:,1]+total_counts[:,:,:,3])
    Markov_probabilities=np.divide(rain_counts,total_counts,
        out=np.broadcast_to(monthly_rate[:,:,:,None],rain_counts.shape).copy(),where=total_counts>0)
    tp.attrs['unsupported_transition_probabilities']='Observed monthly wet fraction; no artificial wet days added'
    tp.attrs['unsupported_transition_count']=int(np.sum(total_counts==0))
    Markov_probabilities=np.append(Markov_probabilities,np.expand_dims(p0, axis=3),axis=3)
    Markov_probabilities=np.append(Markov_probabilities,np.expand_dims(p1, axis=3),axis=3)
    Probability_labels=['p00','p01','p10','p11','p0','p1']
    #Need to add to the x_array file
    tp['labels']=Probability_labels
    tp['Months']=[1,2,3,4,5,6,7,8,9,10,11,12]
    tp['Markov_probabilities'] = (('latitude','longitude','Months','labels'), Markov_probabilities)
    Savename=Directory+'Markov_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    if os.path.isfile(Savename):
        os.remove(Savename)
    tp.to_netcdf(Savename)
    return

def Cross_Correlator(Directory,start_year=1950,end_year=2022):
    '''Calculates cross correlations and generates matricies necessary for the simulation'''
    import xarray as xr
    import numpy as np
    import os
    def corr2_coeff(A, B):
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]
        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)
        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    Year_start=start_year
    Year_end=end_year
    loadname=Directory+'Markov_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    tp=xr.open_dataset(loadname,decode_timedelta=False)
    #First calculate all of the sequence cross-correlations
    lat_len=len(tp['latitude'])
    lon_len=len(tp['longitude'])
    mon_len=len(tp['Months'])
    mon_list=tp['time'].values.astype('datetime64[M]').astype(int) % 12
    correlation_grid=np.zeros([lat_len,lon_len,lat_len,lon_len,mon_len])
    for imon in range(0,mon_len):
        tmp_data=tp['Precipitation_mask'].values[mon_list==imon]
        for ilat in range(0,lat_len):
            for ilon in range(0,lon_len):
                base_vector=tmp_data[:,ilat,ilon]
                for ilat2 in range(0,lat_len):
                    correlation_grid[ilat,ilon,ilat2,:,imon]=np.squeeze(
                corr2_coeff(tmp_data[:,ilat2,:].T,np.atleast_2d(base_vector)))
    #Transform to omega dimmensions
    #Also need to transform the p0 and p1 values to a corresponding matrix
    nan_count=np.sum(np.isnan(tp['tp'].values[0,:,:]))
    correlation_matrix=np.zeros([lat_len*lon_len-nan_count,lat_len*lon_len-nan_count,mon_len])
    P0_vector=np.zeros([lat_len*lon_len-nan_count,mon_len])
    P1_vector=np.zeros([lat_len*lon_len-nan_count,mon_len])
    Station_mask=np.zeros([lat_len,lon_len])
    
    nan_ind=0
    for i in range(0,lat_len):
        for j in range(0,lon_len):
            station_1_ind=i*lon_len+j-nan_ind
            if not np.isnan(tp['tp'].values[0,i,j]):                    
                Station_mask[i,j]=station_1_ind
            else:
                Station_mask[i,j]=np.nan
                nan_ind+=1
                
    for imon in range(0,mon_len):
        nan_ind=0
        P0_data=tp['Markov_probabilities'].sel(labels='p0',Months=imon+1).values
        P1_data=tp['Markov_probabilities'].sel(labels='p1',Months=imon+1).values
        for i in range(0,lat_len):
            for j in range(0,lon_len):
                station_1_ind=i*lon_len+j-nan_ind
                if not np.isnan(tp['tp'].values[0,i,j]):
                    P0_vector[station_1_ind,imon]=P0_data[i,j]
                    P1_vector[station_1_ind,imon]=P1_data[i,j]
                    tmp_vec=correlation_grid[i,j,:,:,imon]
                    relist=tmp_vec.flatten()
                    active=np.isfinite(Station_mask).ravel()
                    values=relist[active]
                    correlation_matrix[station_1_ind,:,imon]=np.where(np.isfinite(values),values,0.)
                    correlation_matrix[station_1_ind,station_1_ind,imon]=1.
                else:
                    nan_ind+=1
    Station_len=int(np.nanmax(Station_mask))
    ds = xr.Dataset(
        data_vars=dict(correlation_matrix=(["Station","Stations_2","Months"],
                                           correlation_matrix)),
        coords=dict(Station=list(range(0,Station_len+1)),
                    Stations_2=list(range(0,Station_len+1)),
                    Months=list(range(1,13))))
    ds['latitude']=tp['latitude']
    ds['longitude']=tp['longitude']
    ds['Station_Mask']=(('latitude','longitude'),Station_mask)
    ds['P0_vector']= (('Station','Months'),P0_vector)
    ds['P1_vector']= (('Station','Months'),P1_vector)
    Corr_Savename=Directory+'Corr_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    if os.path.isfile(Corr_Savename):
        os.remove(Corr_Savename)
    ds.to_netcdf(Corr_Savename)
    return

def Pos_Def(input_matrix):
    '''Transform a matrix to ensure it is pos def'''
    import numpy as np
    eigenvalues=np.linalg.eig(input_matrix)
    pos_eig_list=np.ones(np.shape(eigenvalues[0]))*1E-12
    inds=eigenvalues[0]>=0
    pos_eig_list[inds]=eigenvalues[0][inds]
    Lambda_matrix=np.diag(pos_eig_list)
    B_Dash_Matrix=np.matmul(eigenvalues[1],np.sqrt(Lambda_matrix))
    #Move from B_dash to B with Row normailzation
    T_matrix=np.zeros(np.shape(B_Dash_Matrix))
    for i in range(0,len(B_Dash_Matrix)):
        t_i=[]
        for j in range(0,len(B_Dash_Matrix)):
            t_i.append(eigenvalues[1][i,j]**2*Lambda_matrix[j][j])
        if np.sum(t_i)!=0:
            T_matrix[i][i]=(np.sum(t_i))**-1
    B_Matrix=np.matmul(np.sqrt(T_matrix),B_Dash_Matrix)
    New_Omega=np.matmul(B_Matrix,B_Matrix.T)
    return New_Omega

def Alt_Pos_Def(input_matrix):
    '''Transform a matrix to ensure it is pos def'''
    import numpy as np
    eigenvalues=np.linalg.eigh(input_matrix)
    pos_eig_list=np.ones(np.shape(eigenvalues[0]))*1E-12
    inds=eigenvalues[0]>=0
    pos_eig_list[inds]=eigenvalues[0][inds]
    Lambda_matrix=np.diag(pos_eig_list)
    Eig_mat=eigenvalues[1]
    Eig_mat_I=np.linalg.inv(Eig_mat)
    New_Omega=np.matmul(Eig_mat,np.matmul(Lambda_matrix,Eig_mat_I))
    return New_Omega

def Amounts_Coeff(Directory,start_year=1950,end_year=2022):
    '''Calculate the distribution coefficeints for precipitation'''
    import xarray as xr
    import numpy as np
    import os
    from scipy import optimize

    Year_start=start_year
    Year_end=end_year

    loadname=Directory+'daily_tp_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    tp=xr.open_dataset(loadname)
    lat=tp['latitude'].values
    lon=tp['longitude'].values
    tp_values=tp['tp'].values
    Precipitation_mask=tp['Precipitation_mask'].values

    def Exp_Calc_LF(params):
        y_val=(params[0]/params[1])*np.exp(-filtered_tp/params[1])+(
            (1-params[0])/params[2])*np.exp(-filtered_tp/params[2])
        negLL=-np.sum(np.log(y_val))
        return negLL

    #Filter the precipitation values down to just those that happen when precipitation occurs
    Alpha=np.zeros([len(lat),len(lon),12])
    Beta_1=np.zeros([len(lat),len(lon),12])
    Beta_2=np.zeros([len(lat),len(lon),12])

    mon_vec=tp['time'].values.astype('datetime64[M]').astype(int) % 12
    for i_mon in range(0,12):
        mon_inds=mon_vec==i_mon
        for i in range(0,len(lat)):
            for j in range(0,len(lon)):
                if np.isnan(tp_values[0,i,j]):
                    Alpha[i,j,i_mon]=np.nan
                    Beta_1[i,j,i_mon]=np.nan
                    Beta_2[i,j,i_mon]=np.nan
                else:
                    tmp_tp=tp_values[mon_inds,i,j]
                    tmp_tp_mask=Precipitation_mask[mon_inds,i,j]
                    filtered_tp=tmp_tp[tmp_tp_mask]
                    param_o=[0.6,11,4]
                    bnds = ((0.1, 0.9), (1, 1000), (0.1, 1000))
                    try:
                        results = optimize.minimize(Exp_Calc_LF, param_o,
                                    bounds=bnds,method='Nelder-Mead',tol=1e-5)
                    except:
                        results = optimize.minimize(Exp_Calc_LF, param_o,
                                        bounds=bnds,method='Powell',tol=1e-5)
                    Alpha[i,j,i_mon]=results.x[0]
                    Beta_1[i,j,i_mon]=results.x[1]
                    Beta_2[i,j,i_mon]=results.x[2]
    Beta_flip_ind=Beta_2>Beta_1
    if Beta_flip_ind.any():
        tmp_Beta_2=np.array(Beta_2)
        tmp_Beta_1=np.array(Beta_1)
        tmp_Alpha=np.array(Alpha)
        tmp_Beta_2[Beta_flip_ind]=Beta_1[Beta_flip_ind]
        tmp_Beta_1[Beta_flip_ind]=Beta_2[Beta_flip_ind]
        tmp_Alpha[Beta_flip_ind]=1-Alpha[Beta_flip_ind]
        Alpha=tmp_Alpha
        Beta_1=tmp_Beta_1
        Beta_2=tmp_Beta_2
        
    tp['Months']=[1,2,3,4,5,6,7,8,9,10,11,12]
    tp['Alpha'] = (('latitude','longitude','Months'), Alpha)
    tp['Beta_1'] = (('latitude','longitude','Months'), Beta_1)
    tp['Beta_2'] = (('latitude','longitude','Months'), Beta_2)
    Savename=Directory+'tp_Params_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    if os.path.isfile(Savename):
        os.remove(Savename)
    tp.to_netcdf(Savename)
    return

def Amounts_Corr(Directory,start_year=1950,end_year=2022):
    '''Calculate the correlations required in precipitation amount'''
    import xarray as xr
    import numpy as np
    import os
    Year_start=start_year
    Year_end=end_year
    loadname=Directory+'Markov_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    tp=xr.open_dataset(loadname)
    #First calculate all of the sequence cross-correlations
    lat_len=len(tp['latitude'])
    lon_len=len(tp['longitude'])
    mon_len=len(tp['Months'])
    mon_list=tp['time'].values.astype('datetime64[M]').astype(int) % 12

    correlation_grid=np.zeros([lat_len,lon_len,lat_len,lon_len,mon_len])
    for imon in range(0,mon_len):
        mask=tp['Precipitation_mask'].values[mon_list==imon]
        tmp_data=tp['tp'].values[mon_list==imon]
        for ilat in range(0,lat_len):
            for ilon in range(0,lon_len):
                tmp_mask_1=mask[:,ilat,ilon]
                base_data=tmp_data[:,ilat,ilon]
                for ilat2 in range(0,lat_len):
                    for ilon2 in range(0,lon_len):
                        tmp_mask_2=mask[:,ilat2,ilon2]
                        base_data2=tmp_data[:,ilat2,ilon2]
                        joint_mask=np.logical_and(tmp_mask_1,tmp_mask_2)
                        if np.sum(joint_mask)==0:
                            correlation_grid[ilat,ilon,ilat2,ilon2,imon]=np.nan
                        else:
                            correlation_grid[ilat,ilon,ilat2,ilon2,imon]=np.corrcoef(
                                base_data[joint_mask],base_data2[joint_mask])[0,1]
    Coeff_filename=Directory+'tp_Params_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    Coeff_dataset=xr.open_dataset(Coeff_filename)
    Alpha=Coeff_dataset['Alpha'].values
    Beta_1=Coeff_dataset['Beta_1'].values
    Beta_2=Coeff_dataset['Beta_2'].values
    nan_count=np.sum(np.isnan(tp['tp'].values[0,:,:]))
    correlation_matrix=np.zeros([lat_len*lon_len-nan_count,lat_len*lon_len-nan_count,mon_len])
    Alpha_Vector=np.zeros([lat_len*lon_len-nan_count,12])
    Beta_1_Vector=np.zeros([lat_len*lon_len-nan_count,12])
    Beta_2_Vector=np.zeros([lat_len*lon_len-nan_count,12])
    for i_mon in range(0,12):
        tmp_vec=Alpha[:,:,i_mon].flatten()
        Alpha_Vector[:,i_mon]=tmp_vec[np.logical_not(np.isnan(tmp_vec))]
        tmp_vec=Beta_1[:,:,i_mon].flatten()
        Beta_1_Vector[:,i_mon]=tmp_vec[np.logical_not(np.isnan(tmp_vec))]
        tmp_vec=Beta_2[:,:,i_mon].flatten()
        Beta_2_Vector[:,i_mon]=tmp_vec[np.logical_not(np.isnan(tmp_vec))]
    Station_mask=np.zeros([lat_len,lon_len])
    for imon in range(0,mon_len):
        nan_ind=0
        for i in range(0,lat_len):
            for j in range(0,lon_len):
                station_1_ind=i*lon_len+j-nan_ind
                if not np.isnan(tp['tp'].values[0,i,j]):
                    Station_mask[i,j]=station_1_ind
                    tmp_vec=correlation_grid[i,j,:,:,imon]
                    relist=tmp_vec.flatten()
                    active=np.isfinite(tp['tp'].values[0]).ravel()
                    values=relist[active]
                    correlation_matrix[station_1_ind,:,imon]=np.where(np.isfinite(values),values,0.)
                    correlation_matrix[station_1_ind,station_1_ind,imon]=1.
                else:
                    Station_mask[i,j]=np.nan
                    nan_ind+=1
    Station_len=int(np.nanmax(Station_mask))
    ds = xr.Dataset(
        data_vars=dict(correlation_matrix=(["Station","Stations_2","Months"], correlation_matrix)),
        coords=dict(Station=list(range(0,Station_len+1)),
                    Stations_2=list(range(0,Station_len+1)),
                    Months=list(range(1,13))))
    ds['latitude']=tp['latitude']
    ds['longitude']=tp['longitude']
    ds['Station_Mask']=(('latitude','longitude'),Station_mask)
    ds['Alpha_Vector']=(('Station',"Months"),Alpha_Vector)
    ds['Beta_1_Vector']=(('Station',"Months"),Beta_1_Vector)
    ds['Beta_2_Vector']=(('Station',"Months"),Beta_2_Vector)
    Corr_Savename=Directory+'Amounts_Corr_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    if os.path.isfile(Corr_Savename):
        os.remove(Corr_Savename)
    ds.to_netcdf(Corr_Savename)
    return

def Omega_Mon(P0,P1,C_X_Obs,C_R_0,imonth):
    '''Parallised Omega matrix month calculation'''
    import numpy as np
    if len(P0) == 1:
        return np.eye(1)
    import time
    print('Calculating month:'+str(imonth))
    def Simulate_Day(V_1,prob_vector,Omega):
        #Simulates the next day of precipitation values given:
            #V_1 Current day values
            #P0 and P1 for the sites
            #Omega matrix
        TV_2=list(np.zeros(len(V_1)))
        random_vec=np.random.default_rng().multivariate_normal(
            mean=TV_2,cov=Omega,method='cholesky')
        V_2=random_vec<=prob_vector
        return V_2

    def Simulation_Running_Function(I_Omega,Spin_up_Iterations,station_count,I_P0,I_P1):
        #Initialize the vector
        Output_List=[]
        simulation_vector=np.zeros([station_count])
        for i in range(0,Spin_up_Iterations):
            prob_vector=np.squeeze(I_P0*[abs(simulation_vector-1)] + I_P1*simulation_vector)
            Update_vector=Simulate_Day(simulation_vector,prob_vector,I_Omega)
            simulation_vector=np.atleast_1d(Update_vector*1)
            Output_List.append(simulation_vector)
        return Output_List

    def Correlate_Calculate(Sim_data):
        #Calculate the correlation coefficeints between the simulated data
        Sim_len=len(Sim_data[0])
        Sim_Matrix_Corr=np.zeros([Sim_len,Sim_len])
        Sim_Grid=np.zeros([Sim_len,len(Sim_data)])
        for i in range(0,Sim_len):
            for j in range(0,len(Sim_data)):
                Sim_Grid[i,j]=Sim_data[j][i]
        for i in range(0,Sim_len):
            for j in range(0,Sim_len):
                if i==j:
                    Sim_Matrix_Corr[i,j]=1
                elif j>i:
                    Sim_Matrix_Corr[i,j]=np.corrcoef(Sim_Grid[i,:],Sim_Grid[j,:])[0,1]
                else:
                    Sim_Matrix_Corr[i,j]=Sim_Matrix_Corr[j,i]
        return Sim_Matrix_Corr
    #Iterative solving process    
    Sim_len=6000
    Sim_Iterations=15
    learning_rate=[0.4,0.2,0.3,0.05]
    station_len=len(P0)
    try:
        np.linalg.cholesky(C_R_0)
    except:
        C_R_0=np.real(Alt_Pos_Def(C_R_0))
        C_R_0=np.round(C_R_0,8)
        try:
            np.linalg.cholesky(C_R_0)
        except:
            C_R_0=np.real(Alt_Pos_Def(C_R_0))
    Simulated_data=Simulation_Running_Function(C_R_0,Sim_len,station_len,P0,P1)
    C_X_Sim_i=Correlate_Calculate(Simulated_data)

    old_diff=1
    flag=True
    learningcount=0
    while flag:
        learningflag=True
        while learningflag:
            C_R_U=C_R_0+learning_rate[learningcount]*(C_X_Obs-C_X_Sim_i)
            C_R=Pos_Def(C_R_U)
            t = time.time()
            Diff_list=[]
            for q in range(0,Sim_Iterations):
                Simulated_data=Simulation_Running_Function(C_R,Sim_len,station_len,P0,P1)
                C_X_Sim=Correlate_Calculate(Simulated_data)
                diff=np.abs(C_X_Obs-C_X_Sim)
                Diff_list.append(diff.mean())
            Diff_mean=np.mean(Diff_list)
            print('Month #'+str(imonth)+' --- '+
                  str(np.round(Diff_mean,3))+' : Cycle of time '+
                  str(np.round(time.time() - t,4)) +' seconds and count = '+
                  str(learningcount)+' Out of '+str(len(learning_rate)))
            if np.abs(Diff_mean)>np.abs(old_diff):
                learningcount+=1
                if learningcount==len(learning_rate):
                    learningflag=False
                    flag=False
            else:
                C_R_0=C_R
                learningflag=False
                if Diff_mean<0.005:
                    flag=False
                old_diff=Diff_mean
    return C_R_0

def O_Omega(Directory,start_year=1950,end_year=2022,workers=None,seed=None):
    '''Calculates the correlations required in occurence simulation'''
    import numpy as np
    import xarray as xr
    import scipy.stats as scistat
    import os
    import time
    import multiprocessing 
  
    #Recreate the Brissette approach for building the omega matrix
    Year_start=start_year
    Year_end=end_year
    Corr_Obs_name=Directory+'Corr_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    CORR=xr.open_dataset(Corr_Obs_name)
    Station_mask=CORR['Station_Mask'].values
    P0_vector=CORR['P0_vector'].values
    P1_vector=CORR['P1_vector'].values

    station_len=len(CORR['Station'])
    Full_C_R=np.zeros(np.shape(CORR['correlation_matrix'].values))
    p=[]
    for imonth in range(0,12):
        P0_v=scistat.norm.ppf(CORR['P0_vector'].values[:,imonth],0,1)
        P1_v=scistat.norm.ppf(CORR['P1_vector'].values[:,imonth],0,1)
        # if np.any(np.isinf(P0_v)):
        #     P0_v[np.isinf(P0_v)]=np.sign(P0_v[np.isinf(P0_v)])*5
        # if np.any(np.isinf(P1_v)):
        #     P1_v[np.isinf(P1_v)]=np.sign(P1_v[np.isinf(P1_v)])*5
        C_X_Obs_v=CORR['correlation_matrix'].values[:,:,imonth]
        C_R_0_v=CORR['correlation_matrix'].values[:,:,imonth]
        p.append((P0_v,P1_v,C_X_Obs_v,C_R_0_v,imonth))
    ##### Multithreading Code
    from ai4water_eggs.parallel import parallel_map
    calibration_report = []
    if seed is None:
        r = parallel_map(Omega_Mon, p, workers)
    else:
        from ai4water_eggs.spatial import calibrate_occurrence
        calibrated = parallel_map(calibrate_occurrence, [task + (seed,) for task in p], workers)
        r = [item[0] for item in calibrated]
        calibration_report = [item[1] for item in calibrated]
    
    for i in range(0,12):
        Full_C_R[:,:,i]=r[i]
          
    ds = xr.Dataset(
        data_vars=dict(Brissette_Omega=(["Station","Stations_2","Months"], Full_C_R)),
        coords=dict(Station=list(range(0,station_len)),
                    Stations_2=list(range(0,station_len)),
                    Months=list(range(1,13))))
    ds['latitude']=CORR['latitude']
    ds['longitude']=CORR['longitude']
    ds['P0_vector']= (('Station','Months'),P0_vector)
    ds['P1_vector']= (('Station','Months'),P1_vector)
    ds['Station_Mask']=(('latitude','longitude'),Station_mask)
    import json
    ds.attrs['calibration_report'] = json.dumps(calibration_report)
    Omega_Savename=Directory+'O_Omega_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    if os.path.isfile(Omega_Savename):
        os.remove(Omega_Savename)
    ds.to_netcdf(Omega_Savename)
    return

def A_Omega_Mon(P0,P1,C_X_Obs,C_R_0,Tmp_B_Omega,Full_Coeff,imonth):
    '''Parallised Omega matrix month calculation'''
    import numpy as np
    if len(P0) == 1:
        return np.eye(1)
    import time
    from scipy.special import erf
    def Simulate_Day_Occur(V_1,prob_vector,Omega):
        #Simulates the next day of precipitation values given:
            #V_1 Current day values
            #P0 and P1 for the sites
            #Omega matrix
        TV_2=list(np.zeros(len(V_1)))
        random_vec=np.random.default_rng().multivariate_normal(
            mean=TV_2,cov=Omega,method='cholesky')
        V_2=random_vec<=prob_vector
        return V_2, random_vec
    def Simulate_Day_Amount(Occurence_vector,Omega,Rng_vector,Coeffs,P_Vec):
            #Coeffs: [Alpha, Beta_1, Beta_2]
        TV_2=list(np.zeros(len(Occurence_vector)))
        random_vec=np.random.default_rng().multivariate_normal(
            mean=TV_2,cov=Omega,method='cholesky')
        TV3=Precipitation_Calculate(Coeffs[0],Coeffs[1],Coeffs[2],Rng_vector,random_vec,P_Vec)
        TV3=TV3*Occurence_vector
        return TV3
    def Precipitation_Calculate(Alpha,Beta1,Beta2,occur_rng,amount_rng,P_Value):
        #Calculate the precipitation value
        U_prob=0.5*(1+erf(occur_rng/np.sqrt(2)))
        P_prob=0.5*(1+erf(P_Value/np.sqrt(2)))
        V_prob=0.5*(1+erf(amount_rng/np.sqrt(2)))
        Scale=Alpha*P_prob
        Beta=np.zeros(len(Scale))
        count=0
        for ind_val in Scale:
            if U_prob[count]>= ind_val:
                Beta[count]=Beta2[count]
            else:
                Beta[count]=Beta2[count] + 2*(Beta1[count]-Beta2[count])*(1-U_prob[count]/ind_val)
            count+=1
        return 1 - Beta*np.log(1 - V_prob)
    def Simulation_Running_Function(I_Omega,Spin_up_Iterations,station_count,
                           I_P0,I_P1,Omega,Amount_Coeff):
        #Initialize the vector
        Occurrence_List=[]
        Precipitation_List=[]
        simulation_vector=np.zeros([station_count])
        for iters in range(0,Spin_up_Iterations):
            prob_vector=np.squeeze(I_P0*[abs(simulation_vector-1)] + I_P1*simulation_vector)
            [Update_vector,RNG]=Simulate_Day_Occur(simulation_vector,prob_vector,Omega)
            simulation_vector=np.atleast_1d(Update_vector*1)
            Occurrence_List.append(simulation_vector)
            #Now simulate occurence
            Update_vector2=Simulate_Day_Amount(
                simulation_vector,I_Omega,RNG,Amount_Coeff,prob_vector)
            amount_vector=np.squeeze(Update_vector2*1)
            Precipitation_List.append(amount_vector)
        return Occurrence_List,Precipitation_List
    def Correlate_Calculate(Sim_Occur,Sim_Amount):
        #Calculate the correlation coefficeints between the simulated data
        Sim_len=len(Sim_Occur[0])
        Sim_Matrix_Corr=np.zeros([Sim_len,Sim_len])
        Sim_O_Grid=np.zeros([Sim_len,len(Sim_Amount)])
        Sim_Grid=np.zeros([Sim_len,len(Sim_Amount)])
        for i in range(0,Sim_len):
            for j in range(0,len(Sim_Amount)):
                Sim_Grid[i,j]=Sim_Amount[j][i]
                Sim_O_Grid[i,j]=Sim_Occur[j][i]
        for i in range(0,Sim_len):
            for j in range(0,Sim_len):
                if i==j:
                    Sim_Matrix_Corr[i,j]=1
                elif j>i:
                    index_vec=np.logical_and(Sim_O_Grid[i,:],Sim_O_Grid[j,:])
                    Sim_Matrix_Corr[i,j]=np.corrcoef(Sim_Grid[i,index_vec],
                                                Sim_Grid[j,index_vec])[0,1]
                else:
                    Sim_Matrix_Corr[i,j]=Sim_Matrix_Corr[j,i]
        return Sim_Matrix_Corr
    print('Calculating month:'+str(imonth))
    Sim_len=12000
    Sim_Iterations=15
    learning_rate=[0.4,0.25,0.1]
    station_len=len(P0)
    
    try:
        np.linalg.cholesky(C_R_0)
    except:
        C_R_0=Pos_Def(C_R_0)
    [Simulated_Occurrence,Simulated_Amounts]=Simulation_Running_Function(C_R_0,
                            Sim_len,station_len,P0,P1,Tmp_B_Omega,Full_Coeff)
    C_X_Sim_i=Correlate_Calculate(Simulated_Occurrence,Simulated_Amounts)
    #Build iterative point from here
    old_diff=1
    flag=True
    learningcount=0
    while flag:
        learningflag=True
        while learningflag:
            C_R_U=C_R_0+learning_rate[learningcount]*(C_X_Obs-C_X_Sim_i)
            C_R=Pos_Def(C_R_U)
            t = time.time()
            Diff_list=[]
            for q in range(0,Sim_Iterations):
                [Simulated_Occurrence,Simulated_Amounts]=Simulation_Running_Function(C_R,
                                        Sim_len,station_len,P0,P1,Tmp_B_Omega,Full_Coeff)
                C_X_Sim=Correlate_Calculate(Simulated_Occurrence,Simulated_Amounts)
                diff=np.abs(C_X_Obs-C_X_Sim)
                Diff_list.append(diff.mean())
            Diff_mean=np.mean(Diff_list)
            print('Month #'+str(imonth)+' --- '+
                  str(np.round(Diff_mean,3))+' : Cycle of time '+
                  str(np.round(time.time() - t,4))
                  +' seconds and count = '+str(learningcount)+
                  ' Out of '+str(len(learning_rate)))
            if np.abs(Diff_mean)>np.abs(old_diff):
                learningcount+=1
                if learningcount==len(learning_rate):
                    learningflag=False
                    flag=False
            else:
                C_R_0=C_R
                learningflag=False
                if Diff_mean<0.005:
                    flag=False
                old_diff=Diff_mean
    return C_R_0
 

def A_Omega(Directory,start_year=1950,end_year=2022,workers=None,seed=None):
    '''Calculates the correlations required in amount simulation'''
    import numpy as np
    import xarray as xr
    import scipy.stats as scistat
    from scipy.special import erf
    import os
    import time
    import multiprocessing
    #recreate the Brissette approach for building the omega matrix
    Year_start=start_year
    Year_end=end_year

    
    Corr_Obs_name=Directory+'Amounts_Corr_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    CORR=xr.open_dataset(Corr_Obs_name)
    B_filename=Directory+'O_Omega_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    B_Omega_dataset=xr.open_dataset(B_filename)
    B_Omega=B_Omega_dataset['Brissette_Omega'].values
    Station_mask=B_Omega_dataset['Station_Mask'].values
    Alpha_1=CORR['Alpha_Vector'].values
    Beta_1=CORR['Beta_1_Vector'].values
    Beta_2=CORR['Beta_2_Vector'].values
    station_len=len(CORR['Station'])
    Full_C_R=np.zeros(np.shape(CORR['correlation_matrix'].values))

    p=[]
    for imonth in range(0,12):
        P0_v=scistat.norm.ppf(B_Omega_dataset['P0_vector'].values[:,imonth],0,1)
        P1_v=scistat.norm.ppf(B_Omega_dataset['P1_vector'].values[:,imonth],0,1)
        C_X_Obs_v=CORR['correlation_matrix'].values[:,:,imonth]
        C_R_0_v=CORR['correlation_matrix'].values[:,:,imonth]
        B_Omega_v=B_Omega[:,:,imonth]
        Full_Coeff_v=[Alpha_1[:,imonth],Beta_1[:,imonth],Beta_2[:,imonth]]
        p.append((P0_v,P1_v,C_X_Obs_v,C_R_0_v,B_Omega_v,Full_Coeff_v,imonth))
    ##### Multithreading Code
    from ai4water_eggs.parallel import parallel_map
    calibration_report = []
    if seed is None:
        r = parallel_map(A_Omega_Mon, p, workers)
    else:
        from ai4water_eggs.spatial import calibrate_amounts
        calibrated = parallel_map(calibrate_amounts, [task + (seed,) for task in p], workers)
        r = [item[0] for item in calibrated]
        calibration_report = [item[1] for item in calibrated]
    
    for i in range(0,12):
        Full_C_R[:,:,i]=r[i]
        
    ds = xr.Dataset(
        data_vars=dict(Brissette_Amounts_Omega=(["Station","Stations_2","Months"], Full_C_R)),
        coords=dict(Station=list(range(0,station_len)),
                    Stations_2=list(range(0,station_len)),
                    Months=list(range(1,13))))
    ds['latitude']=B_Omega_dataset['latitude']
    ds['longitude']=B_Omega_dataset['longitude']
    ds['Station_Mask']=(('latitude','longitude'),Station_mask)
    ds['Brissette_Occurrence_Omega']=(('Station', 'Stations_2', 'Months'),B_Omega)
    ds['Alpha_Vector']=(('Station', 'Months'),Alpha_1)
    ds['Beta_1_Vector']=(('Station', 'Months'),Beta_1)
    ds['Beta_2_Vector']=(('Station', 'Months'),Beta_2)
    ds['P0_Vector']=(('Station', 'Months'),B_Omega_dataset['P0_vector'].values)
    ds['P1_Vector']=(('Station', 'Months'),B_Omega_dataset['P1_vector'].values)

    import json
    ds.attrs['calibration_report'] = json.dumps(calibration_report)
    Omega_Savename=Directory+'A_Omega_'+str(Year_start)+'_'+str(Year_end)+'.nc'
    if os.path.isfile(Omega_Savename):
        os.remove(Omega_Savename)
    ds.to_netcdf(Omega_Savename)
    return

def Fourier_Hour_Parallel(in_var_data,in_var_std,in_full_day_vector,station_num,Station_max):
    from ai4water_eggs.seasonal import fit_hourly
    return fit_hourly(in_var_data, in_var_std, in_full_day_vector, station_num, Station_max)


def Fourier_Day_Parallel(in_var_data,in_var_P,in_D_std,in_W_std,in_full_day_vector,station_num,Station_max,seasonal_fit='legacy'):
    from ai4water_eggs.seasonal import fit_conditioned
    return fit_conditioned(in_var_data, in_var_P, in_D_std, in_W_std, in_full_day_vector, station_num, Station_max, fit_mode=seasonal_fit)
    
    
def Fourier_Generation(Directory,Var,start_year=1950,end_year=2022,workers=None,conditioning='legacy',seasonal_fit='legacy'):
    '''Handles the generation of the temperature model components'''
    import xarray as xr
    import numpy as np
    import os
    import multiprocessing
    variabile_name=Var
    if conditioning not in ('legacy', 'matched'):
        raise ValueError('Temperature residual conditioning must be legacy or matched.')
    if seasonal_fit not in ('legacy', 'calendar-hourly'):
        raise ValueError('Temperature Seasonal Fit must be legacy or calendar-hourly.')
    seasonal_support = []
    scale_fallbacks = []
    def Fourier_Hour_Solve(var_data,var_std,full_day_vector):
        Step_len=len(var_data[0,:,0])
        Station_max=len(var_data[0,0,:])
        n_values=[1,365,730,1095,1460,2190]
        Fourier_mean=np.zeros([Step_len*365,Station_max])
        Fourier_std=np.zeros([Step_len*365,Station_max])
        ##### Multithreading Code
        p=[]
        for istation in range(0,Station_max):
            var_data_v=var_data[:,:,istation]
            var_std_v=var_std[:,:,istation]
            p.append((var_data_v,var_std_v,full_day_vector,istation,Station_max))
        from ai4water_eggs.parallel import parallel_map
        r = parallel_map(Fourier_Hour_Parallel, p, workers)
        #####
        for iv in range(0,len(r)):
            index=r[iv][0]
            Fourier_mean[:,index]=r[iv][1]
            Fourier_std[:,index]=r[iv][2]
        return Fourier_mean,Fourier_std

    def Fourier_Day_Solve(var_data,var_P_data,var_D_std,var_W_std,full_day_vector):
        Station_max=len(var_data[0,0,:])
        Fourier_mean_dry=np.zeros([365,Station_max])
        Fourier_std_dry=np.zeros([365,Station_max])
        Fourier_mean_wet=np.zeros([365,Station_max])
        Fourier_std_wet=np.zeros([365,Station_max])
        n_values=[1,2,3,4,6]
        ##### Multithreading Code
        p=[]
        for istation in range(0,Station_max):
            var_data_v=var_data[:,:,istation]
            var_P_data_v=var_P_data[:,istation]
            var_D_std_v=var_D_std[:,:,istation]
            var_W_std_v=var_W_std[:,:,istation]
            task = (var_data_v,var_P_data_v,var_D_std_v,var_W_std_v,full_day_vector,istation,Station_max)
            p.append(task if seasonal_fit == 'legacy' else task + (seasonal_fit,))
        from ai4water_eggs.parallel import parallel_map
        r = parallel_map(Fourier_Day_Parallel, p, workers)
        seasonal_support.extend(item[5] for item in r)
        #####
        for iv in range(0,len(r)):
            index=r[iv][0]
            Fourier_mean_wet[:,index]=r[iv][1]
            Fourier_std_wet[:,index]=r[iv][2]
            Fourier_mean_dry[:,index]=r[iv][3]
            Fourier_std_dry[:,index]=r[iv][4]
        return Fourier_mean_wet,Fourier_std_wet,Fourier_mean_dry,Fourier_std_dry
    
    year_str=str(start_year)+'_'+str(end_year)
    ERA5_name=Directory+'Detrended_'+variabile_name+'_' + year_str+'.nc'
    ERA5_P_name=Directory+'daily_tp_' + year_str+'.nc'
    data=xr.open_dataset(ERA5_name,decode_timedelta=False)
    precipitation_data=xr.open_dataset(ERA5_P_name,decode_timedelta=False)
    lat_len=len(data['latitude'])
    lon_len=len(data['longitude'])
    Lat=data['latitude']
    Lon=data['longitude']
    Station_mask=np.zeros([lat_len,lon_len])
    nan_ind=0
    for i in range(0,lat_len):
        for j in range(0,lon_len):
            station_1_ind=i*lon_len+j-nan_ind
            if not np.isnan(data[variabile_name].values[0,0,i,j]):
                Station_mask[i,j]=station_1_ind
            else:
                Station_mask[i,j]=np.nan
                nan_ind+=1
    #Transform the temperature data into station format from lat lon
    Station_max=int(np.nanmax(Station_mask))+1
    time_len=len(data['time'])
    Step_len=len(data['step'])
    Time_vector=data['time'].values
    grid=np.zeros([time_len,Step_len,Station_max])
    for i in range(0,Station_max):
        Station_ind=Station_mask==i
        grid[:,:,i]=np.squeeze(data[variabile_name].values[:,:,Station_ind])
    P_grid=np.zeros([time_len,Station_max])
    H_grid=np.zeros([time_len,Step_len,Station_max])
    for i in range(0,Station_max):
        Station_ind=Station_mask==i
        P_grid[:,i]=np.squeeze(precipitation_data['Precipitation_mask'].values[:,Station_ind])
        H_grid[:,:,i]=np.squeeze(data[variabile_name].values[:,:,Station_ind])
    day_vector=Time_vector-Time_vector.astype('datetime64[Y]')
    new_day_vector=[]
    for day in day_vector:
        new_day_vector.append(int(day / np.timedelta64(1, 'D')) + 1)
    P_day_vector=np.array(new_day_vector[:])
    #find std grid for mean purpose
    Std_grid=np.zeros([365,24,Station_max])
    print('Calculating Non-conditioned STD')
    for i in range(0,Station_max):
        for j in range(1,366):
            day_inds=np.equal(new_day_vector,j)
            tmp_T=grid[day_inds,:,i]
            Std_grid[j-1,:,i]=np.nanstd(tmp_T,axis=0)
    Std_grid_D=np.zeros([365,24,Station_max])
    Std_grid_W=np.zeros([365,24,Station_max])
    print('Calculating Conditioned STD')
    for i in range(0,Station_max):
        for j in range(1,366):
            day_inds=np.equal(P_day_vector,j)
            tmp_P=P_grid[day_inds,i]
            tmp_T=H_grid[day_inds,:,i]
            Std_grid_D[j-1,:,i]=np.nanstd(tmp_T[np.logical_not(tmp_P),:],axis=0)
            Std_grid_W[j-1,:,i]=np.nanstd(tmp_T[np.logical_and(tmp_P,tmp_P),:],axis=0)
                        
    F_Mean_Grid,F_Std_Grid=Fourier_Hour_Solve(grid,Std_grid,new_day_vector)
    F_Mean_wet,F_Std_wet,F_Mean_dry,F_Std_dry=Fourier_Day_Solve(H_grid,P_grid,
                                            Std_grid_D,Std_grid_W,P_day_vector)
    if not np.isfinite(F_Std_Grid).all() or np.any(F_Std_Grid <= 1e-8):
        raise ValueError('Unconditioned hourly temperature scales must be finite and positive; review seasonal fit.')
    Hour_dim=list(range(0,len(F_Mean_Grid[:,0])))
    Day_dim=list(range(0,len(F_Mean_wet[:,0])))
    Station_dim=list(range(0,Station_max))
    print('Conditioning Fourier data')
    Conditional_Fourier_Mean_W=np.zeros([len(Hour_dim),Station_max])
    Conditional_Fourier_Mean_D=np.zeros([len(Hour_dim),Station_max])
    Conditional_Fourier_Std_W=np.zeros([len(Hour_dim),Station_max])
    Conditional_Fourier_Std_D=np.zeros([len(Hour_dim),Station_max])
    for i in range(0,len(Day_dim)):
        Day_wet_mean=F_Mean_wet[i,:]
        Day_wet_std=F_Std_wet[i,:]
        Day_dry_mean=F_Mean_dry[i,:]
        Day_dry_std=F_Std_dry[i,:]
        t_list=list(range(i*24,(i+1)*24))
        Hours_mean=F_Mean_Grid[t_list,:]
        Hours_std=F_Std_Grid[t_list,:]
        for j in range(0,Station_max):
            day_mean=np.mean(Hours_mean[:,j])
            day_Std=np.mean(Hours_std[:,j])
            D_Diff=Day_dry_mean[j] - day_mean
            W_Diff=Day_wet_mean[j] - day_mean
            D_Diff_std=Day_dry_std[j] - day_Std
            W_Diff_std=Day_wet_std[j] - day_Std
            Conditional_Fourier_Mean_D[t_list,j]=Hours_mean[:,j]+D_Diff
            Conditional_Fourier_Mean_W[t_list,j]=Hours_mean[:,j]+W_Diff
            Conditional_Fourier_Std_D[t_list,j]=Hours_std[:,j]+D_Diff_std
            Conditional_Fourier_Std_W[t_list,j]=Hours_std[:,j]+W_Diff_std
            from ai4water_eggs.temperature import positive_conditioned_scale
            for label, target, conditional_grid in [('wet',Day_wet_std[j],Conditional_Fourier_Std_W),
                                         ('dry',Day_dry_std[j],Conditional_Fourier_Std_D)]:
                if not np.isfinite(conditional_grid[t_list,j]).all() or np.any(conditional_grid[t_list,j] <= 1e-8):
                    conditional_grid[t_list,j], _ = positive_conditioned_scale(Hours_std[:,j],target)
                    scale_fallbacks.append({'day':i+1,'station':j,'state':label})
    if conditioning == 'matched':
        from ai4water_eggs.temperature import matched_residuals
        Residual_grid = matched_residuals(
            grid, P_grid, new_day_vector,
            mean_wet=Conditional_Fourier_Mean_W, mean_dry=Conditional_Fourier_Mean_D,
            std_wet=Conditional_Fourier_Std_W, std_dry=Conditional_Fourier_Std_D)
    else:
        Residual_grid=np.zeros(np.shape(grid))
        for i in range(0,Station_max):
            for j in range(0,Step_len):
                for k in range(0,time_len):
                    i_day=new_day_vector[k]-1
                    if i_day==365:
                        i_day=364
                    time_index=int(j+i_day*Step_len)
                    Residual_grid[k,j,i]=(
                        grid[k,j,i]-F_Mean_Grid[time_index,i])/F_Std_Grid[time_index,i]
    ds = xr.Dataset(
        data_vars=dict(Fourier_Mean_Dry=(["hour_of_year","Station"], Conditional_Fourier_Mean_D)),
        coords=dict(hour_of_year=Hour_dim,Station=Station_dim))
    ds['Fourier_Std_Dry']=(("hour_of_year","Station"),Conditional_Fourier_Std_D)
    ds['Fourier_Mean_Wet']=(("hour_of_year","Station"),Conditional_Fourier_Mean_W)
    ds['Fourier_Std_Wet']=(("hour_of_year","Station"),Conditional_Fourier_Std_W)
    ds['Unconditioned_Fourier_Mean']=(("hour_of_year","Station"),F_Mean_Grid)
    ds['Unconditioned_Fourier_Std']=(("hour_of_year","Station"),F_Std_Grid)
    ds['longitude']=Lon.values
    ds['latitude']=Lat.values
    ds['Station_Mask']=(('latitude','longitude'),Station_mask)
    ds['hour']=range(0,24)
    ds['time']=Time_vector
    ds['Residual_Grid']=(("time","hour","Station"),Residual_grid)
    import json
    ds.attrs['seasonal_support_report'] = json.dumps(seasonal_support)
    ds.attrs['positive_scale_fallbacks'] = json.dumps(scale_fallbacks)
    ds.attrs['temperature_residual_conditioning'] = conditioning
    ds.attrs['temperature_seasonal_fit'] = seasonal_fit
    ds.attrs['temperature_fit_version'] = 'matched-residual-v1' if conditioning == 'matched' else 'legacy'
    Full_Savename=Directory+variabile_name+'_PreSim_Coeff_'+year_str+'.nc'
    if os.path.isfile(Full_Savename):
        os.remove(Full_Savename)
    ds.to_netcdf(Full_Savename)
    return

def XY_Mon(in_grid,in_month,in_Station_max,in_var_len,lagged_grid=None,fit_mode='legacy'):
    import numpy as np
    '''Parallelised XY Matrix calculation'''
    tmp_grid2=in_grid[:,1:] if lagged_grid is None else in_grid
    tmp_grid3=in_grid[:,:-1] if lagged_grid is None else lagged_grid
    if fit_mode == 'forward':
        from ai4water_eggs.temperature import fit_forward_var
        transition, factor, diagnostics = fit_forward_var(tmp_grid2, tmp_grid3)
        return in_month, transition, factor, diagnostics
    if fit_mode != 'legacy':
        raise ValueError('Temperature VAR fit must be legacy or forward.')
    print('Month #'+str(in_month))
    C0_Matrix=np.zeros([in_Station_max*in_var_len,in_Station_max*in_var_len])
    for i in range(0,in_Station_max*in_var_len):
        for j in range(0,in_Station_max*in_var_len):
            if i==j:
                C0_Matrix[i,j]=1
            elif j>i:
                C0_Matrix[i,j]=np.corrcoef(in_grid[i,:],in_grid[j,:])[0,1]
            else:
                C0_Matrix[i,j]=C0_Matrix[j,i]
    C1_Matrix=np.zeros([in_Station_max*in_var_len,in_Station_max*in_var_len])
    for i in range(0,in_Station_max*in_var_len):
        for j in range(0,in_Station_max*in_var_len):
            # Rows = current state; columns = previous state (simulation uses X @ previous).
            C1_Matrix[i,j]=np.corrcoef(tmp_grid2[i,:],tmp_grid3[j,:])[0,1]
    #Now we need to calculate X and Y vectors
    C0_inv=np.linalg.inv(C0_Matrix)
    X=np.matmul(C1_Matrix,C0_inv)
    G=C0_Matrix-np.matmul(X,np.matrix.transpose(C1_Matrix))
    flag=True
    count=0
    while flag:
        if count==5:
            flag=False
            print('failed to solve for XY')
        try:
            Y=np.linalg.cholesky(G)
            flag=False
            print('Successful Cholesky Decomposition')
        except:
            G=Alt_Pos_Def(G)
            count+=1
            print('Matrix altered after failing Cholesky Decomposition')
    return in_month,X,Y
    
    
def XY_Hourly(Directory,variable_list,start_year=1950,end_year=2022,workers=None,fit_mode='legacy'):
    '''Generates the correlation matricies for the variables that are simulated hourly'''
    import xarray as xr
    import numpy as np
    import os
    import multiprocessing
    year_str=str(start_year)+'_'+str(end_year)
    if fit_mode not in ('legacy', 'forward'):
        raise ValueError('Temperature VAR fit must be legacy or forward.')
    res_data={}
    conditioning_modes = set()
    seasonal_modes = set()
    for variable_str in variable_list:
        ERA5_name=Directory+variable_str+'_PreSim_Coeff_' + year_str+'.nc'
        tmp_data=xr.open_dataset(ERA5_name,decode_timedelta=False)
        data_val=tmp_data['Residual_Grid'].values
        res_data[variable_str]=data_val
        conditioning_modes.add(tmp_data.attrs.get('temperature_residual_conditioning', 'legacy'))
        seasonal_modes.add(tmp_data.attrs.get('temperature_seasonal_fit', 'legacy'))
    time_vec=tmp_data['time'].values
    time_repeat=(time_vec[:, None] + np.arange(24)[None, :] * np.timedelta64(1, 'h')).ravel()
    Station_max=len(data_val[0,0,:])
    monvec=time_repeat.astype('datetime64[M]').astype(int) % 12
    time_len=len(data_val[:,0,0])*len(data_val[0,:,0])
    #Want the format of [t2m[0],t2m[1]....t2m[258],d2m[0],...d2m[258]]
    var_len=len(variable_list)
    X_matrix=np.zeros([12,Station_max*var_len,Station_max*var_len])
    Y_matrix=np.zeros([12,Station_max*var_len,Station_max*var_len])
    Full_grid=np.zeros([Station_max*var_len,time_len])
    for j in range(0,Station_max):
        for z in range(0,var_len):
            Data_vector=res_data[variable_list[z]][:,:,j].flatten()
            Full_grid[j+z*Station_max,:]=Data_vector[:]
    ##### Multithreading Code
    p=[]
    for imonth in range(0,12):
        # Use genuine adjacent hours; excluded years/days must not create false lags.
        adjacent = np.diff(time_repeat) == np.timedelta64(1, 'h')
        current = np.flatnonzero(adjacent & (monvec[1:] == imonth)) + 1
        tmp_grid = Full_grid[:, current]
        previous_grid = Full_grid[:, current - 1]
        arguments = (tmp_grid,imonth,Station_max,var_len,previous_grid)
        p.append(arguments if fit_mode == 'legacy' else arguments + (fit_mode,))
    from ai4water_eggs.parallel import parallel_map
    r = parallel_map(XY_Mon, p, workers)
    #####
    for iv in range(0,len(r)):
        index=r[iv][0]
        X_matrix[index,:,:]=r[iv][1]
        Y_matrix[index,:,:]=r[iv][2]
    ds = xr.Dataset(
        data_vars=dict(X_matrix=(["month","Arb_dimm","Arb_dimm_2"], X_matrix)),
        coords=dict(month=range(0,12),Arb_dimm=list(range(0,Station_max*var_len)),
        Arb_dimm_2=list(range(0,Station_max*var_len))))
    ds['Y_matrix']= (("month",'Arb_dimm','Arb_dimm_2'),Y_matrix)
    import json
    ds.attrs['temperature_var_fit'] = fit_mode
    ds.attrs['temperature_fit_version'] = 'forward-var-v1' if fit_mode == 'forward' else 'legacy'
    ds.attrs['temperature_residual_conditioning'] = ','.join(sorted(conditioning_modes))
    ds.attrs['temperature_seasonal_fit'] = ','.join(sorted(seasonal_modes))
    if fit_mode == 'forward':
        ds.attrs['temperature_var_diagnostics'] = json.dumps([item[3] for item in r])
    full_var_str=''
    for i in range(0,len(variable_list)):
        full_var_str=full_var_str+variable_list[i]+'_'
    XY_Savename=Directory+full_var_str+'XY_Matrix.nc'
    if os.path.isfile(XY_Savename):
        os.remove(XY_Savename)
    ds.to_netcdf(XY_Savename)
    return


def orography_prep(L_Directory,S_Directory,LatBounds,LonBounds,start_year=1950,end_year=2022):
    '''Prepares orography for the output'''
    import xarray as xr
    import numpy as np
    import os
    loadname=L_Directory+'ERA_Geopotential.nc'
    data=xr.open_dataset(loadname,decode_timedelta=False)
    geopotential=np.squeeze(data['z'].values)
    lat=data['latitude'].values
    lon=data['longitude'].values
    
    t2m_name=S_Directory+'t2m_'+str(start_year)+'_'+str(end_year)+'.nc'
    t2m=xr.open_dataset(t2m_name,decode_timedelta=False)
    #Set coeffs
    R=6370000
    g=9.80665
    #Restrict geopotential to the appropriate
    t2m_lat=t2m['latitude'].values
    t2m_lon=t2m['longitude'].values
    
    lat_1=np.greater_equal(lat,np.round(np.min(t2m_lat),1))
    lat_2=np.less_equal(lat,np.round(np.max(t2m_lat),1))
    lats=np.logical_and(lat_1,lat_2)
    lon_1=np.greater_equal(lon,np.round(np.min(t2m_lon),1))
    lon_2=np.less_equal(lon,np.round(np.max(t2m_lon),1))
    lons=np.logical_and(lon_1,lon_2)
    
    if len(t2m['latitude'].values)!=np.sum(lats):
        tlat_0=t2m_lat[0]-lat[np.where(lats)[0][0]]
        tlat_1=t2m_lat[-1]-lat[np.where(lats)[0][-1]]
        if np.abs(tlat_0)>np.abs(tlat_1):
            ind=np.where(lats)[0][-1]+1
            aind=np.where(lats)[0][0]-1
        else:
            ind=np.where(lats)[0][0]-1 
            aind=np.where(lats)[0][-1]+1         
        if len(t2m['longitude'].values)-int(np.sum(lons))==2:
                    lons[aind]=True      
        lats[ind]=True

    if len(t2m['longitude'].values)!=int(np.sum(lons)):
        tlon_0=t2m_lon[0]-lon[np.where(lons)[0][0]]
        tlon_1=t2m_lon[-1]-lon[np.where(lons)[0][-1]]
        if np.abs(tlon_0)>np.abs(tlon_1):
            ind=np.where(lons)[0][-1]+1
            aind=np.where(lons)[0][0]-1
        else:
            ind=np.where(lons)[0][0]-1
            aind=np.where(lons)[0][-1]+1
        if len(t2m['longitude'].values)-int(np.sum(lons))==2:
            lons[aind]=True
        lons[ind]=True
    geopotential_grid=geopotential[lats,:]
    geopotential_grid=geopotential_grid[:,lons]
    #Calculate the altitude
    altitude=geopotential_grid*R/(g*R-geopotential_grid)
    #need to load in the t2m data as a way to filter nans
    nangrid=np.sum(np.sum(np.isnan(t2m['t2m'].values),axis=0),axis=0)-1
    nangrid=nangrid!=0
    altitude[nangrid]=np.nan
    Savename=S_Directory+'Altitude.nc'
    ds = xr.Dataset(
        data_vars=dict(altitude=(["Latitude",'Longitude'], altitude)),
        coords=dict(Latitude=t2m["latitude"].values,
                    Longitude=t2m["longitude"].values))
    if os.path.isfile(Savename):
        os.remove(Savename)
    ds.to_netcdf(Savename)
    return