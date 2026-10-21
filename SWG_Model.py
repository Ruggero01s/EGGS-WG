#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint:disable=invalid-name
import os
import gc
import cftime
import xarray as xr
import numpy as np
from scipy.special import erf
import scipy.stats as scistat
import scipy

def Data_load(variable_name,Directory,start_year=1950,end_year=2022):
    '''Load in the simulation coefficeints from generated files'''
    year_str=str(start_year)+'_'+str(end_year)
    filename=Directory+variable_name+'_PreSim_Coeff_'+year_str+'.nc'
    tmp_data=xr.open_dataset(filename)
    tmp_data=tmp_data.drop_vars('Residual_Grid')
    return tmp_data.load()

def XY_load(variable_list,Directory):
    '''Load in the simulation coupling coefficeints'''
    test_list=['t2m','d2m']
    hourly_list=[]
    for var_name in test_list:
        if var_name in variable_list:
            hourly_list.append(var_name)
    full_var_str=''
    h_list=range(0,len(hourly_list))
    for i in h_list:
        full_var_str=full_var_str+hourly_list[i]+'_'
    hourname=Directory+full_var_str+'XY_Matrix.nc'
    Hour_XY=xr.open_dataset(hourname)
    return Hour_XY

def Simulate_Day_Occur(V_1,prob_vector,Omega):
    '''Simulate Daily Occurrence'''
    random_vec=np.random.default_rng().multivariate_normal(
        mean=list(np.zeros(len(V_1))),cov=Omega,method='cholesky')
    V_2=random_vec<=prob_vector
    return V_2, random_vec

def Simulate_Day_Amount(Occurence_vector,Omega,Rng_vector,Coeffs,P_Vec):
    '''Simulate Daily precip amounts'''
        #Coeffs: [Alpha, Beta_1, Beta_2]
    random_vec=np.random.default_rng().multivariate_normal(
        mean=list(np.zeros(len(Occurence_vector))),cov=Omega,method='cholesky')
    TV3=Precipitation_Calculate(Coeffs[0],Coeffs[1],Coeffs[2],
                                Rng_vector,random_vec,P_Vec)
    return TV3*Occurence_vector

def Precipitation_Calculate(Alpha,Beta1,Beta2,occur_rng,amount_rng,P_Value):
    '''Mathematics to simulate precipitation value'''
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

def Simulation_Init(Input_params,Spin_Up_num,month=0):
    '''Initiate the simulation'''
    P_vector=np.zeros(Input_params['S_len'])
    Var_Vector=np.zeros(len(Input_params['H_XY']['X_matrix'].values[0,:,0]))
    I_P0=scistat.norm.ppf(Input_params['TP']['P0_Vector'].values[:,month],0,1)
    I_P1=scistat.norm.ppf(Input_params['TP']['P1_Vector'].values[:,month],0,1)
    HX=Input_params['H_XY']['X_matrix'].values[month,:,:]
    HY=Input_params['H_XY']['Y_matrix'].values[month,:,:]
    rng = Input_params.get('rng', np.random.default_rng())
    occurrence_factor = np.linalg.cholesky(Input_params['TP']['Brissette_Occurrence_Omega'].values[:,:,month])
    for i in range(0,Spin_Up_num):
        #First Initialize precipitation
        prob_vector=np.squeeze(np.where(P_vector.astype(bool), I_P1, I_P0))
        Update_vector = occurrence_factor @ rng.standard_normal(len(P_vector)) <= prob_vector
        P_vector=np.atleast_1d(Update_vector*1)
        #Next T 
        Day_var=np.zeros([len(Input_params['H_XY']['X_matrix'].values[0,:,0]),24])
        for i in range(0,24):
            var_random_vector=rng.standard_normal(len(Var_Vector))
            Var_Vector=np.matmul(HX,Var_Vector)+np.matmul(HY,var_random_vector)
            Day_var[:,i]=Var_Vector
    Init_vectors={'tp':P_vector,'var':Var_Vector}
    return Init_vectors


def Full_Simulate_Day(Sim_Params, Date, input_dict):
    """Simulate daily precipitation and each configured hourly variable."""
    month = int(Date.astype('datetime64[M]').astype(int) % 12)
    previous = np.atleast_1d(input_dict['tp'])
    residual = input_dict['var']
    rain = Sim_Params['TP']
    p0 = scistat.norm.ppf(rain['P0_Vector'].values[:, month])
    p1 = scistat.norm.ppf(rain['P1_Vector'].values[:, month])
    # Selecting avoids 0 * infinity when a fitted probability is exactly 0 or 1.
    threshold = np.where(previous.astype(bool), p1, p0)
    rng = Sim_Params.get('rng')
    if rng is None:
        occurrence, noise = Simulate_Day_Occur(
            previous, threshold, rain['Brissette_Occurrence_Omega'].values[:, :, month])
    else:
        noise = Sim_Params['occurrence_cholesky'][month] @ rng.standard_normal(len(previous))
        occurrence = noise <= threshold
    occurrence = np.atleast_1d(occurrence.astype(int))
    coefficients = [rain[name].values[:, month] for name in
                    ('Alpha_Vector', 'Beta_1_Vector', 'Beta_2_Vector')]
    if rng is None:
        amounts = Simulate_Day_Amount(
            occurrence, rain['Brissette_Amounts_Omega'].values[:, :, month],
            noise, coefficients, threshold)
    else:
        amount_noise = Sim_Params['amount_cholesky'][month] @ rng.standard_normal(len(previous))
        amounts = Precipitation_Calculate(*coefficients, noise, amount_noise, threshold) * occurrence
    x = Sim_Params['H_XY']['X_matrix'].values[month]
    y = Sim_Params['H_XY']['Y_matrix'].values[month]
    hourly_residual = np.empty((len(residual), 24))
    for hour in range(24):
        residual = x @ residual + y @ (rng if rng is not None else np.random.default_rng()).standard_normal(len(residual))
        hourly_residual[:, hour] = residual
    day = min(int((Date.astype('datetime64[D]') - Date.astype('datetime64[Y]')).astype(int)), 364)
    hours = slice(day * 24, (day + 1) * 24)
    output = {'tp_o': occurrence, 'tp_a': np.atleast_1d(amounts)}
    stations = len(occurrence)
    for index, name in enumerate(Sim_Params['Var_list']):
        data = Sim_Params['Var_data'][name]
        mean = np.where(occurrence[None, :], data['Fourier_Mean_Wet'].values[hours],
                        data['Fourier_Mean_Dry'].values[hours])
        std = np.where(occurrence[None, :], data['Fourier_Std_Wet'].values[hours],
                       data['Fourier_Std_Dry'].values[hours])
        output[name] = hourly_residual[index * stations:(index + 1) * stations] * std.T + mean.T
    return output, {'tp': occurrence, 'var': residual}


def Alt_CDF_Calculate(in_res,tp_o,Prob_vec,W_Table,D_Table):
    '''Calculate variable associated with ECDF'''
    #Calculates the value given CDF variables
    prob=0.5*(1+erf(in_res/np.sqrt(2)))
    Ac=np.zeros(len(prob))
    range_iter=range(0,len(prob))
    for ind in range_iter:
        if tp_o[ind]:
            cdf=W_Table[ind,:]
        else:
            cdf=D_Table[ind,:]
        index=prob[ind]*1000
        dist_weight=(prob[ind]-Prob_vec[int(index)])/(0.001)
        Ac[ind]=cdf[int(index)+1]*dist_weight+cdf[int(index)]*(1-dist_weight)
    return Ac


def RH_Calculate(dtas,tas):
    '''calculate RH from d2m and t2m'''
    a1=610.78
    a3=17.27
    a4=35.86
    T0=273.16
    RH=[]
    Ev=a1*np.exp(a3*(dtas-T0)/(dtas-a4))
    Esv=a1*np.exp(a3*(tas-T0)/(tas-a4))
    RH=Ev/Esv
    return RH


def D2M_2_RH(in_data):
    '''Uses the t2m and d2m input data to calculate the Relative Humidity'''
    t2m=in_data['t2m']
    d2m=in_data['d2m']
    RH=RH_Calculate(d2m, t2m)
    ind=RH>=1
    RH[ind]=1
    RH=RH*100
    return RH

def Save_decade(Data, Filename):
    """Save a daily block, including only available variables and metadata."""
    dates = np.asarray(Data['Date']).astype('datetime64[D]').astype('datetime64[ns]')
    ds = xr.Dataset(
        {'Sim_Precipitation': (('simulation_time', 'station'), Data['tp_a'])},
        coords={'simulation_time': dates, 'station': np.arange(Data['tp_a'].shape[1]),
                'hour': np.arange(24), 'latitude': Data['Lat'], 'longitude': Data['Lon']})
    ds['Station_Mask'] = (('latitude', 'longitude'), Data['Station_Mask'])
    for name in ('t2m', 'd2m', 'RH'):
        if name in Data:
            ds['simulation_' + name] = (
                ('simulation_time', 'hour', 'station'), np.moveaxis(Data[name], 2, 0).transpose(0, 2, 1))
    if 'Altitude' in Data:
        ds['Altitude'] = (('latitude', 'longitude'), Data['Altitude'])
    ds.to_netcdf(Filename)
    ds.close()


def Model_run(Directory, Sim_num, Simulation_start_year, Simulation_end_year,
              variable_list, start_year=1950, end_year=2022, spin_up_days=1000,
              start_date=None, end_date=None, seed=None):
    """Run any simulation interval, with a configurable historical training period."""
    current = np.datetime64(start_date or f'{Simulation_start_year}-01-01', 'D')
    end = np.datetime64(end_date or f'{Simulation_end_year}-01-01', 'D')
    if np.isnat(current) or np.isnat(end) or end <= current:
        raise ValueError('Simulation end must be later than simulation start (exclusive).')
    data_dir = Directory + 'Data/'
    sim_dir = Directory + 'Simulation/Simulation_' + str(Sim_num + 1) + '/'
    os.makedirs(sim_dir, exist_ok=True)
    with xr.open_dataset(data_dir + f'A_Omega_{start_year}_{end_year}.nc') as source:
        rain = source.load()
    hourly = {name: Data_load(name, data_dir, start_year, end_year) for name in variable_list}
    xy = XY_load(variable_list, data_dir)
    params = {'TP': rain, 'Var_data': hourly, 'Var_list': variable_list,
              'H_XY': xy, 'S_len': rain.sizes['Station']}
    params['rng'] = np.random.default_rng(seed)
    params['occurrence_cholesky'] = [np.linalg.cholesky(rain['Brissette_Occurrence_Omega'].values[:, :, month]) for month in range(12)]
    params['amount_cholesky'] = [np.linalg.cholesky(rain['Brissette_Amounts_Omega'].values[:, :, month]) for month in range(12)]
    month = int(current.astype('datetime64[M]').astype(int) % 12)
    state = Simulation_Init(params, spin_up_days, month=month)
    altitude = None
    if os.path.isfile(data_dir + 'Altitude.nc'):
        with xr.open_dataset(data_dir + 'Altitude.nc') as source:
            altitude = source['altitude'].values
    while current < end:
        year = int(current.astype('datetime64[Y]').astype(int)) + 1970
        next_decade = np.datetime64(f'{(year // 10 + 1) * 10}-01-01', 'D')
        stop = min(next_decade, end)
        days = np.arange(current, stop, dtype='datetime64[D]')
        count, stations = len(days), params['S_len']
        block = {'Date': days, 'tp_a': np.empty((count, stations)),
                 'Lat': rain['latitude'].values, 'Lon': rain['longitude'].values,
                 'Station_Mask': rain['Station_Mask'].values}
        block.update({name: np.empty((stations, 24, count)) for name in variable_list})
        for index, day in enumerate(days):
            output, state = Full_Simulate_Day(params, day, state)
            block['tp_a'][index] = output['tp_a']
            for name in variable_list:
                block[name][:, :, index] = output[name]
        if 'd2m' in variable_list:
            block['RH'] = D2M_2_RH(block)
        if altitude is not None:
            block['Altitude'] = altitude
        label = f'{year}_{str(stop)[:4]}' if str(current)[5:] == '01-01' and str(stop)[5:] == '01-01' else f'{current}_{stop}'
        filename = f'{sim_dir}Simulation_{label}.nc'
        Save_decade(block, filename)
        print(f'Saved simulation {Sim_num + 1}: {current} to {stop}', flush=True)
        current = stop
    rain.close()
    xy.close()
    for dataset in hourly.values():
        dataset.close()

def Sim_Loader(Sim_Dir):
    '''Loads in an existing simulation output'''
    files=sorted(os.listdir(Sim_Dir))
    data_dict={}
    tmp_list=[]
    for filename in files:
        name=Sim_Dir+filename
        if filename.endswith('.nc') and not filename.startswith('tmp_'):
            tmp_list.append(filename)
            with xr.open_dataset(name,cache=True) as tmp_data:
                data_dict[filename]=tmp_data.load()
                tmp_data.close()
    return tmp_list, data_dict


def Precipitation_Adjust(Base_S_dir,observation_dir,Sim_num,start_year=1950,end_year=2022):
    '''Adjust the precipitation amount simulation distribution'''
    from scipy.interpolate import CubicSpline
    O_filename=observation_dir+f'daily_tp_{start_year}_{end_year}.nc'
    Obs_tp=xr.open_dataset(O_filename)
    t_obs_full_data=Obs_tp['tp'].values
    obs_full_data=t_obs_full_data[Obs_tp['Precipitation_mask'].values]
    #Run through each simulation
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        tmp_data=S_Data[S_Filenames[0]]
        tp=tmp_data['Sim_Precipitation'].values
        time=tmp_data['simulation_time'].values
        Sim_data={}
        Sim_data['Lat']=tmp_data['latitude'].values
        Sim_data['Lon']=tmp_data['longitude'].values
        if len(S_Filenames)>1:
            for z in range(1,len(S_Filenames)):
                tmp_data=S_Data[S_Filenames[z]]
                tmp_tp=tmp_data['Sim_Precipitation'].values
                tmp_time=tmp_data['simulation_time'].values
                tp=np.concatenate([tp,tmp_tp],axis=0)
                time=np.concatenate([time,tmp_time],axis=0)
        sim_full_data=tp[tp>1]
        #Calculate adjustment factors
        ipercent=list(range(0,101,5))
        simulated_months = np.unique(time.astype('datetime64[M]').astype(int) % 12)
        observation_months = Obs_tp['time'].values.astype('datetime64[M]').astype(int) % 12
        seasonal = np.isin(observation_months, simulated_months)
        observed = t_obs_full_data[seasonal][Obs_tp['Precipitation_mask'].values[seasonal]]
        obs_percent=np.percentile(observed,ipercent)
        sim_percent=np.percentile(sim_full_data,ipercent)
        diff_percent=sim_percent-obs_percent
        diff_spline=CubicSpline(sim_percent, diff_percent)
        #Now need to adjust each of the grid cells
        for j in range(0,len(S_Filenames)):
            tmp_data=S_Data[S_Filenames[j]].copy(deep='True')
            tmp_tp=tmp_data['Sim_Precipitation'].values
            for s_ind in range(0,len(tmp_tp[0,:])):
                nonzero_ind=np.logical_not(tmp_tp[:,s_ind]==0)
                gt_50=np.logical_not(np.logical_not(np.greater(tmp_tp[:,s_ind],50)))
                alt_list=[]
                for alt in range(0,int(np.sum(gt_50))):
                    alt_list.append(tmp_tp[int(np.where(gt_50)[0][alt]),s_ind]-diff_spline(50))
                tmp_tp[nonzero_ind,s_ind]=tmp_tp[nonzero_ind,s_ind]-diff_spline(tmp_tp[nonzero_ind,s_ind])
                tmp_tp[gt_50,s_ind]=alt_list
            tmp_data.attrs['negative_precipitation_clipped'] = int(np.sum(tmp_tp < 0))
            np.maximum(tmp_tp, 0, out=tmp_tp)
            tmp_data.attrs['precipitation_adjustment'] = 'Upstream percentile correction with nonnegative rainfall support'
            S_Data[S_Filenames[j]].close()
            f_t_name=Sim_Dir+'tmp_'+S_Filenames[j]
            if os.path.isfile(f_t_name):
                os.remove(f_t_name)
            tmp_data.to_netcdf(f_t_name)
            tmp_data.close()
    Obs_tp.close()
    return


def Temperature_Cycle_Adjust(Base_S_dir,observation_dir,Sim_num,start_year=1950,end_year=2022):
    '''Adjusts the diurnal cycle of temperature to match observations'''
    O_filename=observation_dir+f'Detrended_t2m_{start_year}_{end_year}.nc'
    Obs_t2m=xr.open_dataset(O_filename,decode_timedelta=False)
    Obs_full_data=Obs_t2m['t2m'].values
    Obs_time=Obs_t2m['time'].values
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        tmp_data=S_Data[S_Filenames[0]]
        t2m=tmp_data['simulation_t2m'].values
        time=tmp_data['simulation_time'].values
        Sim_data={}
        Sim_data['Lat']=tmp_data['latitude'].values
        Sim_data['Lon']=tmp_data['longitude'].values
        Station_Mask=tmp_data['Station_Mask'].values
        if len(S_Filenames)>1:
            for z in range(1,len(S_Filenames)):
                tmp_data=S_Data[S_Filenames[z]]
                tmp_t2m=tmp_data['simulation_t2m'].values
                tmp_time=tmp_data['simulation_time'].values
                t2m=np.concatenate([t2m,tmp_t2m],axis=0)
                time=np.concatenate([time,tmp_time],axis=0)
        Sim_season_di=np.zeros([12,24])
        sim_mon=time.astype('datetime64[M]').astype(int) % 12
        Obs_season_di=np.zeros([12,24])
        Obs_mon=Obs_time.astype('datetime64[M]').astype(int) % 12
        lat_len=len(Obs_full_data[0,0,:,0])
        lon_len=len(Obs_full_data[0,0,0,:])
        for i in range(0,12):
            if not np.any(sim_mon == i):
                continue
            Obs_mon_mean=np.nanmean(Obs_full_data[Obs_mon==i,:,:,:])
            Sim_mon_mean=np.nanmean(t2m[sim_mon==i,:,:])
            for z in range(0,24):
                Sim_mon_data=t2m[sim_mon==i,z,:]
                Obs_mon_data=Obs_full_data[Obs_mon==i,z,:,:].astype('float64')
                nan_filter=np.isnan(Station_Mask[:,:])
                for j in range(0,lat_len):
                    for k in range(0,lon_len):
                        if nan_filter[j,k]:
                            Obs_mon_data[:,j,k]=np.nan
                Sim_season_di[i,z]=np.nanmean(Sim_mon_data)-Sim_mon_mean
                Obs_season_di[i,z]=np.nanmean(Obs_mon_data)-Obs_mon_mean
        Adj_vectors=Obs_season_di-Sim_season_di
        #Now need to adjust each of the grid cells
        for j in range(0,len(S_Filenames)):
            tmp_data=S_Data[S_Filenames[j]].copy(deep='True')
            tmp_t2m=tmp_data['simulation_t2m'].values
            tmp_time=tmp_data['simulation_time'].values   
            for time_ind in range(0,len(tmp_time)):
                tmp_mon=int(tmp_time[time_ind].astype('datetime64[M]').astype(int) % 12)
                for i_hour in range(0,24):
                    tmp_t2m[time_ind,i_hour,:]+=Adj_vectors[tmp_mon,i_hour]
            S_Data[S_Filenames[j]].close()
            f_t_name=Sim_Dir+'tmp_'+S_Filenames[j]
            if os.path.isfile(f_t_name):
                os.remove(f_t_name)
            tmp_data.to_netcdf(f_t_name)
            tmp_data.close()
    Obs_t2m.close()
    return

def Sim_replace(Base_S_dir, Sim_num):
    """Atomically replace generated files after a completed adjustment."""
    for index in range(Sim_num):
        directory = Base_S_dir + f'Simulation/Simulation_{index + 1}/'
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.nc') and not filename.startswith('tmp_'):
                temporary = directory + 'tmp_' + filename
                if not os.path.isfile(temporary):
                    raise FileNotFoundError(f'Adjusted output is missing: {temporary}')
                os.replace(temporary, directory + filename)


def metadata_handling(Base_F_dir,Sim_number):
    '''Handles all the metadata for the file saving to disk'''
    adjust_metadata(Base_F_dir, Sim_number)
    Sim_replace(Base_F_dir, Sim_number)
    add_alt_metadata(Base_F_dir, Sim_number)
    Sim_replace(Base_F_dir, Sim_number)
    return


def adjust_metadata(Base_S_dir, Sim_num):
    """Expand available station variables to their geographic grid."""
    for index in range(Sim_num):
        directory = Base_S_dir + f'Simulation/Simulation_{index + 1}/'
        filenames, datasets = Sim_Loader(directory)
        for filename in filenames:
            data = datasets[filename]
            dates = np.asarray(data['simulation_time'].values, dtype='datetime64[ns]')
            hours = (dates[:, None] + np.arange(24)[None, :] * np.timedelta64(1, 'h')).ravel()
            mask = data['Station_Mask'].values
            valid = np.isfinite(mask)
            station = mask[valid].astype(int)

            def grid(values):
                output = np.full((len(values), *mask.shape), np.nan)
                output[:, valid] = values[:, station]
                return output

            ds = xr.Dataset(
                {'simulation_Precipitation': (('simulation_day', 'latitude', 'longitude'),
                                             grid(data['Sim_Precipitation'].values))},
                coords={'simulation_day': dates, 'simulation_time': hours,
                        'latitude': data['latitude'], 'longitude': data['longitude']}, attrs=data.attrs)
            for name in ('t2m', 'd2m', 'RH'):
                key = 'simulation_' + name
                if key in data:
                    values = data[key].values.reshape(len(hours), data.sizes['station'])
                    ds[key] = (('simulation_time', 'latitude', 'longitude'), grid(values))
            if 'Altitude' in data:
                ds['Altitude'] = data['Altitude']
            ds.to_netcdf(directory + 'tmp_' + filename)
            ds.close()
            data.close()

def add_alt_metadata(Base_S_dir, Sim_num):
    """Add units only to variables actually generated."""
    attributes = {
        'simulation_Precipitation': {'units': 'mm', 'long_name': 'Simulated daily precipitation'},
        'simulation_t2m': {'units': 'K', 'long_name': 'Simulated 2m air temperature'},
        'simulation_d2m': {'units': 'K', 'long_name': 'Simulated 2m dewpoint temperature'},
        'simulation_RH': {'units': '%', 'long_name': 'Simulated relative humidity'},
        'Altitude': {'units': 'm', 'long_name': 'Altitude'},
        'latitude': {'units': 'degrees_north'}, 'longitude': {'units': 'degrees_east'},
    }
    for index in range(Sim_num):
        directory = Base_S_dir + f'Simulation/Simulation_{index + 1}/'
        filenames, datasets = Sim_Loader(directory)
        for filename in filenames:
            data = datasets[filename]
            for name, attrs in attributes.items():
                if name in data:
                    data[name].attrs.update(attrs)
            data.to_netcdf(directory + 'tmp_' + filename)
            data.close()

def Nonstationairy_mode(Base_S_dir, observation_dir, Sim_num, start_year=1950, end_year=2022):
    """Restore fitted temperature trends, clamped to the historical interval."""
    trends = {}
    for name in ('t2m', 'd2m'):
        path = observation_dir + f'Detrended_{name}_{start_year}_{end_year}.nc'
        if os.path.isfile(path):
            with xr.open_dataset(path, decode_timedelta=False) as source:
                dates = source['time'].values.astype('datetime64[D]')
                offset = (dates - dates[0]).astype(int)
                pivot = int(source['Detrend Pivot'].values)
                trends[name] = (dates[0], int(offset[-1]), int(offset[pivot]),
                                float(source['Detrend Coefficients'].values[1]))
    for index in range(Sim_num):
        directory = Base_S_dir + f'Simulation/Simulation_{index + 1}/'
        filenames, datasets = Sim_Loader(directory)
        for filename in filenames:
            data = datasets[filename]
            for name, (origin, maximum, pivot, slope) in trends.items():
                key = 'simulation_' + name
                if key in data:
                    elapsed = (data['simulation_time'].values.astype('datetime64[D]') - origin).astype(int)
                    adjustment = slope * (np.clip(elapsed, 0, maximum) - pivot)
                    data[key] = data[key] + xr.DataArray(adjustment, dims='simulation_time')
            if 'simulation_d2m' in data:
                data['simulation_RH'] = (100 * model_rh(data)).clip(max=100)
            data.to_netcdf(directory + 'tmp_' + filename)
            data.close()

def model_rh(data):
    return RH_Calculate(data['simulation_d2m'], data['simulation_t2m'])
