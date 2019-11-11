# -*- coding: utf-8 -*-
"""
Created on Mon May 26th 2019

A script to run and post-process the Italy Calliope 20-node model based on a NOS (near-optimal solutions) logic

@author: F.Lombardi
"""
#%% Initialisation
import calliope
from nos_utils import cap_loc_score_potential, cap_loc_calc, update_nos_score_params, update_excl_score_params
import numpy as np
import calliope.core.io
import os

calliope.set_log_verbosity('INFO') #sets the level of verbosity of Calliope's operations

#%% 
'''
Initialising NOS stuff
'''
cost_list = []
slacks = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5]
techs_new = [
             'biogas_new', 'wind_new','wind_offshore','pv_farm_new','pv_rooftop_new','phs_new','battery',
             'inter_zonal_new:FR','inter_zonal_new:AT','inter_zonal_new:CH','inter_zonal_new:SI','inter_zonal_new:GR',
             'inter_zonal_new:NORD','inter_zonal_new:CNOR','inter_zonal_new:CSUD','inter_zonal_new:SUD','inter_zonal_new:SARD','inter_zonal_new:SICI',
             ]
nos_number = 50 # number of NOS to be generated by standard method
excl_nos_number = 3 # number of NOS to be generated for each loc::tech minimisation attempt
desired_slack = 'max_cost20' # selects the desired slack percentage

#%% 
'''
--------------------------------------------------------
---------------ITERATION 0---------------(min total_cost)
--------------------------------------------------------
'''

'''
Model creation, run and saving to netCDF - Iteration 0
'''

model = calliope.Model('Model/model.yaml', scenario='2050_eff') #this version only includes the power sector
model.run()
model.to_netcdf('NetCDFs/results_0.nc')

'''
Alternatively, previously run solutions can be read from netCDF files
'''
# model = calliope.read_netcdf('NetCDFs/results_0.nc')
# model.run(build_only=True, force_rerun=True)
model_0 = calliope.read_netcdf('NetCDFs/results_0.nc')

'''
Computation of nos_scores per location
'''
cap_loc_score_0 = cap_loc_score_potential(model_0,techs=techs_new)

'''
Extrapolation of relevant indicators
'''
#Cost class
costs_0 =  model.get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas()
cost_list.append(costs_0)

'''
Creation and saving of a list of slacked neighbourhoods of the optimal cost
'''
slacked_costs_list = slacks*costs_0 
slacked_costs = {'max_cost1': slacked_costs_list[0], 'max_cost5': slacked_costs_list[1], 'max_cost10': slacked_costs_list[2],
                 'max_cost20': slacked_costs_list[3], 'max_cost30': slacked_costs_list[4], 'max_cost50': slacked_costs_list[5]}

#%%
'''
--------------------------------------------------------
---------------ITERATIONS 1:n---------------(min cap_in_same_locs, within selected total_cost slack)
--------------------------------------------------------
'''

nos_dict = {}

cap_per_loc_dict = {}
cap_loc_score_dict = {}
incremental_score = {}

nos_dict[0] = model_0
incremental_score[0] = cap_loc_score_0
cap_per_loc_dict[0] = cap_loc_calc(model_0, techs=techs_new)

'''
Updating pyomo parameters
'''
model.backend.update_param('objective_cost_class', {'monetary' : 0})
model.backend.update_param('objective_cost_class', {'nos_score' : 1})
model.backend.update_param('group_cost_max', {('monetary','systemwide_max_slacked_cost') : slacked_costs[desired_slack]})
update_nos_score_params(model, incremental_score[0])

'''
Model creation and run - NOS 1:n
'''
n = nos_number

for j in range(1,(n+1)):

    try:
        nos_dict[j] = calliope.read_netcdf('NetCDFs/results_nos_%d.nc' % j)
        for v in nos_dict[j]._model_data.data_vars:
            if (isinstance(nos_dict[j]._model_data[v].values.flatten()[0],(np.bool_,bool))):
                nos_dict[j]._model_data[v] = nos_dict[j]._model_data[v].astype(float)
        cap_loc_score_dict[j] = cap_loc_score_potential(nos_dict[j],techs=techs_new)
        incremental_score[j] = cap_loc_score_dict[j].add(incremental_score[j-1])
        update_nos_score_params(model,incremental_score[j])
        cap_per_loc_dict[j] = cap_loc_calc(nos_dict[j], techs=techs_new)
        
        '''
        Extrapolation of relevant indicators
        '''
        cost_list.append(nos_dict[j].get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas())
    
    except:
        nos_dict[j] = model.backend.rerun()
        for v in nos_dict[j]._model_data.data_vars:
            if (isinstance(nos_dict[j]._model_data[v].values.flatten()[0],(np.bool_,bool))):
                nos_dict[j]._model_data[v] = nos_dict[j]._model_data[v].astype(float)
        cap_loc_score_dict[j] = cap_loc_score_potential(nos_dict[j],techs=techs_new)
        incremental_score[j] = cap_loc_score_dict[j].add(incremental_score[j-1])
        update_nos_score_params(model,incremental_score[j])
        cap_per_loc_dict[j] = cap_loc_calc(nos_dict[j], techs=techs_new)
        
        '''
        Extrapolation of relevant indicators, and saving to NetCDFs
        '''
        cost_list.append(nos_dict[j].get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas())
        nos_dict[j].to_netcdf('NetCDFs/results_nos_%d.nc' % j)
        
        '''
        Stopping criterion: if all loc::tech combinations have been explored
        '''
        if (cap_loc_score_dict[j][list(techs_new)].any().any() !=0) == False:
            break
        else:
            continue

#%%
'''
-------------------------------------------------------------------------
---------------ITERATIONS forcing exlcusion ogf loc::techs---------------(min cap_of_given_loc::tech, within selected total_cost slack)
-------------------------------------------------------------------------
'''

'''
Creation of variables to store the NOS results
'''
count = 1
for t in techs_new:

    '''
    Updating pyomo parameters
    '''
    model.backend.update_param('objective_cost_class', {'nos_score' : 0.01})
    model.backend.update_param('objective_cost_class', {'excl_score' : 0.99})
    model.backend.update_param('group_cost_max', {('monetary','systemwide_max_slacked_cost') : slacked_costs[desired_slack]})
    update_nos_score_params(model, incremental_score[0])
    update_excl_score_params(model,t,cap_per_loc_dict[0],1)
    for tt in (set(techs_new)-set([t])):
        update_excl_score_params(model,tt,cap_per_loc_dict[0],0)
        
    '''
    Model creation and run
    '''
    m = excl_nos_number
    
    for j in range(n+1+m*(count-1),(n+1+m*count)):
    
        try:
            nos_dict[j] = calliope.read_netcdf('NetCDFs/results_nos_%d.nc' % j)
            for v in nos_dict[j]._model_data.data_vars:
                if (isinstance(nos_dict[j]._model_data[v].values.flatten()[0],(np.bool_,bool))):
                    nos_dict[j]._model_data[v] = nos_dict[j]._model_data[v].astype(float)
            cap_loc_score_dict[j] = cap_loc_score_potential(nos_dict[j],techs=techs_new)
            incremental_score[j] = cap_loc_score_dict[j].add(incremental_score[j-1])
            update_nos_score_params(model,incremental_score[j])
            cap_per_loc_dict[j] = cap_loc_calc(nos_dict[j], techs=techs_new)
        
            '''
            Extrapolation of relevant indicators, and saving to NetCDFs
            '''
            cost_list.append(nos_dict[j].get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas())
            
        except:
            nos_dict[j] = model.backend.rerun()
            for v in nos_dict[j]._model_data.data_vars:
                if (isinstance(nos_dict[j]._model_data[v].values.flatten()[0],(np.bool_,bool))):
                    nos_dict[j]._model_data[v] = nos_dict[j]._model_data[v].astype(float)
            cap_loc_score_dict[j] =  cap_loc_score_potential(nos_dict[j],techs=techs_new)
            incremental_score[j] = cap_loc_score_dict[j].add(incremental_score[j-1])
            update_nos_score_params(model,incremental_score[j])
            cap_per_loc_dict[j] = cap_loc_calc(nos_dict[j], techs=techs_new)
        
            '''
            Extrapolation of relevant indicators, and saving to NetCDFs
            '''
            cost_list.append(nos_dict[j].get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas())
            nos_dict[j].to_netcdf('NetCDFs/results_nos_%d.nc' % j)
    
    count += 1

'''
Repeat for tech_groups of interest
'''
tech_groups = {}
tech_group_eu_transm = ['inter_zonal_new:FR','inter_zonal_new:AT','inter_zonal_new:CH','inter_zonal_new:SI','inter_zonal_new:GR']
tech_groups['tech_group_eu_transm'] = tech_group_eu_transm

for t in tech_groups.keys():

    '''
    Updating pyomo parameters
    '''
    model.backend.update_param('objective_cost_class', {'nos_score' : 0.01})
    model.backend.update_param('objective_cost_class', {'excl_score' : 0.99})
    model.backend.update_param('group_cost_max', {('monetary','systemwide_max_slacked_cost') : slacked_costs[desired_slack]})
    update_nos_score_params(model, incremental_score[0])
    for tt in tech_groups[t]:
        update_excl_score_params(model,tt,cap_per_loc_dict[0],1)
    for tt in (set(techs_new)-set(tech_groups[t])):
        update_excl_score_params(model,tt,cap_per_loc_dict[0],0)
        
    '''
    Model creation and run
    '''
    m = excl_nos_number
    
    for j in range(n+1+m*(count-1),(n+1+m*count)):
    
        try:
            nos_dict[j] = calliope.read_netcdf('NetCDFs/results_nos_%d.nc' % j)
            for v in nos_dict[j]._model_data.data_vars:
                if (isinstance(nos_dict[j]._model_data[v].values.flatten()[0],(np.bool_,bool))):
                    nos_dict[j]._model_data[v] = nos_dict[j]._model_data[v].astype(float)
            cap_loc_score_dict[j] = cap_loc_score_potential(nos_dict[j],techs=techs_new)
            incremental_score[j] = cap_loc_score_dict[j].add(incremental_score[j-1])
            update_nos_score_params(model,incremental_score[j])
            cap_per_loc_dict[j] = cap_loc_calc(nos_dict[j], techs=techs_new)
        
            '''
            Extrapolation of relevant indicators, and saving to NetCDFs
            '''
            cost_list.append(nos_dict[j].get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas())
            
        except:
            nos_dict[j] = model.backend.rerun()
            for v in nos_dict[j]._model_data.data_vars:
                if (isinstance(nos_dict[j]._model_data[v].values.flatten()[0],(np.bool_,bool))):
                    nos_dict[j]._model_data[v] = nos_dict[j]._model_data[v].astype(float)
            cap_loc_score_dict[j] =  cap_loc_score_potential(nos_dict[j],techs=techs_new)
            incremental_score[j] = cap_loc_score_dict[j].add(incremental_score[j-1])
            update_nos_score_params(model,incremental_score[j])
            cap_per_loc_dict[j] = cap_loc_calc(nos_dict[j], techs=techs_new)
        
            '''
            Extrapolation of relevant indicators, and saving to NetCDFs
            '''
            cost_list.append(nos_dict[j].get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs','techs']).to_pandas())
            nos_dict[j].to_netcdf('NetCDFs/results_nos_%d.nc' % j)
    
    count += 1


