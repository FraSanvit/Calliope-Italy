# import calliope


# model=calliope.Model("test/model.yaml")
# model.run()
# model.plot.summary(to_file="plot_summary.html")

#%%

import os

import pyomo.core as po
import numpy as np
import calliope
import pandas as pd

from calliope.backend.pyomo.util import (
    get_param,
    split_comma_list,
    get_timestep_weight,
    invalid,
)

path_to_model="test/model_v3.yaml"
path_to_result="ev-draft.nc"
scenario = None

#%%
def add_eurocalliope_constraints(model):
    backend_model = model._backend_model
    # if "energy_cap_max_time_varying" in model._model_data.data_vars:
    #     print("Building consumption_max_time_varying constraint")
    #     add_consumption_max_time_varying_constraint(model, backend_model)
    if "energy_cap_max_time_varying" in model._model_data.data_vars:
        print("Building production_max_time_varying constraint")
        add_production_max_time_varying_constraint(model, backend_model)

def equalizer(lhs, rhs, sign):
    if sign == "max":
        return lhs <= rhs
    elif sign == "min":
        return lhs >= rhs
    elif sign == "equals":
        return lhs == rhs
    else:
        raise ValueError("Invalid sign: {}".format(sign))


def add_production_max_time_varying_constraint(model, backend_model):

    def _carrier_production_max_time_varying_constraint_rule(
        backend_model, loc_tech, timestep
    ):
        """
        Set maximum carrier production for technologies with time varying maximum capacity
        """
        energy_cap_max = backend_model.energy_cap_max_time_varying[loc_tech, timestep]
        if invalid(energy_cap_max):
            return po.Constraint.Skip
        model_data_dict = backend_model.__calliope_model_data["data"]
        timestep_resolution = backend_model.timestep_resolution[timestep]
        loc_tech_carriers_out = split_comma_list(
            model_data_dict["lookup_loc_techs_conversion_plus"]["out", loc_tech]
        )

        carrier_prod = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in loc_tech_carriers_out
        )
        return carrier_prod <= (
            backend_model.energy_cap[loc_tech] * timestep_resolution * energy_cap_max
        )

    backend_model.loc_tech_carrier_production_max_time_varying_constraint = po.Set(
        initialize=[
            loc_tech for loc_tech in backend_model.loc_techs_conversion_plus
            if model.inputs.energy_cap_max_time_varying.loc[{"loc_techs": loc_tech}].notnull().all()
        ],
        ordered=True
    )
    model.backend.add_constraint(
        "carrier_production_max_time_varying_constraint",
        ["loc_tech_carrier_production_max_time_varying_constraint", "timesteps"],
        _carrier_production_max_time_varying_constraint_rule,
    )

#%%

def add_consumption_max_time_varying_constraint(model, backend_model):
    
    def remove_carrier(loc_tech_carrier):
        carriers = model.inputs.carriers.to_pandas().to_list()

        for c in carriers:
            if c not in loc_tech_carrier:
                continue
            else:
                loc_tech = loc_tech_carrier.replace("::%s" %c,'')
        
        return(loc_tech)
        
    
    def _carrier_consumption_max_time_varying_constraint_rule(
        backend_model, loc_tech, timestep
    ):
        """
        Set maximum carrier consumption for technologies with time varying maximum capacity
        """
        energy_cap_max = backend_model.energy_cap_max_time_varying[loc_tech, timestep]
        if invalid(energy_cap_max):
            print("MALE MALE MALE!!!!")
            return po.Constraint.Skip
        model_data_dict = backend_model.__calliope_model_data["data"]
        timestep_resolution = backend_model.timestep_resolution[timestep]
        loc_tech_carriers_in = backend_model.loc_tech_carriers_con
        # carriers = model.inputs.carriers.to_pandas().to_list()
        loc_techs_we_want = []
        
        carrier_con = sum(
            backend_model.carrier_con[loc_tech_carrier, timestep]
            for loc_tech_carrier in loc_tech_carriers_in
            if (
            model.inputs.energy_cap_max_time_varying.loc[{"loc_techs": remove_carrier(loc_tech_carrier)}].notnull().all()
            )
        )
        return carrier_con >= (
            -1 * backend_model.energy_cap[loc_tech] * timestep_resolution * energy_cap_max
        )
    
    def loc_techs_cap_time_varying(
        backend_model
    ):
        backend_model=model._backend_model

        
        carriers_we_want = model.inputs.carriers.to_pandas().to_list()
        loc_techs_we_want = []
        
        for loc_tech_temp in backend_model.loc_tech_carriers_con.data():
            loc_techs_we_want.append(str(loc_tech_temp))
        
        k=0
        for loc_tech_temp in loc_techs_we_want:
            for c in carriers_we_want:
                if c in loc_tech_temp:
                    loc_techs_we_want[k] = loc_tech_temp.replace("::%s" %c,'')
                    k += 1
        return loc_techs_we_want
    


    backend_model.loc_tech_carrier_consumption_max_time_varying_constraint = po.Set(
        initialize=[

            loc_tech for loc_tech in loc_techs_cap_time_varying(backend_model)
            
            if (

            model.inputs.energy_cap_max_time_varying.loc[{"loc_techs": loc_tech}].notnull().all()
            )

        ],
        ordered=True
    )
    model.backend.add_constraint(
        "carrier_consumption_max_time_varying_constraint",
        ["loc_tech_carrier_consumption_max_time_varying_constraint", "timesteps"],
        _carrier_consumption_max_time_varying_constraint_rule,
    )
#%%
""" split up of run_model """

current_path = os.path.abspath(os.curdir)

model = calliope.Model(
    os.path.join(current_path, path_to_model),
    scenario=scenario
)

run=model.run(build_only=True)
run.cyclic_storage: False

add_eurocalliope_constraints(model)
new_model = model.backend.rerun()



#%%

# caps = new_model.get_formatted_array('energy_cap').to_pandas()
new_model.plot.summary(to_file="conv_plus_check_v1.html")
# new_model.get_formatted_array('carrier_con').loc[{'techs':'battery','locs':'region1'}].to_pandas().T.min()

# ev_charging_prod=new_model.get_formatted_array('carrier_prod').loc[{'techs':'ev_charging','locs':'region1'}].to_pandas()
# ev_charging_prod.to_csv('ev_charging_prod',index=False)
# tot_cost_ev_charging=new_model.get_formatted_array('cost').loc[{'techs':'ev_charging','locs':'region1'}].to_pandas()

# ev_battery_prod=new_model.get_formatted_array('carrier_prod').loc[{'techs':'ev_battery','locs':'region1'}].to_pandas()
# ev_battery_prod.to_csv('ev_battery_prod',index=False)
# tot_cost_ev_charging=new_model.get_formatted_array('cost').loc[{'techs':'ev_charging','locs':'region1'}].to_pandas()


# ev_battery_storage_r2=new_model.get_formatted_array('storage').loc[{'techs':'ev_battery','locs':'region2'}].to_pandas()
# ev_battery_storage_r3=new_model.get_formatted_array('storage').loc[{'techs':'ev_battery','locs':'region3'}].to_pandas()
# ev_battery_storage_tot=new_model.get_formatted_array('storage').loc[{'techs':'ev_battery'}].to_pandas()
# ev_battery_storage_tot=ev_battery_storage_tot.T
# ev_battery_storage_tot.to_csv('ev_batt_storage',index=True)

regions=(['region1','region2','region3'])

for region in regions:
    power_prod=new_model.get_formatted_array('carrier_prod').loc[{'locs':region,'carriers':'power'}].to_pandas()
    power_con=new_model.get_formatted_array('carrier_con').loc[{'locs':region,'carriers':'power'}].to_pandas()
    ev_power_prod=new_model.get_formatted_array('carrier_prod').loc[{'locs':region,'carriers':'ev_power'}].to_pandas()
    ev_power_con=new_model.get_formatted_array('carrier_con').loc[{'locs':region,'carriers':'ev_power'}].to_pandas()
    ev_battery_storage=new_model.get_formatted_array('storage').loc[{'techs':'ev_battery','locs':region}].to_pandas()
    
    node=pd.concat([power_prod,power_con,ev_power_prod,ev_power_con]).T
    node["ev_battery_storage"]=ev_battery_storage
    # node["ev_battery_exchange"]=-node.iloc[:,30].diff() #30 senza v2g, 33 con v2g , 34 con curtailment
    node.to_csv(region,index=True)
    
    costs=new_model.get_formatted_array('cost').loc[{'locs':region}].to_pandas()
    costs.to_csv(region+'_cost',index=True)
    
capacities=new_model.get_formatted_array('energy_cap').to_pandas()
capacities.to_csv('capacities',index=True)




#%%
"""
    Defines decision variables.

    ==================== ========================================
    Variable             Dimensions
    ==================== ========================================
    energy_cap           loc_techs
    carrier_prod         loc_tech_carriers_prod, timesteps
    carrier_con          loc_tech_carriers_con, timesteps
    cost                 costs, loc_techs_cost
    resource_area        loc_techs_area,
    storage_cap          loc_techs_store
    storage              loc_techs_store, timesteps
    resource_con         loc_techs_supply_plus, timesteps
    resource_cap         loc_techs_supply_plus
    carrier_export       loc_tech_carriers_export, timesteps
    cost_var             costs, loc_techs_om_cost, timesteps
    cost_investment      costs, loc_techs_investment_cost
    purchased            loc_techs_purchase
    units                loc_techs_milp
    operating\\_units     loc_techs_milp, timesteps
    unmet\\_demand        loc_carriers, timesteps
    unused\\_supply       loc_carriers, timesteps
    ==================== ========================================
"""
