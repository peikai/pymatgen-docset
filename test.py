
import json
import sqlite3
import time

import numpy as np
import pandas as pd
from emmet.core.thermo import ThermoType
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.apps.battery.conversion_battery import (ConversionElectrode,
                                                      ConversionVoltagePair)
from pymatgen.core import Composition, Element, get_el_sp
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction
from scipy.constants import N_A
from pymatgen.core.units import Charge, Time


def Json2entry(entryJson):
    # Loading the MSONable json file will return a MSONable dict that can be converted to the Entry object
    entryDict = json.loads(entryJson)
    entry = ComputedStructureEntry.from_dict(entryDict)

    return (entry)


# fetch entries from local database via chemical system
def Local_PD(PD_database, chemsys):
    # use sqlite database that is built in python
    dbConnection_REST = sqlite3.connect(PD_database)
    # select entries via chemical system from local database
    cursor_chemsys = dbConnection_REST.cursor()
    # coverage of chemical systems has been assured in local database
    # for a given chemical system with build-in thermo type, there would be no duplicated entry id
    cursor_chemsys.execute("SELECT entry_id, correction FROM chemsys_table WHERE chemsys=?", (chemsys,))
    key_tupleList = cursor_chemsys.fetchall()
    # SQL execution here expects tuples wrapped in parentheses, like ((), ()) for IN
    keys_nestTuple = tuple(key_tupleList)
    # get entries via the foreign keys, convert a list of tuple to a nest tuple as input
    cursor_chemsys.execute("SELECT entry FROM entries_table WHERE (entry_id, correction) IN {}".format(keys_nestTuple))
    entry_tupleList = cursor_chemsys.fetchall()
    # formulate the response, which is a list of tuple objects
    entryJson_list = [e[0] for e in entry_tupleList]
    entryList = [Json2entry(entryJson) for entryJson in entryJson_list]
    
    dbConnection_REST.close()

    return (entryList)


def Tied_entries(phaseDiagram, working_ion):
    # find its coordinate in phase diagram.
    comp = Composition(working_ion)
    c = phaseDiagram.pd_coords(comp)
    # find facets that key element acted as a vertice.
    # vertices of facets are stable phases.
    facet_index_list = list()
    for f, s in zip(phaseDiagram.facets, phaseDiagram.simplexes):
        if s.in_simplex(c, PhaseDiagram.numerical_tol / 10):
            facet_index_list.append(f)
    # covert index list to entry list.
    qhull_entries = phaseDiagram.qhull_entries
    # find other phases in those facets.
    vertice_array = np.array(facet_index_list).flatten()
    tied_entry_id_list = [
        qhull_entries[each].entry_id for each in vertice_array]

    return (tied_entry_id_list)


# Source codes revise
# Goal: 
# construct conversion electrode object allowing for working ion containing phases as initial electrode,
# ensuring properties are valid such as voltage, capacity, rxn, etc.
# Problem: 
# Original ConversionElectrode.from_composition_and_pd() always return a complete set of voltage plateaus,
# from like a Li-free electrode to reference electrode. As a result, some voltage plateaus present Li releasing 
# rather than Li insertion for a Li-containing electrode. 
# Methods: 
# 1. retrieve phase evolutions only along the discharging tieline, instead of a complete tieline always;
# 2. revise element profiles which should discharge from the electrode instead of default working ion free composition
# 3. calculate x_discharge with the proportion of elements other than working ion;
# 4. update framework formula.

# Revise get_element_profile() of pymatgen here, in order to cut off profiles as a subset of it
# by getting critical compositions along the discharging tieline only, instead of a complete tieline.
def get_element_profile(PhaseDiagram, working_ion, comp):
    element = get_el_sp(working_ion)

    if element not in PhaseDiagram.elements:
        raise ValueError("get_transition_chempots can only be called with elements in the phase diagram.")

    el_ref = PhaseDiagram.el_refs[element]
    el_comp = Composition(element.symbol)
    evolution = []

    # delete the codes where the tieline is extended to composition axis
    # gc_comp = Composition({el: amt for el, amt in comp.items() if el != element})
    # for cc in PhaseDiagram.get_critical_compositions(el_comp, gc_comp)[1:]:
    for cc in PhaseDiagram.get_critical_compositions(el_comp, comp)[1:]:
        decomp_entries = list(PhaseDiagram.get_decomposition(cc))
        decomp = [k.composition for k in decomp_entries]
        rxn = Reaction([comp], [*decomp, el_comp])
        rxn.normalize_to(comp)
        c = PhaseDiagram.get_composition_chempots(cc + el_comp * 1e-5)[element]
        amt = -rxn.coeffs[rxn.all_comp.index(el_comp)]
        evolution.append(
            {
                "chempot": c,
                "evolution": amt,
                "element_reference": el_ref,
                "reaction": rxn,
                "entries": decomp_entries,
                "critical_composition": cc,
            }
        )
    return evolution


# Source codes revise
# Revised ConversionElectrode.from_composition_and_pd() of pymatgen
def from_composition_and_pd(comp, pd, working_ion_symbol, allow_unstable):
    working_ion = Element(working_ion_symbol)
    entry = working_ion_entry = None
    for ent in pd.stable_entries:
        if ent.reduced_formula == comp.reduced_formula:
            entry = ent
        elif ent.is_element and ent.reduced_formula == working_ion_symbol:
            working_ion_entry = ent

    if not allow_unstable and not entry:
        raise ValueError(f"Not stable compound found at composition {comp}.")

    # discharging from the electrode instead of default working ion free composition.
    # profile = pd.get_element_profile(working_ion, comp)
    profile = get_element_profile(pd, working_ion, comp)
    # Need to reverse because voltage goes form most charged to most discharged.
    profile.reverse()
    if len(profile) < 2:
        return None

    if working_ion_entry is None:
        raise ValueError("working_ion_entry is None.")
    working_ion_symbol = working_ion_entry.elements[0].symbol
    # elements in normalization_els are used for scale reaction of a vpair, does not affect properties.
    normalization_els = {el: amt for el, amt in comp.items() if el != Element(working_ion_symbol)}
    # delete the codes to allowing for working ion containing phases as electrodes
    # framework = comp.as_dict()
    # if working_ion_symbol in framework:
        # framework.pop(working_ion_symbol)
    # framework = Composition(framework)
    framework = comp

    v_pairs: list[ConversionVoltagePair] = [
        ConversionVoltagePair.from_steps(
            profile[i],
            profile[i + 1],
            normalization_els,
            framework_formula=framework.reduced_formula,
        )
        for i in range(len(profile) - 1)
    ]

    # return cls(
    return ConversionElectrode(
        voltage_pairs=v_pairs,
        working_ion_entry=working_ion_entry,
        initial_comp_formula=comp.reduced_formula,
        framework_formula=framework.reduced_formula,
    )


def x_discharge_calculation(voltage_pair):
    # a customized algorithm to calculate the insertion number of working ions per formula unit to the specified electrode
    # allowing for working ions containing electrode, 
    # it should also work correctly when unstable composition as initial electrode, since the insertion number is calculated with proportion of elements other than working ion 
    framework = voltage_pair.framework
    working_ion_selfcontained_num = framework.num_atoms * framework.get_atomic_fraction(Element(working_ion))
    # working ion number of entries in charged/discharged state per unit formula of initial phase (the framework phase here has been replaced by initial phase)
    working_ion_num_charged = voltage_pair.frac_charge * (framework.num_atoms - working_ion_selfcontained_num) / (1 - voltage_pair.frac_charge)
    working_ion_num_discharged = voltage_pair.frac_discharge * (framework.num_atoms - working_ion_selfcontained_num) / (1 - voltage_pair.frac_discharge)
    # for prev_rxn Li3CrS4 (a metastable lithium phase) -> 0.5 LiS4 + LiCrS2 + 1.5 Li, it release 1.5 working ions
    working_ion_prev_rxn_released = working_ion_selfcontained_num - working_ion_num_charged
    # for curr_rxn Li3CrS4 + 2 Li -> LiCrS2 + 2 Li2S, it releases -2 working ions
    working_ion_curr_rxn_released = working_ion_selfcontained_num - working_ion_num_discharged
    # for overall rxn 3.5 Li + 0.5 LiS4 -> 2 Li2S, which is balanced with products of prev_rxn and curr_rxn,
    # there are 3.5 working ion inserted.
    working_ion_rxn_inserted_num = working_ion_prev_rxn_released - working_ion_curr_rxn_released

    return (working_ion_rxn_inserted_num)


def Line_integral(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    # line equation Ax+By+C=0
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    # polynomial order is always one order
    polynomial_equation = np.poly1d([-A/B, -C/B])
    polynomial_integral = polynomial_equation.integ()
    integral_value = polynomial_integral(x2) - polynomial_integral(x1)

    return (integral_value)


def FoM_calculation(x_discharged_series, normalized_volume_series):
    # a figure of merit (FoM), which is proportional to strain energy density is defined as 
    # the integral of |ln_y|^2, from zero to x_end
    # y is the volume_of_products / volume_of_reactants
    # x is the insertion number of working ions per formula unit of electrode
    y_series = np.power(np.log(normalized_volume_series), 2)
    # list of points (x, y)
    point_list = [p for p in zip(x_discharged_series, y_series)]
    # list of endpoint pair of A-B segment
    point_pair_list = [p for p in zip(point_list, point_list[1:])]
    FoM_piecewise_list = [Line_integral(point_A, point_B) for (point_A, point_B) in point_pair_list]
    FoM_piecewise_series = pd.Series(FoM_piecewise_list, dtype='float64')
    # sum of the piecewise FoM value of each voltage plateaus
    FoM = FoM_piecewise_series.sum()

    return (FoM)


def CE(initial_comp, working_ion, phase_diagram):
    if not Element(working_ion) in initial_comp.elements:
        # take account of metastable electrodes that can be experimentally synthesized.
        conversion_electrode = ConversionElectrode.from_composition_and_pd(initial_comp, phase_diagram, working_ion, allow_unstable=True)
    else:
        # allowing for working ion containing electrodes
        conversion_electrode = from_composition_and_pd(initial_comp, phase_diagram, working_ion, allow_unstable=True)

    return (conversion_electrode)


def FoMs_within_expansion_threshold(CE, vol_threshold=0.3, band_gap_threshold=1):
    x_discharged_series = pd.Series([], dtype='float64')
    volume_ratio_series = pd.Series([], dtype='float64')
    average_voltage_series = pd.Series([], dtype='float64')
    gravimetric_capacity_series = pd.Series([], dtype='float64')
    FoM_series = pd.Series([], dtype='float64')
    conductive_boolean_series = pd.Series([], dtype='object')

    # set the first item as a reference, to get relative values afterwards
    x_discharged_series[0] = 0
    # make the entire path as default
    toEnd_boolean = True

    # acquire voltage plateaus
    voltage_pairs_list = CE.voltage_pairs
    n_steps = len(voltage_pairs_list)
    for step, voltage_pair in zip(range(1, n_steps+1), voltage_pairs_list):
        # calculate the relative volume expansion between endpoint compositions in a given voltage plateau
        volume_ratio = voltage_pair.vol_discharge / voltage_pair.vol_charge
        # if volume_ratio exceed the limits at certain step, not all voltage plateaus are qualified, which is to end
        if not (1-vol_threshold <= volume_ratio <= 1+vol_threshold):
            toEnd_boolean = False
            break
        else:
            volume_ratio_series[step] = volume_ratio
            x_discharged_series[step] = x_discharge_calculation(voltage_pair) + x_discharged_series[step-1]
            average_voltage_series[step] = CE.get_average_voltage(min_voltage=voltage_pair.voltage)
            gravimetric_capacity_series[step] = CE.get_capacity_grav(min_voltage=voltage_pair.voltage, use_overall_normalization=True)
            FoM_series[step] = FoM_calculation(x_discharged_series, volume_ratio_series)
            # it is required that at least one of mixed phases is conductive to conduct electrons as electrode
            # here, a bug in pymatgen codes was fixed to ensure overall naming consistency, see my PR in https://github.com/materialsproject/pymatgen/pull/2483,
            conductive_boolean_series[step] = True in [entry.data['band_gap'] < band_gap_threshold for entry in voltage_pair.entries_discharge]

    # In the case of volume_ratio exceeds the criterion at the first step, tags will not get value from the loop above.
    if step == 1 and toEnd_boolean == False:
        gravimetric_capacity = None
        average_voltage = None
        figure_of_merit = None
        conductive_boolean = None
    else:
        gravimetric_capacity = gravimetric_capacity_series.iloc[-1]
        average_voltage = average_voltage_series.iloc[-1]
        figure_of_merit = FoM_series.iloc[-1]
        # always be conductive along the path or not.
        conductive_boolean = all(conductive_boolean_series.values)

    return ([gravimetric_capacity, average_voltage, figure_of_merit, toEnd_boolean, conductive_boolean])


working_ion_list = ['Li', 'Na', 'K', 'Mg', 'Ca', 'H']
# thermo types: GGA_GGA_U_R2SCAN, GGA_GGA_U or R2SCAN
thermo_type = 'GGA_GGA_U'

# max change of normalized volume, 30%
vol_threshold = 0.3
# bandgap limit to be regarded as conductive, 1eV
band_gap_threshold = 1
# e_above_hull limit to be regarded as not very unstable, 0.05eV
e_above_hull_threshold = 0.050  # eV

for working_ion in working_ion_list:
    start_time = time.time()
    # path of phase diagram database
    PD_database = f'DB/{thermo_type}/PD_{working_ion}.sqlite'
    # set up conversion electrode database
    CE_database = f'DB/{thermo_type}/CE_{working_ion}.sqlite'
    # prepare to calculate figure_of_merit for conversion electrodes
    CE_dataframe = pd.DataFrame([])
    # get entries from local database for each chemical system
    chemsys_list = pd.read_csv(f'Tables/{thermo_type}/{working_ion}/chemsys.csv').chemsys.to_list()
    for chemsys in tqdm(chemsys_list, total=len(chemsys_list)):
        # fetch entries from local database via chemical systems
        entry_list = Local_PD(PD_database, chemsys)
        # if no entry found in local database, the new PhaseDiagram method will raise an error and stop the program
        # see my PR to pymatgen for more details: https://github.com/materialsproject/pymatgen/pull/2819,
        # which is merged in pymatgen v2023.1.30.
        phase_diagram = PhaseDiagram(entry_list)
        # entries that have a tieline connected to reference electrode are not qualified as conversion electrodes
        # working ion entries are not qualified as conversion electrodes
        tied_entries_list = Tied_entries(phase_diagram, working_ion)
        valid_entries_list = [entry for entry in entry_list 
                              if (entry.entry_id not in tied_entries_list and entry.reduced_formula != working_ion)]
        for entry in valid_entries_list:
            # Note: it is not required to use reduced formula for ConversionElectrode construction, even though considering for normalization correction
            # especially in case of unstable entries as input comp in from_composition_and_pd(),
            # which may affect some properties of electrode, like volume inconsistency.
            # This maybe a bug in pymatgen that target entry some different entries having the same reduce formula have e_above_hull != 0
            # On the contrary, using entry.composition as input would always be correct, when a proper scale for rxn is applied.
            # for stable entries, from_composition_and_pd() would always assign for electrode from PD if it is existed,
            material_id = entry.data['material_id']
            theoretical_boolean = entry.data['theoretical']
            e_above_hull = entry.data['e_above_hull']
            fromZero_boolean = Element(working_ion) not in entry.elements
            conversion_electrode = CE(entry.composition, working_ion, phase_diagram)
            framework_formula = conversion_electrode.framework.reduced_formula if not isinstance(conversion_electrode, type(None)) else None
            # FOM calculations with a volume expansion threshold, i.e. volume of (Lithiated) electrode / volume of initial electrode in [0.7,1.3].
            [grav_capacity, average_voltage, figure_of_merit, toEnd_boolean, conductive_boolean] = FoMs_within_expansion_threshold(
                initial_entry, conversion_electrode, vol_threshold=vol_threshold, band_gap_threshold=band_gap_threshold) if not isinstance(conversion_electrode, type(None)) else [None, None, None, None, None]

            profile_dict = {'material_id': initial_material_id,
                            'formula': initial_formula,
                            'framework': framework_formula,
                            'grav_capacity_mAh/g': grav_capacity,
                            'average_voltage_V': average_voltage,
                            'figure_of_merit': figure_of_merit,
                            'stability(e_above_hull)': e_above_hull,
                            'theoretical': theoretical_boolean,
                            'fromZero_boolean': fromZero_boolean,
                            'toEnd_boolean': toEnd_boolean,
                            'conductive_boolean': conductive_boolean
                            }

            CE_dataframe.loc[index, profile_dict.keys()] = profile_dict.values()

CE_stable_dataframe = CE_dataframe.loc[CE_dataframe['stability(e_above_hull)'] == 0]
CE_metastable_dataframe = CE_dataframe.loc[CE_dataframe['stability(e_above_hull)'] > 0]

CE_stable_dataframe.to_csv(f'Tables/{thermo_type}/{working_ion}/FOM_stable.csv', float_format='%.5f', index=False)
CE_metastable_dataframe.to_csv(f'Tables/{thermo_type}/{working_ion}/FOM_metastable.csv', float_format='%.5f', index=False)
CE_dataframe.to_csv(f'Tables/{thermo_type}/{working_ion}/FOM_all.csv', float_format='%.5f', index=False)

    # record time usage for FOMs calculations for conversion electrodes in terms of working ions
    end_time = time.time()
    runtime = end_time - start_time
    with open('runtime.txt', 'a') as t:
        print(f'FOM, thermo type: {thermo_type}, working ion: {working_ion}: , runtime: {(runtime/3600):.2f} h', file=t)