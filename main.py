import os
import config as config
import pdb

from visualizations.distilled_sewage_metal_concentrations import plot_distilled_distributions_for_metals
from visualizations.distilled_sewage_metal_concentrations import plot_distilled_sewage_reduction_percentage
from visualizations.cetrophylum_metal_concentrations import plot_cetrophylum_distributions_for_metals
from visualizations.cetrophylum_metal_concentrations import plot_cetrophylum_reduction_percentage
from visualizations.biomass import plot_biomass_distribution_and_percentage_difference
from visualizations.physiochemical_concentrations import plot_physiochemical_distributions
from visualizations.physiochemical_concentrations import plot_physiochemical_reduction_percentage
from visualizations.coorelations import plot_corelation_parameters_and_distilled_metals
if config.PLOT:
    
    plot_distilled_distributions_for_metals()
    plot_distilled_sewage_reduction_percentage()
    plot_cetrophylum_distributions_for_metals()
    plot_cetrophylum_reduction_percentage()   
    plot_biomass_distribution_and_percentage_difference()
    plot_physiochemical_distributions()
    plot_physiochemical_reduction_percentage()
    
    plot_cetrophylum_reduction_percentage()  
    #plot_corelation_parameters_and_distilled_metals()