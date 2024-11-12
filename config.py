import os
from pathlib import Path

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'Data')

DPI = 600

DISTILLED_METALS_FILEPATH = Path(CUR_DIR, 'Data/distilled_sewage_metals_samples.csv')
CITROPHYLUM_METALS = Path(CUR_DIR, 'Data/cetrophylum_sewage_metals_samples.csv')
BIOMASS_FILEPATH = Path(CUR_DIR, 'Data/biomass_samples.csv')
PHYSIOCHEMICL_PARAMETERS_PATH = Path(CUR_DIR, 'Data/physiochemical_parameters_samples.csv')

PHYSIO_DIFFRENCE_FILEPATH = Path(DATA_DIR, 'physiochemical_parameters_difference.csv')
METALS_DIFF_DISTILLED_FILEPATH = Path(DATA_DIR, 'distilled_sewage_metals_difference.csv')
METALS_DIFF_CETROPHYLUM_FILEPATH = Path(DATA_DIR, 'biomass_difference_percentage.csv')

TITLE_FONT_SIZE = 16
PLOT = True