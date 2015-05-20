DualPol README
--------------
This is an object-oriented Python module that facilitates precipitation retrievals (e.g., hydrometeor type, precipitation rate, precipitation mass, particle size distribution information) from polarimetric radar data. It leverages existing open source radar software packages to perform all-in-one retrievals that are then easily visualized or saved using existing software.

DualPol Installation
--------------------
DualPol works under Python 2.x on most Mac/Linux setups. Windows installation is currently untested.

Put dualpol.py in your PYTHONPATH.

The following dependencies need to be installed first:
A robust version of Python 2.x w/ most standard scientific packages (e.g., numpy, matplotlib, pandas, etc.) - Get one for free here: https://store.continuum.io/cshop/anaconda/
The Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART; https://github.com/ARM-DOE/pyart)
CSU_RadarTools (https://github.com/CSU-Radarmet/CSU_RadarTools)
SkewT (https://pypi.python.org/pypi/SkewT) - an older GitHub version can be found here: https://github.com/tjlang/SkewT

Specific import calls in the DualPol source code:
import numpy as np
import warnings
import pyart
import matplotlib.colors as colors
from pyart.io.common import radar_coords_to_cart
from skewt import SkewT
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp)

Using DualPol
-------------
To access everything:
import dualpol

A demonstration notebook is under construction.
