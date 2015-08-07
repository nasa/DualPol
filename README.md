DualPol README
--------------
This is an object-oriented Python module that facilitates precipitation retrievals (e.g., hydrometeor type, precipitation rate, precipitation mass, particle size distribution information) from polarimetric radar data. It leverages existing open source radar software packages to perform all-in-one retrievals that are then easily visualized or saved using existing software.

DualPol Installation
--------------------
DualPol works under Python 2.7  and 3.4 on most Mac/Linux setups. Windows installation and other Python versions are currently untested.
<p>
Put dualpol.py in your PYTHONPATH.
<p>
The following dependencies need to be installed first:<br>
<ul>
<li>A robust version of Python 2.7  or 3.4 w/ most standard scientific packages (e.g., numpy, matplotlib, pandas, etc.) - Get one for free here: https://store.continuum.io/cshop/anaconda/
<li>The Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART; https://github.com/ARM-DOE/pyart)
CSU_RadarTools (https://github.com/CSU-Radarmet/CSU_RadarTools) - a Python 3 version can be found here: https://github.com/tjlang/CSU_RadarTools
<li>SkewT (https://pypi.python.org/pypi/SkewT) - a Python 3 version can be found here: https://github.com/tjlang/SkewT
</ul>
Specific import calls in the DualPol source code:<br>
import numpy as np<br>
import warnings<br>
import pyart<br>
import matplotlib.colors as colors<br>
from pyart.io.common import radar_coords_to_cart<br>
from skewt import SkewT<br>
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,<br>
                            csu_dsd, csu_kdp)<br>
<p>
Using DualPol
-------------
To access everything:
import dualpol
<p>
A demonstration notebook is in the notebooks directory.
