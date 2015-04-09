"""
Title/Version
-------------
Python Interface to Dual-Pol Radar Algorithms (DualPol)
DualPol v0.5
Developed & tested with Python 2.7.8
Last changed 3/13/2015
    
    
Author
------
Timothy Lang
NASA MSFC
timothy.j.lang@nasa.gov
(256) 961-7861


Overview
--------
To access this module, add the following to your program and then make sure
the path to this script is in your PYTHONPATH:
import dualpol


Notes
-----
Dependencies: numpy, pyart, singledop, warnings, skewt, csu_radartools


References
----------


Change Log
----------
v0.5 Major Changes (03/13/15):
1. KDP calculation implemented.
2. Moved keyword arguments to separate dictionary (kwargs) and implemented 
   check_kwargs() function to process them.

v0.4 Major Changes (03/05/15):
1. DSD calculations implemented.
2. Project renamed to DualPol from RadBro.

v0.3 Major Changes (02/20/15):
1. Rainfall rate implemented

v0.2 Major Changes (01/27/15):
1. Ice/liquid mass calculations implemented.

v0.1 Functionality(01/26/15):
1. Summer HID calculations implemented.
2. Support for sounding import.

"""

import numpy as np
import warnings
import pyart
import matplotlib.colors as colors
from pyart.io.common import radar_coords_to_cart
from skewt import SkewT
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp)
from singledop import fn_timer

RNG_MULT = 1000.0
DEFAULT_WEIGHTS = csu_fhc.DEFAULT_WEIGHTS
BAD = -32768
DEFAULT_SDP = 12

#####################################

DEFAULT_KW = {'dz': 'DZ', 'dr': 'DR', 'dp': None, 'rh': 'RH',
          'kd': None, 'ld': None, 'sounding': None,
          'verbose': False, 'thresh_sdp': DEFAULT_SDP, 'fhc_T_factor': 1,
          'fhc_weights': DEFAULT_WEIGHTS, 'fhc_name': 'FH', 'band': 'S',
          'fhc_method': 'hybrid', 'kdp_method': 'CSU', 'bad': BAD,
          'use_temp': True, 'ice_flag': False, 'dsd_flag': True,
          'fhc_flag': True, 'rain_method': 'hidro', 'precip_flag': True,
          'liquid_ice_flag': True, 'winter': False}

kwargs = np.copy(DEFAULT_KW)

#####################################

class DualPolRetrieval(object):

    def __init__(self, radar, **kwargs):
        """
        radar = Py-ART radar object
        dz = String name of reflectivity field
        dr = String name of differential reflectivity field
        kd = String name of specific differential phase field
        rh = String name of correlation coefficient field
        ld = String name of linear depolarization ratio field
        dp = String name of differential phase field
        sounding = Name of UWYO sounding file or 2xN array where:
                   sounding['z'] = Heights (km MSL), must be montonic
                   sounding['T'] = Temperatures (C)
        winter = Flag to note whether to use wintertime retrievals
        band = Radar frequency band letter
        """
        #Set radar fields
        kwargs = check_kwargs(kwargs, DEFAULT_KW)
        self.verbose = kwargs['verbose']
        flag = self.do_radar_check(radar)
        if not flag:
            return
        self.name_dz = kwargs['dz']
        self.name_dr = kwargs['dr']
        self.name_kd = kwargs['kd']
        self.name_rh = kwargs['rh']
        self.name_ld = kwargs['ld']
        self.name_dp = kwargs['dp']
        self.kdp_method = kwargs['kdp_method']
        self.bad = kwargs['bad']
        self.thresh_sdp = kwargs['thresh_sdp']
        flag = self.do_name_check()
        if not flag:
            return
        
        #Get sounding info
        self.T_flag = kwargs['use_temp']
        self.T_factor = kwargs['fhc_T_factor']
        self.get_sounding(kwargs['sounding'])
        self.winter_flag = kwargs['winter']
        
        #Do FHC
        if kwargs['fhc_flag']:
            self.fhc_weights = kwargs['fhc_weights']
            self.fhc_method = kwargs['fhc_method']
            self.band = kwargs['band']
            self.name_fhc = kwargs['fhc_name']
            self.get_hid()
        
        #Other precip retrievals
        if kwargs['precip_flag']:
            self.get_precip_rate(ice_flag=kwargs['ice_flag'],
                                 rain_method=kwargs['rain_method'])
        if kwargs['dsd_flag']:
            self.get_dsd()
        if kwargs['liquid_ice_flag']:
            self.get_liquid_and_frozen_mass()

    def do_radar_check(self, radar):
        """
        Checks to see if radar variable is a file or a Py-ART radar object.
        """
        if isinstance(radar, str):
            try:
                self.radar = pyart.io.read(radar)
            except:
                warnings.warn('Bad file name provided, try again')
                return False
        else:
            self.radar = radar
        #Checking for actual radar object
        try:
            junk = self.radar.latitude['data']
        except:
            warnings.warn('Need a real Py-ART radar object, try again')
            return False
        return True #Actual radar object provided by user
        
    def do_name_check(self):
        """
        Simple name checking to ensure the file actually contains the
        right polarimetric variables.
        """
        wstr = ' field not in radar object, check variable names'
        if self.name_dz in self.radar.fields:
            if self.name_dr in self.radar.fields:
                if self.name_rh in self.radar.fields:
                    if self.name_ld is not None:
                        if not self.name_ld in self.radar.fields:
                            if self.verbose:
                                print 'Not finding LDR field, not using'
                            self.name_ld = None
                    else:
                        if self.verbose:
                            print 'Not provided LDR field, not using'
                    if self.name_kd is not None:
                        if not self.name_kd in self.radar.fields:
                            if self.verbose:
                                print 'Not finding KDP field, calculating'
                            kdp_flag = self.calculate_kdp()
                        else:
                            kdp_flag = True
                    else:
                        if self.verbose:
                            print 'Not provided KDP field, calculating'
                        kdp_flag = self.calculate_kdp()
                    return kdp_flag #All required variables present?
                else:
                    warnings.warn(self.name_rh+wstr)
                    return False
            else:
                warnings.warn(self.name_dr+wstr)
                return False
        else:
           warnings.warn(self.name_dz+wstr)
           return False

    @fn_timer
    def calculate_kdp(self):
        """
        Wrapper method for calculating KDP.
        """
        wstr = 'Missing differential phase and KDP fields, failing ...'
        if self.name_dp is not None:
            if self.name_dp in self.radar.fields:
                if self.kdp_method.upper() == 'CSU':
                    kdp = self.call_csu_kdp()
                self.name_kd = 'KDP_' + self.kdp_method
                self.add_field_to_radar_object(kdp, standard_name='KDP',
                                    field_name=self.name_kd, units='deg km-1',
                                    long_name='Specific Differential Phase')
            else:
                warnings.warn(wstr)
                return False
        else:
            warnings.warn(wstr)
            return False
        return True
    
    def call_csu_kdp(self):
        """
        Calls the csu_radartools.csu_kdp module to obtain KDP, FDP, and SDP.
        """
        dp = self.extract_unmasked_data(self.name_dp)
        dz = self.extract_unmasked_data(self.name_dz)
        kdp = np.zeros_like(dp) + self.bad
        fdp = kdp * 1.0
        sdp = kdp * 1.0
        rng = self.radar.range['data'] / RNG_MULT
        nrays, ngates = np.shape(dz)
        for i in xrange(nrays):
            if self.verbose:
                if i % 100 == 0:
                    print 'i =', i, 'of', nrays-1
            kdp[i,:],fdp[i,:],sdp[i,:] = csu_kdp.calc_kdp_bringi(dp=dp[i,:],
                        dz=dz[i,:], rng=rng, thsd=self.thresh_sdp, bad=self.bad)
        self.name_fdp = 'FDP_'+self.kdp_method
        self.add_field_to_radar_object(fdp, units='deg',
                        standard_name='Filtered Differential Phase',
                        field_name=self.name_fdp,
                        long_name='Filtered Differential Phase')
        self.name_sdp = 'SDP_'+self.kdp_method
        self.add_field_to_radar_object(sdp, units='deg',
                        standard_name='Std Dev Differential Phase',
                        field_name=self.name_sdp,
                        long_name='Standard Deviation of Differential Phase')
        return kdp

    def extract_unmasked_data(self, field, bad=None):
        """Extracts an unmasked field from the radar object."""
        var = self.radar.fields[field]['data']
        if hasattr(var, 'mask'):
            if bad is None:
                bad = self.bad
            var = var.filled(fill_value=bad)
        return var

    def get_sounding(self, sounding):
        """
        Ingests the sounding (either a skewt - i.e., UWYO - formatted file
        or a properly formatted dict).
        """
        if sounding is None:
            print 'No sounding provided'
            self.T_flag = False
        else:
            if isinstance(sounding, str):
                try:
                    snd = SkewT.Sounding(sounding)
                    self.snd_T = snd.data['temp']
                    self.snd_z = snd.data['hght']
                except:
                    print 'Sounding read fail'
                    self.T_flag = False
            else:
                try:
                    self.snd_T = sounding['T']
                    self.snd_z = sounding['z']
                except:
                    print 'Sounding in wrong data format'
                    self.T_flag = False
        self.interpolate_sounding_to_radar()
    
    @fn_timer
    def get_hid(self):
        """Calculate hydrometeror ID, add to radar object."""
        dz = self.radar.fields[self.name_dz]['data']
        dr = self.radar.fields[self.name_dr]['data']
        kd = self.radar.fields[self.name_kd]['data']
        rh = self.radar.fields[self.name_rh]['data']
        if self.name_ld is not None:
            ld = self.radar.fields[self.name_ld]['data']
        else:
            ld = None
        if not self.winter_flag:
            scores = csu_fhc.csu_fhc_summer(dz=dz, zdr=dr, rho=rh, kdp=kd,
                          ldr=ld, use_temp=self.T_flag, band=self.band,
                          method=self.fhc_method, T=self.radar_T,
                          verbose=self.verbose, temp_factor=self.T_factor,
                          weights=self.fhc_weights)
            fh = np.argmax(scores, axis=0) + 1
            self.add_field_to_radar_object(fh, field_name=self.name_fhc)
        else:
            print 'Winter HID not enabled yet, sorry!'

    @fn_timer
    def get_precip_rate(self, ice_flag=False, rain_method='hidro'):
        """Calculate rain rate, add to radar object."""
        dz = self.radar.fields[self.name_dz]['data']
        dr = self.radar.fields[self.name_dr]['data']
        kd = self.radar.fields[self.name_kd]['data']
        if not self.winter_flag:
            if rain_method == 'hidro':
                fhc = self.radar.fields[self.name_fhc]['data']
                rain, method = csu_blended_rain.csu_hidro_rain(dz=dz, zdr=dr,
                                                               kdp=kd, fhc=fhc)
            else:
                if not ice_flag:
                    rain, method = csu_blended_rain.calc_blended_rain(dz=dz,
                                                                 zdr=dr, kdp=kd)
                else:
                    rain, method, zdp, fi =\
                          csu_blended_rain.calc_blended_rain(dz=dz, zdr=dr,
                                                      kdp=kd, ice_flag=ice_flag)
                    self.add_field_to_radar_object(zdp, field_name='ZDP',
                                           units='dB',
                                           long_name='Difference Reflectivity',
                                        standard_name='Difference Reflectivity')
                    self.add_field_to_radar_object(fi, field_name='FI', units='',
                                                   long_name='Ice Fraction',
                                                   standard_name='Ice Fraction')
        else:
            print 'Winter precip not enabled yet, sorry!'
            return
        self.add_field_to_radar_object(rain, field_name='rain', units='mm h-1',
                                       long_name='Rainfall Rate',
                                        standard_name='Rainfall Rate')
        self.add_field_to_radar_object(method, field_name='method', units='',
                                       long_name='Rainfall Method',
                                       standard_name='Rainfall Method')

    @fn_timer
    def get_dsd(self):
        """Calculate DSD information, add to radar object."""
        dz = self.radar.fields[self.name_dz]['data']
        dr = self.radar.fields[self.name_dr]['data']
        kd = self.radar.fields[self.name_kd]['data']
        d0, Nw, mu = csu_dsd.calc_dsd(dz=dz, zdr=dr, kdp=kd, band=self.band,
                                      method='2009')
        self.add_field_to_radar_object(d0, field_name='D0', units='mm',
                                       long_name='Median Volume Diameter',
                                       standard_name='Median Volume Diameter')
        self.add_field_to_radar_object(Nw, field_name='NW', units='mm-1 m-3',
                                    long_name='Normalized Intercept Parameter',
                                standard_name='Normalized Intercept Parameter')
        self.add_field_to_radar_object(mu, field_name='MU', units=' ',
                                       long_name='Mu', standard_name='Mu')

    @fn_timer
    def get_liquid_and_frozen_mass(self):
        """Calculate liquid/ice mass, add to radar object."""
        mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(
                         self.radar.fields[self.name_dz]['data'],
                         self.radar.fields[self.name_dr]['data'],
                         self.radar_z/1000.0, T=self.radar_T)
        self.add_field_to_radar_object(mw, field_name='MW', units='g m-3',
                                       long_name='Liquid Water Mass',
                                       standard_name='Liquid Water Mass')
        self.add_field_to_radar_object(mi, field_name='MI', units='g m-3',
                                       long_name='Ice Water Mass',
                                       standard_name='Ice Water Mass')

    def add_field_to_radar_object(self, field, field_name='FH',
                                  units='unitless', long_name='Hydrometeor ID',
                                  standard_name='Hydrometeor ID'):
        """
        Adds a newly created field to the Py-ART radar object.
        """
        masked_field = np.ma.asanyarray(field)
        fill_value = self.bad
        if hasattr(self.radar.fields[self.name_dz]['data'], 'mask'):
            setattr(masked_field, 'mask',
                    self.radar.fields[self.name_dz]['data'].mask)
            fill_value = self.radar.fields[self.name_dz]['_FillValue']
        field_dict = {'data': masked_field,
                      'units': units,
                      'long_name': long_name,
                      'standard_name': standard_name,
                      '_FillValue': fill_value}
        self.radar.add_field(field_name, field_dict)

    def interpolate_sounding_to_radar(self):
        """Takes sounding data and interpolates it to every radar gate."""
        self.radar_z = get_z_from_radar(self.radar)
        self.radar_T = None
        self.check_sounding_for_montonic()
        if self.T_flag:
            shape = np.shape(self.radar_z)
            rad_z1d = self.radar_z.ravel()
            rad_T1d = np.interp(rad_z1d, self.snd_z, self.snd_T)
            if self.verbose:
                print 'Trying to get radar_T'
            self.radar_T = np.reshape(rad_T1d, shape)

    def check_sounding_for_montonic(self):
        """
        So the sounding interpolation doesn't fail, force the sounding to behave
        monotonically so that z always increases. This eliminates data from
        descending balloons.
        """
        dummy_z = []
        dummy_T = []
        if hasattr(self, 'snd_T'):
            if not self.snd_T.mask[0]: #May cause issue for specified soundings
                dummy_z.append(self.snd_z[0])
                dummy_T.append(self.snd_T[0])
            for i, height in enumerate(self.snd_z):
                if i > 0:
                    if self.snd_z[i] > self.snd_z[i-1] and not\
                       self.snd_T.mask[i]:
                        dummy_z.append(self.snd_z[i])
                        dummy_T.append(self.snd_T[i])
            self.snd_z = np.array(dummy_z)
            self.snd_T = np.array(dummy_T)

################################

class HidColors(object):

    """
    Experimental, untested class to help with colormaps/bars when plotting 
    hydrometeor ID data with Py-ART.
    """

    def __init__(self, winter=False):
        if not winter:
            self.hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange',
                               'LightPink', 'Cyan', 'DarkGray', 'Lime', 'Yellow',
                               'Red', 'Fuchsia']
            self.cmapmeth = colors.ListedColormap(self.hid_colors[0:6])
        self.cmaphid = colors.ListedColormap(self.hid_colors)

    def adjust_fhc_colorbar_for_pyart(self, cb):
        """Mods to make a hydrometeor ID colorbar"""
        cb.set_ticks(np.arange(1.4, 10, 0.9))
        cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                               'Wet Snow', 'Vertical Ice', 'LD Graupel',
                               'HD Graupel', 'Hail', 'Big Drops'])
        cb.ax.set_ylabel('')
        cb.ax.tick_params(length=0)
        return cb

    def adjust_meth_colorbar_for_pyart(self, cb):
        """Mods to make a rainfall method colorbar"""
        cb.set_ticks(np.arange(1.25, 5, 0.833))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z)',
                               'R(Zrain)'])
        cb.ax.set_ylabel('')
        cb.ax.tick_params(length=0)
        return cb

################################

def get_z_from_radar(radar):
    """Input radar object, return z from radar (km, 2D)"""
    azimuth_1D = radar.azimuth['data']
    elevation_1D = radar.elevation['data']
    srange_1D = radar.range['data']
    sr_2d, az_2d = np.meshgrid(srange_1D, azimuth_1D)
    el_2d = np.meshgrid(srange_1D, elevation_1D)[1]
    xx, yy, zz = radar_coords_to_cart(sr_2d/RNG_MULT, az_2d, el_2d)
    return zz + radar.altitude['data']

def check_kwargs(kwargs, default_kw):
    """
    Check user-provided kwargs against the defaults, and if some defaults are not
    provided by the user make sure they are provided to the function regardless.
    """
    for key in default_kw:
        if key not in kwargs:
            kwargs[key] = default_kw[key]
    return kwargs

#####################################














