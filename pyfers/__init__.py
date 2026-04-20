import os
import h5py
import numpy as np
import subprocess as sbp
import scipy.constants as constants
from lxml import etree as ET
from abc import ABC, abstractmethod

def wavelength_to_frequency(wavelength):
    return constants.c/wavelength

def frequency_to_wavelength(frequency):
    return constants.c/frequency

def time_to_range(time):
    return constants.c*time/2

def range_to_time(range):
    return 2*range/constants.c

def linear_to_dB(linear):
    return 10*np.log10(linear)

def dB_to_linear(decibels):
    return pow(10, decibels/10)

def achirp(period, sample_rate, bandwidth, init_freq=0, tau=0, phi=0):
    '''
    Generate an analytic baseband chirp.
    '''
    ns_chirp = int(np.ceil(period*sample_rate))
    t_chirp = np.linspace(-period/2, period/2, ns_chirp, endpoint=False)
    return np.exp(1.j*(np.pi*bandwidth/(2*max(t_chirp))*pow((t_chirp - tau), 2) + 2*np.pi*init_freq*(t_chirp - tau) + phi))

def write_hdf5(dataset, filename):
    '''
    Write IQ data to an HDF5 file.
    '''
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('/I/value', data=np.real(dataset))
    h5.create_dataset('/Q/value', data=np.imag(dataset))
    h5.close()


def read_hdf5(filename):
    '''
    Read IQ data from an HDF5 file.
    '''
    if (os.path.exists(filename) == False):
        print("HDF5 file not found. Please check the path.")
        exit()

    h5 = h5py.File(filename, 'r')

    dataset_list = list(h5.keys())

    # read attributes
    # attribute_list = h5[dataset_list[0]].attrs.keys()
    # for attr in attribute_list:
        # print(attr, h5[dataset_list[0]].attrs[attr])

    scale = np.float64(h5[dataset_list[0]].attrs['fullscale'])
    # rate = np.float64(h5[dataset_list[0]].attrs['rate'])
    # time = np.float64(h5[dataset_list[0]].attrs['time'])

    n_pulses = int(np.floor(np.size(dataset_list)/2))
    ns_pulse = int(np.size(h5[dataset_list[0]]))

    i_matrix = np.zeros((n_pulses, ns_pulse), dtype='float64')
    q_matrix = np.zeros((n_pulses, ns_pulse), dtype='float64')

    for i in range(0, n_pulses):
        i_matrix[i, :] = np.array(h5[dataset_list[2*i + 0]], dtype='float64')
        q_matrix[i, :] = np.array(h5[dataset_list[2*i + 1]], dtype='float64')

    dataset = np.array(i_matrix + 1j*q_matrix).astype('complex128')

    dataset *= scale

    return dataset

class Antenna:
    def __init__(self, name:str, type:str, gain:float, efficiency:float=1, az_beta=None, el_beta=None, az_gamma=None, el_gamma=None, az_factor=None, el_factor=None):
        """
        Parameters
        ----------
            name : str
                Unique antenna name.
            type : str
                Type of antenna. ['sinc', 'gaussian']
            gain : float
                Antenna gain (dBi).
            efficiency : float
                Antenna efficiency.
        """
        self._name = name
        self._type = type.lower()
        self._gain = dB_to_linear(gain)
        self._efficiency = efficiency
        self._theta = np.linspace(-np.pi, np.pi, 1001, endpoint=False)
        self._az_pattern = None
        self._el_pattern = None

        if self._type == 'sinc':
            if (az_beta is None) or (el_beta is None) or (az_gamma is None) or (el_gamma is None):
                print("ERROR: All beta and gamma parameters are required for sinc antenna.")
            self._az_beta = az_beta
            self._el_beta = el_beta
            self._az_gamma = az_gamma
            self._el_gamma = el_gamma
            self._az_pattern = self._gain * pow((np.sin(self._az_beta*self._theta)/(self._az_beta*self._theta)), self._az_gamma)
            self._el_pattern = self._gain * pow((np.sin(self._el_beta*self._theta)/(self._el_beta*self._theta)), self._el_gamma)
        elif self._type == 'gaussian':
            if (az_factor is None) or (el_factor is None):
                print("ERROR: All factor parameters are required for gaussian antenna.")
            self._az_factor = az_factor
            self._el_factor = el_factor
            self._az_pattern = self._gain * np.exp(-1*pow(self._theta, 2)*self._az_factor)
            self._el_pattern = self._gain * np.exp(-1*pow(self._theta, 2)*self._el_factor)
        else:
            print("ERROR: Unsupported antenna type.")

    @property
    def name(self):
        return self._name

    @property
    def efficiency(self):
        return self._efficiency

    @property
    def theta(self):
        return self._theta

    @property
    def az_pattern(self):
        return self._az_pattern

    @property
    def el_pattern(self):
        return self._el_pattern


class Waveform:
    def __init__(self, name:str, f_carrier:float, type:str, power:float, f_sample:float, bandwidth:float=None, t_pulse:float=None):
        """
        Parameters
        ----------
            name : str
                Unique name for the waveform.
            f_carrier : float
                Carrier frequency (Hz).
            type : str
                Type of waveform. ['pulse', 'cw']
            power : float
                Constant power of the pulse (W).
            f_sample : float
                Rate at which the waveform is sampled (Hz).
            bandwidth : float
                Bandwidth of the waveform (Hz).
            t_pulse : float
                Length of the pulse (s).
        """
        self._name = name
        self._f_carrier = f_carrier
        self._type = type
        self._power = power
        self._f_sample = f_sample
        self._bandwidth = bandwidth
        self._t_pulse = t_pulse
        self._samples = None
        self._n_samples = None

        if (self._f_sample < self._bandwidth):
            print("WARNING: Sample rate is insufficient. Nyquist is violated.")

        if type == 'pulse':
            self._samples = achirp(period=self._t_pulse, sample_rate=self._f_sample, bandwidth=self._bandwidth, tau=0)

    @property
    def name(self):
        return self._name

    @property
    def f_carrier(self):
        return self._f_carrier

    @property
    def wavelength(self):
        return frequency_to_wavelength(self._f_carrier)

    @property
    def power(self):
        return self._power

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def t_pulse(self):
        return self._t_pulse

    @property
    def samples(self):
        return self._samples

    @property
    def n_samples(self):
        return len(self._samples)

    @property
    def time_bandwidth(self):
        """
        Time-bandwidth product.
        """
        if type == 'pulse':
            return self._bandwidth * self._t_pulse

class Clock:
    def __init__(self, name:str, frequency:float, f_offset:float=0, random_f_offset:float=0, p_offset:float=0, random_p_offset:float=0):
        """
        Parameters
        ----------
            name : str
                Unique name of the clock.
            frequency : float
                Frequency of the clock (Hz).
            f_offset : float
                Constant frequency offset (Hz).
            random_f_offset : float
                Standard deviation of random frequency offset (Hz).
            p_offset : float
                Constant phase offset (rad).
            random_p_offset : float
                Standard deviation of random phase offset (rad).
        """
        self._name = name
        self._frequency = frequency
        self._f_offset = f_offset
        self._p_offset = p_offset
        self._random_f_offset = random_f_offset
        self._random_p_offset = random_p_offset

    @property
    def name(self):
        return self._name

    @property
    def frequency(self):
        return self._frequency

    @property
    def f_offset(self):
        return self._f_offset

    @property
    def p_offset(self):
        return self._p_offset

    @property
    def random_f_offset(self):
        return self._random_f_offset

    @property
    def random_p_offset(self):
        return self._random_p_offset

class Transmitter:
    def __init__(self, name:str, antenna:Antenna, waveform:Waveform, clock:Clock, f_prf:float):
        """
        Parameters
        ----------
            name : str
                Unique name for transmitter.
            antenna : Antenna
                Antenna used for transmission.
            waveform : Waveform
                Waveform to transmit.
            clock : Clock
                Timing source for the transmitter.
            f_prf : float
                Pulse repetition frequency (Hz).
        """
        self._name = name
        self._antenna = antenna
        self._waveform = waveform
        self._clock = clock
        self._f_prf = f_prf

    @property
    def name(self):
        return self._name

    @property
    def f_prf(self):
        return self._f_prf

    @property
    def t_pri(self):
        return 1/self._f_prf

    @property
    def duty_cycle(self):
        return self._waveform._t_pulse / self.t_pri

    @property
    def p_average(self):
        return self._waveform.power * self.duty_cycle

class Receiver:
    def __init__(self, name:str, antenna:Antenna, clock:Clock, f_prf:float, gate:float=None, noise_temp:float=290):
        """
        Parameters
        ----------
            name : str
                Unique name for the receiver.
            antenna : Antenna
                Antenna used for transmission.
            clock : Clock
                Timing source for the receiver.
            f_prf : float
                The receiver pulse repetition frequency (Hz).
            gate : float
                The extent of range to digitise (m). Starts from zero metres. If None, the maximum range gate is used.
            noise_temp : float
                The noise temperature of the receiver (K).
        """
        self._name = name
        self._antenna = antenna
        self._clock = clock
        self._f_prf = f_prf
        self._gate = gate
        self._noise_temp = noise_temp

        if self._gate is None:
            self._gate = time_to_range(1/self._f_prf)
        else:
            if (self._gate < 0) or (self._gate > time_to_range(1/self._f_prf)):
                print("ERROR: Invalid range gate.")

    @property
    def name(self):
        return self._name

    @property
    def f_prf(self):
        return self._f_prf

    @property
    def noise_temp(self):
        return self._noise_temp

    @property
    def noise_density(self):
        """
        Calculate the noise density (W/Hz).
        """
        return constants.Boltzmann * self._noise_temp

    @property
    def noise_power(self):
        """
        Calculate the receiver noise power (W). Receiver bandwidth is determined by the clock frequency.
        """
        return self.noise_density * self._clock._frequency

    @property
    def gate(self):
        """
        Range gate of the receiver (m).
        """
        return self._gate

    @property
    def ns_gate(self):
        """
        Return the number of samples in the range gate. The ADC rate is determined by the clock frequency.
        """
        return int(np.ceil(self._clock._frequency * range_to_time(self._gate)))

class Platform(ABC):
    def __init__(self, name:str, interpolation:str):
        """
        Parameters
        ----------
            name : str
                Unique name for the platform.
            interpolation : str
                Type of interpolation to employ ['static', 'linear', 'cubic'].
        """

        self._name = name
        self._interpolation = interpolation
        self._t = 0
        self._x = 0
        self._y = 0
        self._z = 0
        self._az = 0
        self._el = 0

        #TODO input validation for interpolation

    @property
    def name(self):
        return self._name

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def t(self):
        """
        Sampled time axis.
        """
        return self._t

    @property
    def x(self):
        """
        Sampled motion in x-axis (along-track).
        """
        return self._x

    @property
    def y(self):
        """
        Sampled motion in y-axis (cross-track).
        """
        return self._y

    @property
    def z(self):
        """
        Sampled motion in z-axis (altitude).
        """
        return self._z

    @property
    def az(self):
        """
        Sampled motion in azimuth (squint).
        """
        return self._az

    @property
    def el(self):
        """
        Sampled motion in elevation (depression).
        """
        return self._el

    @property
    @abstractmethod
    def n_samples(self):
        """Must be implemented by all subclasses"""
        pass

    @property
    def position_waypoints(self):
        waypoints = []
        for i in range(self.n_samples):
            waypoints.append(PositionWaypoint(self.x[i], self.y[i], self.z[i], self.t[i]))
        return waypoints

    @property
    def rotation_waypoints(self):
        waypoints = []
        for i in range(self.n_samples):
            waypoints.append(RotationWaypoint(self.az[i], self.el[i], self.t[i]))
        return waypoints

class StaticPlatform(Platform):
    def __init__(self, name:str, x:float=0, y:float=0, z:float=0, az:float=0, el:float=0):
        """
        Parameters
        ----------
            name : str
                Unique name for the platform.
            x : float
                Position of platform in x-axis (m).
            y : float
                Position of platform in y-axis (m).
            z : float
                Position of platform in z-axis (m).
            az : float
                Orientation of platform in azimuth (rad).
            el : float
                Orientation of platform in elevation (rad).
        """
        super().__init__(name, 'static')

        self._t = [0]
        self._x = [x]
        self._y = [y]
        self._z = [z]
        self._az = [az]
        self._el = [el]

    @property
    def n_samples(self):
        return 1

class DynamicPlatform(Platform):
    def __init__(self, name:str, d:float, fs:float, interpolation:str='linear'):
        """
        Parameters
        ----------
            name : str
                Unique name for the platform.
            d : float
                Duration of platform motion (s).
            fs : float
                Sample rate of platform motion (Hz).
            interpolation : str
                Type of interpolation to employ ['linear', 'cubic'].
        """
        super().__init__(name, interpolation)

        self._duration = d
        self._fs = fs
        self._t = np.linspace(0, self._duration, self.n_samples, endpoint=False)

    @property
    def n_samples(self):
        return int(np.ceil(self._duration * self._fs))

    def add_motion(self, type:str, axis:str, constant:float=None, gradient:float=None, amplitude:float=None, frequency:float=None, phase:float=None):
        """
        Parameters
        ----------
            type : str
                Type of motion to apply.
                ['constant', 'linear', 'sinusoid']
            axis : str
                Axis to which the motion is applied.
                ['x', 'y', 'z', 'az', 'el']
            amplitude : float
                Amplitude of the sinusoid (m).
            frequency : float
                Frequency of the sinusoid (Hz). Must be less than fs/2.
            phase : float
                Phase of the sinusoid (rad).
        """

        if type == 'constant':
            if constant is None:
                print("ERROR: Constant parameter is required.")
                return
            motion = constant * np.ones(self.n_samples)
        elif type == 'linear':
            if gradient is None:
                print("ERROR: Gradient parameter is required.")
                return
            motion = gradient * self._t
        elif type == 'sinusoid':
            if (amplitude is None) or (frequency is None) or (phase is None):
                print("ERROR: Amplitude, frequency, and phase parameters are required.")
                return
            if frequency > self._fs/2:
                print("ERROR: Frequency violates Nyquist (< fs/2).")
                return

            motion = amplitude * np.sin(2*np.pi*frequency*self._t + phase)
        else:
            print("ERROR: Unsupported motion type, use ['constant', 'linear'].")
            return

        if axis.lower() == 'x':
            self._x += motion
        elif axis.lower() == 'y':
            self._y += motion
        elif axis.lower() == 'z':
            self._z += motion
        elif axis.lower() == 'az':
            self._az += motion
        elif axis.lower() == 'el':
            self._el += motion
        else:
            print("ERROR: Unknown axis, use ['x', 'y', 'z', 'az', 'el'].")


class Target:
    def __init__(self, name:str, rcs:float, platform:Platform, model:str='constant', pattern:str='isotropic'):
        self._name = name
        self._rcs = rcs
        self._platform = platform
        self._model = model
        self._pattern = pattern

        # TODO input validation for model and pattern

    @property
    def name(self):
        return self._name

    @property
    def rcs(self):
        return self._rcs

    @property
    def platform(self):
        return self._platform

    @property
    def model(self):
        return self._model

    @property
    def pattern(self):
        return self._pattern


class PositionWaypoint:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t


class RotationWaypoint:
    def __init__(self, az, el, t):
        self.az = az
        self.el = el
        self.t = t


class AntennaXML:
    def __init__(self, xml_filename, unit:str='rad', format:str='linear', symmetry:str='none'):
        """
        AntennaXML constructor.

        Parameters
        ----------
            unit : str
                Unit of the angle.
                ['rad', 'deg']
            format : str
                Format of the angle.
                ['linear', 'dBi']
            symmetry : str
                Symmetry of the antenna.
                ['mirrored', 'none']

        """
        self.filename = xml_filename
        self.root = ET.Element('antenna')

        if unit not in ['rad', 'deg']:
            print("ERROR: Unsupported unit, use ['rad', 'deg'].")
            return

        if format not in ['linear', 'dBi']:
            print("ERROR: Unsupported format, use ['linear', 'dBi'].")
            return

        if symmetry not in ['mirrored', 'none']:
            print("ERROR: Unsupported symmetry, use ['mirrored', 'none'].")
            return

        self.azimuth_element = ET.SubElement(self.root, 'azimuth')
        self.azimuth_element.set('unit', unit)
        self.azimuth_element.set('format', format)
        self.azimuth_element.set('symmetry', symmetry)

        self.elevation_element = ET.SubElement(self.root, 'elevation')
        self.elevation_element.set('unit', unit)
        self.elevation_element.set('format', format)
        self.elevation_element.set('symmetry', symmetry)

        self.tree = ET.ElementTree(self.root)

    def add_gainsample(self, plane, angle, gain):
        gainsample = ET.SubElement(plane, "gainsample")

        a = ET.SubElement(gainsample, "angle")
        a.text = str(angle)

        g = ET.SubElement(gainsample, "gain")
        g.text = str(gain)

    def write_xml(self):
        self.tree.write(self.filename,
            pretty_print=True,
            encoding="utf-8",
            xml_declaration=True)


class Simulation:
    def __init__(self, name:str, filename:str):
        """
        Simulation constructor.

        Parameters
        ----------
            name : str
                Unique name for the simulation.
            filename : str
                Filename for the .fersxml simulation definition.
        """
        self.filename = os.path.abspath(filename)
        self.root = ET.Element('simulation')
        self.root.set('name', name)
        self.tree = ET.ElementTree(self.root)

    def add_parameters(self, t_start, t_end, sim_rate, bits, over_sample=1):
        parameters = ET.SubElement(self.root, 'parameters')

        starttime = ET.SubElement(parameters, 'starttime')
        starttime.text = str(t_start)

        endtime = ET.SubElement(parameters, 'endtime')
        endtime.text = str(t_end)

        rate = ET.SubElement(parameters, 'rate')
        rate.text = str(sim_rate)

        light_speed = ET.SubElement(parameters, 'c')
        light_speed.text = str(constants.c)

        # sim_sample_rate = ET.SubElement(parameters, 'simSamplingRate')
        # sim_sample_rate.text = str(0)

        # seed = ET.SubElement(parameters, 'randomseed')
        # seed.text = str(0)

        adc_bits = ET.SubElement(parameters, 'adc_bits')
        adc_bits.text = str(bits)

        oversample = ET.SubElement(parameters, 'oversample')
        oversample.text = str(over_sample)

        # # Geodetic Origin for the simulation coordinate system (used for ENU frame)
        # origin = ET.SubElement(parameters, 'origin')
        # origin.set('latitude', '1')
        # origin.set('longitude', '2')
        # origin.set('altitude', '3')

        # # Coordinate System for the simulation input coordinates
        # coordinatesystem = ET.SubElement(parameters, 'coordinatesystem')
        # coordinatesystem.set('frame', 'ENU') # ENU, UTM, ECEF
        # coordinatesystem.set('zone', '32')
        # coordinatesystem.set('hemisphere', 'N') # N, S

    def add_waveform(self, waveform:Waveform, filename:str):
        """
        Add a Waveform instance to the fersxml definition.

        Parameters
        ----------
            waveform : Waveform
                Waveform instance to use as input.
            filename : str
                Filename to store waveform for FERS input.
        """
        filename = os.path.abspath(filename)

        write_hdf5(waveform.samples, filename)

        waveform_element = ET.SubElement(self.root, 'waveform')
        waveform_element.set('name', waveform.name)

        power = ET.SubElement(waveform_element, 'power')
        power.text = str(waveform.power)

        carrier = ET.SubElement(waveform_element, 'carrier_frequency')
        carrier.text = str(waveform.f_carrier)

        pulsed_from_file = ET.SubElement(waveform_element, 'pulsed_from_file')
        pulsed_from_file.set('filename', filename)

    def add_clock(self, clock:Clock, synconpulse='false'):
        timing_element = ET.SubElement(self.root, 'timing')
        timing_element.set('name', clock.name)
        timing_element.set('synconpulse', synconpulse)

        freq = ET.SubElement(timing_element, 'frequency')
        freq.text = str(clock.frequency)

        freq_offset = ET.SubElement(timing_element, 'freq_offset')
        freq_offset.text = str(clock.f_offset)

        random_freq_offset = ET.SubElement(timing_element, 'random_freq_offset_stdev')
        random_freq_offset.text = str(clock.random_f_offset)

        phase_offset = ET.SubElement(timing_element, 'phase_offset')
        phase_offset.text = str(clock.p_offset)

        random_phase_offset = ET.SubElement(timing_element, 'random_phase_offset_stdev')
        random_phase_offset.text = str(clock.random_p_offset)

        # add_noise(timing_element, -2, 1e-6)
        # add_noise(timing_element, -1, 1e-6)
        # add_noise(timing_element, 0, 1e-6)
        # add_noise(timing_element, 1, 1e-6)
        # add_noise(timing_element, 2, 1e-6)

    def add_antenna(self, antenna:Antenna, filename:str):
        """
        Add an Antenna instance to the fersxml definition.

        Parameters
        ----------
            antenna : Antenna
                Antenna instance to use as input.
            filename : str
                Filename to store antenna XML for FERS input.
        """
        filename = os.path.abspath(filename)

        # generate an antenna xml file
        fers_antenna = AntennaXML(filename, unit='rad', format='linear', symmetry='none')

        for i, angle in enumerate(antenna.theta):
            fers_antenna.add_gainsample(fers_antenna.azimuth_element, angle, antenna.az_pattern[i])
            fers_antenna.add_gainsample(fers_antenna.elevation_element, angle, antenna.el_pattern[i])

        fers_antenna.write_xml()

        # currently only xml definitions are supported
        pattern='xml'

        antenna_element = ET.SubElement(self.root, 'antenna')
        antenna_element.set('name', antenna.name)
        antenna_element.set('pattern', pattern)

        if (pattern == "xml"):
            antenna_element.set('filename', filename)

        # if (pattern == "parabolic"):
        #     diameter = ET.SubElement(antenna_element, 'diameter')
        #     diameter.text = str(d)

        # if (pattern == "sinc"):
        #     alpha = ET.SubElement(antenna_element, 'alpha')
        #     alpha.text = str(a)
        #     beta = ET.SubElement(antenna_element, 'beta')
        #     beta.text = str(b)
        #     gamma = ET.SubElement(antenna_element, 'gamma')
        #     gamma.text = str(g)

        # if (pattern == "gaussian"):
        #     az = ET.SubElement(antenna_element, 'azscale')
        #     az.text = str(azscale)
        #     el = ET.SubElement(antenna_element, 'elscale')
        #     el.text = str(elscale)

        efficiency = ET.SubElement(antenna_element, 'efficiency')
        efficiency.text = str(antenna.efficiency)

    def add_monostatic(self, platform:Platform, transmitter:Transmitter, receiver:Receiver, antenna:Antenna, waveform:Waveform, clock:Clock, window_skip=0, nodirect='false', nopropagationloss='false'):
        platform_element = self._add_platform(platform)

        monostatic = ET.SubElement(platform_element, 'monostatic')
        monostatic.set('name', receiver.name)
        monostatic.set('antenna', antenna.name)
        monostatic.set('waveform', waveform.name)
        monostatic.set('timing', clock.name)
        monostatic.set('nodirect', nodirect)
        monostatic.set('nopropagationloss', nopropagationloss)

        # TODO add cw mode
        mode = ET.SubElement(monostatic, 'pulsed_mode')

        rx_prf = ET.SubElement(mode, 'prf')
        rx_prf.text = str(transmitter.f_prf)

        skip = ET.SubElement(mode, 'window_skip')
        skip.text = str(window_skip)

        window = ET.SubElement(mode, 'window_length')
        window.text = str(range_to_time(receiver.gate)) # time period

        noise = ET.SubElement(monostatic, 'noise_temp')
        noise.text = str(receiver.noise_temp)

    def add_target(self, target:Target):
        platform = self._add_platform(target.platform)

        target_element = ET.SubElement(platform, 'target')
        target_element.set('name', target.name)

        t_rcs = ET.SubElement(target_element, 'rcs')
        t_rcs.set('type', target.pattern)

        t_rcs_v = ET.SubElement(t_rcs, 'value')
        t_rcs_v.text = str(target.rcs)

        model = ET.SubElement(target_element, 'model')
        model.set('type', target.model)

    def write_xml(self):
        self.tree.write(self.filename,
                        pretty_print=True,
                        encoding="utf-8",
                        xml_declaration=True,
                        doctype='<!DOCTYPE simulation SYSTEM "fers-xml.dtd">')

    def run(self):
        try:
            sbp.run(['fers-cli', self.filename])
        except:
            print('ERROR: failed to launch - check that FERS is installed correctly.')
            exit(1)

    def add_transmitter(self, platform:Platform, transmitter:Transmitter, antenna:Antenna, waveform:Waveform, clock:Clock):
        platform_element = self._add_platform(platform)

        transmitter_element = ET.SubElement(platform_element, 'transmitter')
        transmitter_element.set('name', transmitter.name)
        transmitter_element.set('waveform', waveform.name)
        transmitter_element.set('antenna', antenna.name)
        transmitter_element.set('timing', clock.name)

        # TODO add cw_mode
        mode = ET.SubElement(transmitter_element, 'pulsed_mode')

        tx_prf = ET.SubElement(mode, 'prf')
        tx_prf.text = str(transmitter.f_prf)

    def add_receiver(self, platform:Platform, receiver:Receiver, antenna:Antenna, clock:Clock, window_skip=0, nodirect='false', nopropagationloss='false'):
        platform_element = self._add_platform(platform)

        receiver_element = ET.SubElement(platform_element, 'receiver')
        receiver_element.set('name', receiver.name)
        receiver_element.set('antenna', antenna.name)
        receiver_element.set('timing', clock.name)
        receiver_element.set('nodirect', nodirect)
        receiver_element.set('nopropagationloss', nopropagationloss)

        # TODO add cw_mode
        mode = ET.SubElement(receiver_element, 'pulsed_mode')

        rx_prf = ET.SubElement(mode, 'prf')
        rx_prf.text = str(receiver.f_prf)

        skip = ET.SubElement(mode, 'window_skip')
        skip.text = str(window_skip)

        window = ET.SubElement(mode, 'window_length')
        window.text = str(range_to_time(receiver.gate)) # time period

        noise = ET.SubElement(receiver_element, 'noise_temp')
        noise.text = str(receiver.noise_temp)

    def _add_platform(self, platform:Platform) -> ET.SubElement:
        # add platform parent
        platform_element = ET.SubElement(self.root, 'platform')
        platform_element.set('name', platform.name)

        # add motionpath child
        motionpath = ET.SubElement(platform_element, 'motionpath')
        motionpath.set('interpolation', platform.interpolation)

        for position_waypoint in platform.position_waypoints:
            self._add_positionwaypoint(motionpath, position_waypoint)

        # add rotationpath child
        rotationpath = ET.SubElement(platform_element, 'rotationpath')
        rotationpath.set('interpolation', platform.interpolation)

        for rotation_waypoint in platform.rotation_waypoints:
            self._add_rotationwaypoint(rotationpath, rotation_waypoint)

        return platform_element


    def _add_positionwaypoint(self, path, waypoint:PositionWaypoint):
        point = ET.SubElement(path, 'positionwaypoint')

        t_x = ET.SubElement(point, 'x')
        t_x.text = str(waypoint.x)

        t_y = ET.SubElement(point, 'y')
        t_y.text = str(waypoint.y)

        t_a = ET.SubElement(point, 'altitude')
        t_a.text = str(waypoint.z)

        t_t = ET.SubElement(point, 'time')
        t_t.text = str(waypoint.t)

    def _add_rotationwaypoint(self, path, waypoint:RotationWaypoint):
        point = ET.SubElement(path, 'rotationwaypoint')

        t_az = ET.SubElement(point, 'azimuth')
        t_az.text = str(waypoint.az)

        t_el = ET.SubElement(point, 'elevation')
        t_el.text = str(waypoint.el)

        t_t = ET.SubElement(point, 'time')
        t_t.text = str(waypoint.t)

    def _add_fixedrotation(self, platform, s_az=0, az_rate=0, s_el=0, el_rate=0):
        rotation = ET.SubElement(platform, 'fixedrotation')

        t_s_az = ET.SubElement(rotation, 'startazimuth')
        t_s_az.text = str(s_az)

        t_s_el = ET.SubElement(rotation, 'startelevation')
        t_s_el.text = str(s_el)

        t_az_r = ET.SubElement(rotation, 'azimuthrate')
        t_az_r.text = str(az_rate)

        t_el_r = ET.SubElement(rotation, 'elevationrate')
        t_el_r.text = str(el_rate)


    def _add_noise(self, clock, alpha, weight):
        noise_entry = ET.SubElement(clock, 'noise_entry')

        t_alpha = ET.SubElement(noise_entry, 'alpha')
        t_alpha.text = str(alpha)

        t_weight = ET.SubElement(noise_entry, 'weight')
        t_weight.text = str(weight)
