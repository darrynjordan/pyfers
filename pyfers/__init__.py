import os
import h5py
import numpy as np
import subprocess as sbp
import scipy.constants as constants
from lxml import etree as ET

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

class Transmitter:
    def __init__(self, antenna:Antenna, waveform:Waveform, clock:Clock, f_prf:float):
        """
        Parameters
        ----------
            antenna : Antenna
                Antenna used for transmission.
            waveform : Waveform
                Waveform to transmit.
            clock : Clock
                Timing source for the transmitter.
            f_prf : float
                Pulse repetition frequency (Hz).
        """
        self._antenna = antenna
        self._waveform = waveform
        self._clock = clock
        self._f_prf = f_prf

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
        return self._waveform.power * self._duty_cycle

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

class Platform:
    """
    Along-Track (m):    X-Axis
    Cross-Track (m):    Y-Axis
    Altitude (m):       Z-Axis
    Azimuth (deg):      X-Y Plane
    Elevation (deg):    X-Z Plane
    """
    def __init__(self, d:float, fs:float, altitude:float, velocity:float, depression:float=0, squint:float=0):
        """
        Parameters
        ----------
            d : float
                Duration of platform motion (s).
            fs : float
                Sample rate of platform motion (Hz).
            altitude : float
                Mean altitude of the platform (m).
            velocity : float
                Mean velocity of the platform (m/s).
            depression : float
                Mean depression angle of the sensor (deg).
            squint : float
                Mean squint angle of the platform (deg).
        """
        self.duration = d
        self.fs = fs
        self.altitude = altitude
        self.velocity = velocity
        self.depression = depression
        self.squint = squint

        # sampled time axis
        self._t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

        # arrays that hold the ideal motion
        self._x = self.velocity * self._t
        self._y = np.zeros(self.n_samples)
        self._z = self.altitude * np.ones(self.n_samples)
        self._az = self.squint * np.ones(self.n_samples)
        self._el = self.depression * np.ones(self.n_samples)

        # arrays that hold the deviations from ideal motion
        self._x_noise = 0
        self._y_noise = 0
        self._z_noise = 0
        self._az_noise = 0
        self._el_noise = 0

    def add_sinusoidal_noise(self, axis:str, amplitude:float, frequency:float, phase:float):
        """
        Parameters
        ----------
            axis : str
                Axis to which the sinusoidal noise is applied.
                ['x', 'y', 'z', 'az', 'el']
            amplitude : float
                Amplitude of the sinusoid (m).
            frequency : float
                Frequency of the sinusoid (Hz). Must be less than fs/2.
            phase : float
                Phase of the sinusoid (rad).
        """
        if frequency > self.fs/2:
            print("Error: The frequency of sinusoidal motion noise violates Nyquist. No noise applied.")
            return

        noise = amplitude * np.sin(2*np.pi*frequency*self._t + phase)

        if axis.lower() == 'x':
            self._x_noise += noise
        elif axis.lower() == 'y':
            self._y_noise += noise
        elif axis.lower() == 'z':
            self._z_noise += noise
        elif axis.lower() == 'az':
            self._az_noise += noise
        elif axis.lower() == 'el':
            self._el_noise += noise
        else:
            print("Error: Unknown axis, use ['x', 'y', 'z', 'az', 'el'].")

    @property
    def n_samples(self):
        return int(np.ceil(self.duration * self.fs))

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
        return self._x + self._x_noise

    @property
    def y(self):
        """
        Sampled motion in y-axis (cross-track).
        """
        return self._y + self._y_noise

    @property
    def z(self):
        """
        Sampled motion in z-axis (altitude).
        """
        return self._z + self._z_noise

    @property
    def az(self):
        """
        Sampled motion in azimuth (squint).
        """
        return self._az + self._az_noise

    @property
    def el(self):
        """
        Sampled motion in elevation (depression).
        """
        return self._el + self._el_noise

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


class FersTarget:
    def __init__(self, name, rcs, position_waypoints):
        self.name = name
        self.rcs = rcs
        self.position_waypoints = position_waypoints


class FersPositionWaypoint:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t


class FersRotationWaypoint:
    def __init__(self, az, el, t):
        self.az = az
        self.el = el
        self.t = t


class FersAntennaXML:
    def __init__(self, xml_filename):
        """
        FersAntennaXML constructor.
        """
        self.filename = xml_filename
        self.root = ET.Element('antenna')
        self.elevation = ET.SubElement(self.root, 'elevation')
        self.azimuth = ET.SubElement(self.root, 'azimuth')
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


class FersXMLGenerator:
    def __init__(self, sim_name:str, xml_filename:str):
        """
        FersXMLGenerator constructor.
        """
        self.filename = xml_filename
        self.simulation = ET.Element('simulation')
        self.simulation.set('name', sim_name)
        self.tree = ET.ElementTree(self.simulation)

    def add_parameters(self, t_start, t_end, sim_rate, bits, over_sample=1):
        parameters = ET.SubElement(self.simulation, 'parameters')

        starttime = ET.SubElement(parameters, 'starttime')
        starttime.text = str(t_start)

        endtime = ET.SubElement(parameters, 'endtime')
        endtime.text = str(t_end)

        rate = ET.SubElement(parameters, 'rate')
        rate.text = str(sim_rate)

        light_speed = ET.SubElement(parameters, 'c')
        light_speed.text = str(299792458)

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

    def add_waveform(self, name, waveform_file, power_watts, carrier_frequency):
        waveform = ET.SubElement(self.simulation, 'waveform')
        waveform.set('name', name)

        power = ET.SubElement(waveform, 'power')
        power.text = str(power_watts)

        carrier = ET.SubElement(waveform, 'carrier_frequency')
        carrier.text = str(carrier_frequency)

        pulsed_from_file = ET.SubElement(waveform, 'pulsed_from_file')
        pulsed_from_file.set('filename', waveform_file)

    def from_waveform(self, waveform:Waveform, filename:str):
        """
        Add a Waveform instance to the fersxml definition.

        Parameters
        ----------
            waveform : Waveform
                Waveform instance to use as input.
            filename : str
                Filename to store waveform for FERS input.
        """
        write_hdf5(waveform.samples, filename)
        self.add_waveform(waveform.name, waveform_file=filename, power_watts=waveform.power, carrier_frequency=waveform.f_carrier)

    def add_clock(self, name, frequency, f_offset=0, random_f_offset=0, p_offset=0, random_p_offset=0, synconpulse='false'):
        timing = ET.SubElement(self.simulation, 'timing')
        timing.set('name', name)
        timing.set('synconpulse', synconpulse)

        freq = ET.SubElement(timing, 'frequency')
        freq.text = str(frequency)

        freq_offset = ET.SubElement(timing, 'freq_offset')
        freq_offset.text = str(f_offset)

        random_freq_offset = ET.SubElement(timing, 'random_freq_offset_stdev')
        random_freq_offset.text = str(random_f_offset)

        phase_offset = ET.SubElement(timing, 'phase_offset')
        phase_offset.text = str(p_offset)

        random_phase_offset = ET.SubElement(timing, 'random_phase_offset_stdev')
        random_phase_offset.text = str(random_p_offset)

        # add_noise(timing, -2, 1e-6)
        # add_noise(timing, -1, 1e-6)
        # add_noise(timing, 0, 1e-6)
        # add_noise(timing, 1, 1e-6)
        # add_noise(timing, 2, 1e-6)

    def from_clock(self, clock:Clock, synconpulse='false'):
        self.add_clock(name=clock.name, frequency=clock.frequency, synconpulse=synconpulse)

    def add_antenna(self, name, pattern, eff=1, a=1, b=2, g=5, d=1, azscale=1, elscale=1, filename=None):
        antenna = ET.SubElement(self.simulation, 'antenna')
        antenna.set('name', name)
        antenna.set('pattern', pattern)

        if (pattern == "xml"):
            antenna.set('filename', filename)

        if (pattern == "parabolic"):
            diameter = ET.SubElement(antenna, 'diameter')
            diameter.text = str(d)

        if (pattern == "sinc"):
            alpha = ET.SubElement(antenna, 'alpha')
            alpha.text = str(a)

            beta = ET.SubElement(antenna, 'beta')
            beta.text = str(b)

            gamma = ET.SubElement(antenna, 'gamma')
            gamma.text = str(g)

        if (pattern == "gaussian"):
            az = ET.SubElement(antenna, 'azscale')
            az.text = str(azscale)

            el = ET.SubElement(antenna, 'elscale')
            el.text = str(elscale)

        efficiency = ET.SubElement(antenna, 'efficiency')
        efficiency.text = str(eff)

    def from_antenna(self, antenna:Antenna, filename:str):
        """
        Add an Antenna instance to the fersxml definition.

        Parameters
        ----------
            antenna : Antenna
                Antenna instance to use as input.
            filename : str
                Filename to store antenna XML for FERS input.
        """
        # generate an antenna xml file
        fers_antenna = FersAntennaXML(filename)

        # FERS only accepts 0 to pi, assumes symmetry
        # angles in radians, gain is linear
        for i, angle in enumerate(antenna.theta):
            if (angle >= 0) and (angle <= np.pi):
                fers_antenna.add_gainsample(fers_antenna.azimuth, angle, antenna.az_pattern[i])
                fers_antenna.add_gainsample(fers_antenna.elevation, angle, antenna.el_pattern[i])

        fers_antenna.write_xml()

        self.add_antenna(antenna.name, pattern='xml', eff=antenna.efficiency, filename=filename)

    def add_monostatic_radar(self, antenna, timing, prf, waveform, position_waypoints, rotation_waypoints, window_length, noise_temp=290, window_skip=0, nodirect='false', nopropagationloss='false', interp='linear'):
        platform = self._add_platform('radar_platform', self.simulation)
        self._add_motionpath(platform, position_waypoints, interp)
        self._add_rotationpath(platform, rotation_waypoints, interp)
        self._add_monostatic(platform, 'receiver', antenna, waveform, timing, prf, window_length, noise_temp, window_skip, nodirect, nopropagationloss)

    def add_target(self, fers_target: FersTarget, interp='linear'):
        platform = self._add_platform('target_platform_' + fers_target.name, self.simulation)
        self._add_motionpath(platform, fers_target.position_waypoints, interp)
        self._add_fixedrotation(platform)

        target = ET.SubElement(platform, 'target')
        target.set('name', fers_target.name)

        t_rcs = ET.SubElement(target, 'rcs')
        t_rcs.set('type', 'isotropic')

        t_rcs_v = ET.SubElement(t_rcs, 'value')
        t_rcs_v.text = str(fers_target.rcs)

        model = ET.SubElement(target, 'model')
        model.set('type', "constant")

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

    def _add_monostatic(self, platform, name, antenna, waveform, timing, prf, window_length, noise_temp=290, window_skip=0, nodirect='false', nopropagationloss='false'):
        monostatic = ET.SubElement(platform, 'monostatic')
        monostatic.set('name', name)
        monostatic.set('antenna', antenna)
        monostatic.set('waveform', waveform)
        monostatic.set('timing', timing)
        monostatic.set('nodirect', nodirect)
        monostatic.set('nopropagationloss', nopropagationloss)

        # TODO add cw mode
        mode = ET.SubElement(monostatic, 'pulsed_mode')

        rx_prf = ET.SubElement(mode, 'prf')
        rx_prf.text = str(prf)

        skip = ET.SubElement(mode, 'window_skip')
        skip.text = str(window_skip)

        window = ET.SubElement(mode, 'window_length')
        window.text = str(window_length)

        noise = ET.SubElement(monostatic, 'noise_temp')
        noise.text = str(noise_temp)

    def _add_transmitter(self, platform, name, tx_type, antenna, pulse, timing, prf):
        transmitter = ET.SubElement(platform, 'transmitter')
        transmitter.set('name', name)
        transmitter.set('type', tx_type)
        transmitter.set('antenna', antenna)
        transmitter.set('pulse', pulse)
        transmitter.set('timing', timing)

        tx_prf = ET.SubElement(transmitter, 'prf')
        tx_prf.text = str(prf)

    def _add_receiver(self, platform, name, nodirect, antenna, nopropagationloss, timing, prf, window_length, noise_temp=290, window_skip=0):
        receiver = ET.SubElement(platform, 'receiver')
        receiver.set('name', name)
        receiver.set('nodirect', nodirect)
        receiver.set('antenna', antenna)
        receiver.set('nopropagationloss', nopropagationloss)
        receiver.set('timing', timing)

        skip = ET.SubElement(receiver, 'window_skip')
        skip.text = str(window_skip)

        window = ET.SubElement(receiver, 'window_length')
        window.text = str(window_length)

        rx_prf = ET.SubElement(receiver, 'prf')
        rx_prf.text = str(prf)

        noise = ET.SubElement(receiver, 'noise_temp')
        noise.text = str(noise_temp)

    def _add_platform(self, name, root):
        platform = ET.SubElement(root, 'platform')
        platform.set('name', name)
        return platform

    def _add_motionpath(self, platform, position_waypoints, interp='linear'):
        motionpath = ET.SubElement(platform, 'motionpath')
        motionpath.set('interpolation', interp)

        for waypoint in position_waypoints:
            self._add_positionwaypoint(motionpath, waypoint)


    def _add_rotationpath(self, platform, rotation_waypoints, interp='linear'):
        rotationpath = ET.SubElement(platform, 'rotationpath')
        rotationpath.set('interpolation', interp)

        for waypoint in rotation_waypoints:
            self._add_rotationwaypoint(rotationpath, waypoint)

    def _add_positionwaypoint(self, path, waypoint: FersPositionWaypoint):
        point = ET.SubElement(path, 'positionwaypoint')

        t_x = ET.SubElement(point, 'x')
        t_x.text = str(waypoint.x)

        t_y = ET.SubElement(point, 'y')
        t_y.text = str(waypoint.y)

        t_a = ET.SubElement(point, 'altitude')
        t_a.text = str(waypoint.z)

        t_t = ET.SubElement(point, 'time')
        t_t.text = str(waypoint.t)

    def _add_rotationwaypoint(self, path, waypoint: FersRotationWaypoint):
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
