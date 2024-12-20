import os
import h5py
import numpy as np
import subprocess as sbp

from xml.etree.ElementTree import Element, SubElement
from xml.etree.ElementTree import ElementTree as Tree
from xml.etree import ElementTree
from xml.dom import minidom

def prettify_xml(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

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
        self.root = Element('antenna')
        self.elevation = SubElement(self.root, 'elevation')
        self.azimuth = SubElement(self.root, 'azimuth')

    def add_gainsample(self, plane, angle, gain):
        gainsample = SubElement(plane, "gainsample")

        a = SubElement(gainsample, "angle")
        a.text = str(angle)

        g = SubElement(gainsample, "gain")
        g.text = str(gain)

    def write(self):
        with open(self.filename, "w") as f:
            f.write(prettify_xml(self.root))


class FersXMLGenerator:
    def __init__(self, xml_filename):
        """
        FersXMLGenerator constructor.
        """
        self.filename = xml_filename
        self.simulation = Element('simulation')
        self.simulation.set('name', 'milosar')
        self.tree = Tree(self.simulation)

    def add_parameters(self, t_start, t_end, sim_rate, bits, over_sample=1, interp_rate=1000):
        parameters = SubElement(self.simulation, 'parameters')

        starttime = SubElement(parameters, 'starttime')
        starttime.text = str(t_start)

        endtime = SubElement(parameters, 'endtime')
        endtime.text = str(t_end)

        rate = SubElement(parameters, 'rate')
        rate.text = str(sim_rate)

        adc_bits = SubElement(parameters, 'adc_bits')
        adc_bits.text = str(bits)

        oversample = SubElement(parameters, 'oversample')
        oversample.text = str(over_sample)

        interprate = SubElement(parameters, 'interprate')
        interprate.text = str(interp_rate)

        light_speed = SubElement(parameters, 'c')
        light_speed.text = str(299792458)

    def add_pulse(self, name, pulse_file, power_watts, centre_freq):
        pulse = SubElement(self.simulation, 'pulse')
        pulse.set('name', name)
        pulse.set('type', 'file')
        pulse.set('filename', pulse_file)

        power = SubElement(pulse, 'power')
        power.text = str(power_watts)

        carrier = SubElement(pulse, 'carrier')
        carrier.text = str(centre_freq)

    def add_clock(self, name, frequency, f_offset=0, random_f_offset=0, p_offset=0, random_p_offset=0, synconpulse='true'):
        timing = SubElement(self.simulation, 'timing')
        timing.set('name', name)
        timing.set('synconpulse', synconpulse)

        freq = SubElement(timing, 'frequency')
        freq.text = str(frequency)

        freq_offset = SubElement(timing, 'freq_offset')
        freq_offset.text = str(f_offset)

        random_freq_offset = SubElement(timing, 'random_freq_offset')
        random_freq_offset.text = str(random_f_offset)

        phase_offset = SubElement(timing, 'phase_offset')
        phase_offset.text = str(p_offset)

        random_phase_offset = SubElement(timing, 'random_phase_offset')
        random_phase_offset.text = str(random_p_offset)

        # add_noise(timing, -2, 1e-6)
        # add_noise(timing, -1, 1e-6)
        # add_noise(timing, 0, 1e-6)
        # add_noise(timing, 1, 1e-6)
        # add_noise(timing, 2, 1e-6)

    def add_antenna(self, name, pattern, eff=1, a=1, b=2, g=5, d=1, azscale=1, elscale=1, filename=None):
        antenna = SubElement(self.simulation, 'antenna')
        antenna.set('name', name)
        antenna.set('pattern', pattern)

        efficiency = SubElement(antenna, 'efficiency')
        efficiency.text = str(eff)

        if (pattern == "parabolic"):
            diameter = SubElement(antenna, 'diameter')
            diameter.text = str(d)

        if (pattern == "sinc"):
            alpha = SubElement(antenna, 'alpha')
            alpha.text = str(a)

            beta = SubElement(antenna, 'beta')
            beta.text = str(b)

            gamma = SubElement(antenna, 'gamma')
            gamma.text = str(g)

        if (pattern == "gaussian"):
            az = SubElement(antenna, 'azscale')
            az.text = str(azscale)

            el = SubElement(antenna, 'elscale')
            el.text = str(elscale)

        if (pattern == "xml"):
            antenna.set('filename', filename)

    def add_monostatic_radar(self, antenna, timing, prf, pulse, position_waypoints, rotation_waypoints, window_length, noise_temp=290, window_skip=0, tx_type='pulsed', interp='linear'):
        platform = add_platform('radar_platform', self.simulation)
        add_motionpath(platform, position_waypoints, interp)
        add_rotationpath(platform, rotation_waypoints, interp)
        add_monostatic(platform, 'receiver', tx_type, antenna, pulse, timing, prf, window_length, noise_temp, window_skip)

    def add_target(self, fers_target: FersTarget, interp='linear'):
        platform = add_platform('target_platform', self.simulation)
        add_motionpath(platform, fers_target.position_waypoints, interp)
        add_fixedrotation(platform)

        target = SubElement(platform, 'target')
        target.set('name', fers_target.name)

        t_rcs = SubElement(target, 'rcs')
        t_rcs.set('type', 'isotropic')

        t_rcs_v = SubElement(t_rcs, 'value')
        t_rcs_v.text = str(fers_target.rcs)

    def write_xml(self):
        self.tree.write(self.filename)

        my_file = open(self.filename, "w")
        my_file.write(prettify_xml(self.simulation))
        my_file.close()

    def run(self):
        try:
            sbp.run(['fers', self.filename])
        except:
            print('ERROR: failed to launch - check that FERS is installed correctly.')
            exit(1)

def add_monostatic(platform, name, tx_type, antenna, pulse, timing, prf, window_length, noise_temp=290, window_skip=0):
    monostatic = SubElement(platform, 'monostatic')
    monostatic.set('name', name)
    monostatic.set('type', tx_type)
    monostatic.set('antenna', antenna)
    monostatic.set('pulse', pulse)
    monostatic.set('timing', timing)

    skip = SubElement(monostatic, 'window_skip')
    skip.text = str(window_skip)

    window = SubElement(monostatic, 'window_length')
    window.text = str(window_length)

    rx_prf = SubElement(monostatic, 'prf')
    rx_prf.text = str(prf)

    noise = SubElement(monostatic, 'noise_temp')
    noise.text = str(noise_temp)

def add_transmitter (platform, name, tx_type, antenna, pulse, timing, prf):
    transmitter = SubElement(platform, 'transmitter')
    transmitter.set('name', name)
    transmitter.set('type', tx_type)
    transmitter.set('antenna', antenna)
    transmitter.set('pulse', pulse)
    transmitter.set('timing', timing)

    tx_prf = SubElement(transmitter, 'prf')
    tx_prf.text = str(prf)

def add_receiver (platform, name, nodirect, antenna, nopropagationloss, timing, prf, window_length, noise_temp=290, window_skip=0):
    receiver = SubElement(platform, 'receiver')
    receiver.set('name', name)
    receiver.set('nodirect', nodirect)
    receiver.set('antenna', antenna)
    receiver.set('nopropagationloss', nopropagationloss)
    receiver.set('timing', timing)

    skip = SubElement(receiver, 'window_skip')
    skip.text = str(window_skip)

    window = SubElement(receiver, 'window_length')
    window.text = str(window_length)

    rx_prf = SubElement(receiver, 'prf')
    rx_prf.text = str(prf)

    noise = SubElement(receiver, 'noise_temp')
    noise.text = str(noise_temp)

def add_platform (name, root):
    platform = SubElement(root, 'platform')
    platform.set('name', name)
    return platform

def add_motionpath(platform, position_waypoints, interp='linear'):
    motionpath = SubElement(platform, 'motionpath')
    motionpath.set('interpolation', interp)

    for waypoint in position_waypoints:
        add_positionwaypoint(motionpath, waypoint)


def add_rotationpath (platform, rotation_waypoints, interp='linear'):
    rotationpath = SubElement(platform, 'rotationpath')
    rotationpath.set('interpolation', interp)

    for waypoint in rotation_waypoints:
        add_rotationwaypoint(rotationpath, waypoint)

def add_positionwaypoint(path, waypoint: FersPositionWaypoint):
    point = SubElement(path, 'positionwaypoint')

    t_x = SubElement(point, 'x')
    t_x.text = str(waypoint.x)

    t_y = SubElement(point, 'y')
    t_y.text = str(waypoint.y)

    t_a = SubElement(point, 'altitude')
    t_a.text = str(waypoint.z)

    t_t = SubElement(point, 'time')
    t_t.text = str(waypoint.t)

def add_rotationwaypoint(path, waypoint: FersRotationWaypoint):
    point = SubElement(path, 'rotationwaypoint')

    t_el = SubElement(point, 'elevation')
    t_el.text = str(waypoint.el)

    t_az = SubElement(point, 'azimuth')
    t_az.text = str(waypoint.az)

    t_t = SubElement(point, 'time')
    t_t.text = str(waypoint.t)

def add_fixedrotation(platform, s_az=2*np.pi, az_rate=0, s_el=2*np.pi, el_rate=0):
    rotation = SubElement(platform, 'fixedrotation')

    t_s_az = SubElement(rotation, 'startazimuth')
    t_s_az.text = str(s_az)

    t_az_r = SubElement(rotation, 'azimuthrate')
    t_az_r.text = str(az_rate)

    t_s_el = SubElement(rotation, 'startelevation')
    t_s_el.text = str(s_el)

    t_el_r = SubElement(rotation, 'elevationrate')
    t_el_r.text = str(el_rate)


def add_noise(clock, alpha, weight):
    noise_entry = SubElement(clock, 'noise_entry')

    t_alpha = SubElement(noise_entry, 'alpha')
    t_alpha.text = str(alpha)

    t_weight = SubElement(noise_entry, 'weight')
    t_weight.text = str(weight)

if __name__ == "__main__":
    antenna = FersAntennaXML("test.xml")
    antenna.add_gainsample(antenna.elevation, 0, 1)
    antenna.add_gainsample(antenna.azimuth, 0, 5)
    antenna.write()