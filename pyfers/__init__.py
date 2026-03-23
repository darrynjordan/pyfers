import os
import h5py
import numpy as np
import subprocess as sbp
from lxml import etree as ET

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
