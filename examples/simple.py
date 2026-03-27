import pyfers as fers

# radar parameters
chirp = fers.Waveform(name='chirp', f_carrier=9e6, type='pulse', power=20, f_sample=150e6, bandwidth=100e6, t_pulse=2e-6)
antenna = fers.Antenna(name='antenna', type='sinc', gain=2, efficiency=1, az_beta=20, el_beta=5, az_gamma=2, el_gamma=2)
clock = fers.Clock(name='clock', frequency=150e6)
transmitter = fers.Transmitter(name='transmitter', antenna=antenna, waveform=chirp, clock=clock, f_prf=1000)
receiver = fers.Receiver(name='receiver', antenna=antenna, clock=clock, f_prf=1000, gate=1000, noise_temp=100)

# platforms
radar_platform = fers.StaticPlatform(name='radar platform', x=0, y=0, z=0)
target_platform = fers.StaticPlatform(name='target platform', x=250, y=400, z=0)

# targets
target = fers.Target('target', rcs=100, platform=target_platform)

# FERS simulation
sim = fers.Simulation(name='simple', filename="simple.fersxml")
sim.add_parameters(t_start=0, t_end=10, sim_rate=150e6, bits=16)
sim.add_waveform(chirp, "waveform.h5")
sim.add_clock(clock)
sim.add_antenna(antenna, "antenna.xml")
sim.add_monostatic(radar_platform, transmitter, receiver, antenna, chirp, clock)
sim.add_target(target)
sim.write_xml()
sim.run()

# read results
rx_matrix = fers.read_hdf5(receiver.name + "_results.h5")