# pyfers
Python package that enables simple generation of XML descriptors required by FERS.

## Install
```
pip3 install pyfers
```

## Example
```
import pyfers.fers as fers

fers.write_hdf5(tx_chirp, FERS_INPUT_FILENAME)

fers_xml = fers.FersXMLGenerator(FERS_XML_FILENAME)

fers_xml.add_parameters(
    t_start=T_pri,
    t_end=T_slow,
    sim_rate=adc_rate,
    bits=adc_bits
    )

fers_xml.add_pulse(
    name='up_chirp',
    pulse_file=FERS_INPUT_FILENAME,
    power_watts=P_tx,
    centre_freq=fc
    )

fers_xml.add_clock(name='clock', frequency=adc_rate, synconpulse='false')

# fers_xml.add_antenna(name='tx_rx_antenna', pattern='isotropic')
# fers_xml.add_antenna(name='tx_rx_antenna', pattern='parabolic', d=10)
fers_xml.add_antenna(name='tx_rx_antenna', pattern='sinc', a=alpha, b=beta, g=gamma)

fers_xml.add_pseudo_monostatic_radar(
    spacing=antenna_spacing,
    waypoints=waypoints,
    antenna='tx_rx_antenna',
    timing='clock',
    prf=F_prf,
    pulse='up_chirp',
    window_length=T_keep,
    noise_temp=noise_temp,
    nodirect='true'
)

for target in targets:
    fers_xml.add_target(
        name = target.name,
        x = target.x,
        y = target.y,
        z = target.z,
        t = target.t,
        rcs = target.rcs
    )

fers_xml.write_xml()
fers_xml.run()
```
