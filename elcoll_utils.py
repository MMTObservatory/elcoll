import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from astropy import units as u
from astropy.io import ascii
from astropy.modeling import custom_model

colnames = ["el", "tiltx", "tilty", "transx", "transy", "focus", "temp", "UT", "oss1", "oss2", "oss3", "oss4", "oss5", "oss6"]
keys = ['tiltx', 'tilty', 'transx', 'transy', 'focus']
titles = ['Tilt X', 'Tilt Y', 'Trans X', 'Trans Y', 'Focus']
ylabels = ['arcsec', 'arcsec', 'microns', 'microns', 'microns']


# elcoll utility functions
def elcoll_read(filename):
    """
    Use astropy.io.ascii to read the data logged during elcoll runs
    """
    data = ascii.read(filename, names=colnames)
    data['oss'] = (data['oss1'] + data['oss3'] + data['oss5'])/3.0
    return data


def elcoll_plot(data, models=None, mean_temp=False):
    """
    Plot up all five hexapod coordinates as a function of elevation
    """
    f, axes = plt.subplots(5, sharex=True, figsize=(6, 15))
    for a, k, t, l in zip(axes, keys, titles, ylabels):
        a.scatter(data['el'], data[k])
        if models:
            if mean_temp:
                a.plot(data['el'], models[k](data['el'], np.mean(data['oss'])))
            else:
                data = data.sort('el')
                a.plot(data['el'], models[k](data['el'], data['oss']))
        a.set_title(t)
        a.set_ylabel(l)
    axes[0].set_xlim(20, 90)
    axes[-1].set_xlabel("Elevation (deg)")
    plt.show()


def elcoll_results(models):
    """
    Print up the results of the model fits in a more human readable form
    """
    for k, t, y in zip(keys, titles, ylabels):
        print("%s: %+.2f*sin(el) %+.2f*cos(el) %+.2f*T_oss %+.2f" % (
                t,
                models[k].c0.value,
                models[k].c1.value,
                models[k].c2.value,
                models[k].c3.value
            )
        )


@custom_model
def elcoll_model(el, t, c0=0.0, c1=0.0, c2=0.0, c3=0.0):
    """
    Model the elevation dependence as a combination of sin/cos and include a linear term for OSS temperature
    """
    model = c0 * np.sin(el * u.deg) + c1 * np.cos(el * u.deg) + c2 * t + c3
    return model


def load_current_models(xmlfile="../settings/settings.xml", hexapod='f5', inst=None):
    """
    Load the current open loop model parameters from the XML configuration file
    """
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    # hexapods is the 2nd entry in the tree.  within hexapods the order is f15, f5, and f9.
    hex_i = {}
    hex_i['f15'] = 0
    hex_i['f5'] = 1
    hex_i['f9'] = 2

    open_loop = root[1][hex_i[hexapod]][3]
    trans = {}
    trans['tx'] = "tiltx"
    trans['ty'] = "tilty"
    trans['x'] = "transx"
    trans['y'] = "transy"
    trans['z'] = "focus"
    current_mods = {}

    for e in open_loop.getchildren():
        term = trans[e.tag]
        coeffs = {}

        for coeff in e.getchildren():
            coeffs[coeff.tag] = float(coeff.text)

        current_mods[term] = elcoll_model()
        current_mods[term].c0 = coeffs['sin']
        current_mods[term].c1 = coeffs['cos']
        current_mods[term].c2 = coeffs['t_oss']
        key = 'zeropoint_%s' % inst
        if key in coeffs:
            current_mods[term].c3 = coeffs[key]
        else:
            current_mods[term].c3 = coeffs['zeropoint']

    return current_mods
