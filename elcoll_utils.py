import numpy as np
import matplotlib.pyplot as plt

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


def elcoll_plot(data):
    """
    Plot up all five hexapod coordinates as a function of elevation
    """
    f, axes = plt.subplots(5, sharex=True, figsize=(6, 15))
    for a, k, t, l in zip(axes, keys, titles, ylabels):
        a.scatter(data['el'], data[k])
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
