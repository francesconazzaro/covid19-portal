import numpy as np
import scipy.stats


UNITA = 100000

RULE_MAP = {
    'Dati per 100.000 abitanti': 'percentage',
    'Totali': 'total',
}


def exp2(t, t_0, t_d):
    return 2 ** ((t - t_0) / t_d)


def linear(t, t_0, t_d):
    return (t - t_0) / t_d


P0 = (np.datetime64("2020-02-12", "s"), np.timedelta64(48 * 60 * 60, "s"))


def linear_fit(data, start=None, stop=None, p0=P0):
    t_0_guess, T_d_guess = p0
    data_fit = data[start:stop]
    x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
    y_fit = data_fit.values

    x_fit = x_norm[np.isfinite(y_fit)]

    m, y, r2, _, _ = scipy.stats.linregress(x_fit, y_fit)
    t_0_norm = -y / m
    T_d_norm = 1 / m

    t_d = T_d_norm * T_d_guess
    t_0 = t_0_guess + t_0_norm * T_d_guess
    return t_0, t_d, r2


def normalisation(data, population, rule):
    if RULE_MAP[rule] == 'percentage':
        new_data = data / population * UNITA
        try:
            new_data.name = data.name
        except AttributeError:
            pass
        return new_data
    else:
        new_data = data
        try:
            new_data.name = data.name
        except AttributeError:
            pass
        return new_data


def fit(data, start=None, stop=None, p0=P0, **kwargs):
    t_0_guess, T_d_guess = p0
    data_fit = data[start:stop]

    x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
    log2_y = np.log2(data_fit.values)

    t_fit = data_fit.index.values[np.isfinite(log2_y)]
    x_fit = x_norm[np.isfinite(log2_y)]
    log2_y_fit = log2_y[np.isfinite(log2_y)]

    m, y, r2, _, _ = scipy.stats.linregress(x_fit, log2_y_fit)
    t_0_norm = -y / m
    T_d_norm = 1 / m

    t_d = T_d_norm * T_d_guess
    t_0 = t_0_guess + t_0_norm * T_d_guess
    return t_0, t_d, r2
