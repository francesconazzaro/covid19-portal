
import datetime
import itertools
import numpy as np
import scipy.optimize
import scipy.stats
from matplotlib import cm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

UNITA = 100000


def exp2(t, t_0, T_d):
    return 2 ** ((t - t_0) / T_d)


def linear(t, t_0, T_d):
    return (t - t_0) / T_d


RULE_MAP = {
    'Dati per 100.000 abitanti': 'percentage',
    'Totali': 'total',
}

PERCENTAGE_RULE = {
    'Percentuale di tamponi positivi' : 'tamponi',
    'Percentuale di casi positivi': 'casi',
}

ITALY_EVENTS = [
    # {'x': '2020-02-19', 'label': 'First alarm'},
    {'x': '2020-02-24', 'label': 'Chiusura scuole al Nord'},
    {'x': '2020-03-01', 'label': 'Contenimento parziale Nord'},
    {'x': '2020-03-05', 'label': 'Chiusura scuole Italia'},
    {'x': '2020-03-08', 'label': 'Lockdown Nord'},
    {'x': '2020-03-10', 'label': 'Lockdown Italia'},
    {'x': '2020-10-13', 'label': 'Mascherine obbligatorie'},
    {'x': '2020-10-24', 'label': 'Lockdown morbido'},
]


def add_events(fig, events=ITALY_EVENTS, start=None, stop=None, offset=0, **kwargs):
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    for event in events[start:stop]:
        label = '{x} {label}'.format(offset=offset, **event)
        fig.add_trace(go.Scatter(x=[event['x'], event['x']], y=[0, 10 ** 10], mode='lines',
                                 legendgroup='events', showlegend=False, line=dict(color=next(PALETTE), width=2),
                                 name=label))


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

    T_d = T_d_norm * T_d_guess
    t_0 = t_0_guess + t_0_norm * T_d_guess
    return t_0, T_d, r2


def fit(data, start=None, stop=None, p0=P0):
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

    T_d = T_d_norm * T_d_guess
    t_0 = t_0_guess + t_0_norm * T_d_guess
    return t_0, T_d, r2


def plot_fit(data, fig, start=None, stop=None, label=None, shift=5, **kwargs):
    t0, td, r2 = fit(data, start, stop)
    if label is not None:
        label = 'Fit {label}<br>(T_d = {0:.2f})'.format(td / np.timedelta64(1, "D"), label=label)
    x = data[start:stop].index
    x_base = np.arange(x[0] - np.timedelta64(shift, 'D'), x[-1] + np.timedelta64(shift, 'D'),
                       np.timedelta64(1, 'D'))
    trace = go.Line(x=x_base, y=exp2(x_base, t0, td), name=label, visible="legendonly", **kwargs)
    fig.add_trace(trace, 1, 1)
    return fig


def prediction(x, t0, td, r2, label, ax, days_after=1, func=exp2, **plot_kwargs):
    ax.plot(x, exp2(x, t0, td), linestyle=':', **plot_kwargs)
    x_predict = x[-1] + np.timedelta64(days_after, 'D')
    y_predict = func(x_predict, t0, td)
    if label is not None:
        label = 'Previsione {1} per oggi {0:.2f}'.format(y_predict, label)
    ax.scatter(x_predict, y_predict, marker='o', facecolors='none',
               label=label,
               **plot_kwargs)
    ax.text(x_predict + datetime.timedelta(days=1), y_predict, int(y_predict), **plot_kwargs)
    return ax


def get_matplotlib_cmap(cmap_name, bins, alpha=1):
    if bins is None:
        bins = 10
    cmap = cm.get_cmap(cmap_name)
    h = 1.0 / bins
    contour_colour_list = []

    for k in range(bins):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))

    C = list(map(np.uint8, np.array(cmap(bins * h)[:3]) * 255))
    contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))
    return contour_colour_list


def test_positivity_rate(data, country, rule):
    data = data[country]
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    PALETTE_ALPHA = itertools.cycle(get_matplotlib_cmap('tab10', bins=8, alpha=.3))
    fig = make_subplots(1, 1, subplot_titles=[rule])
    if PERCENTAGE_RULE[rule] == 'tamponi':
        plot_data = data.nuovi_positivi / data.tamponi.diff() * 100
    elif PERCENTAGE_RULE[rule] == 'casi':
        plot_data = data.nuovi_positivi / data.casi_testati.diff() * 100
    fig.add_trace(go.Line(
        x=plot_data.index,
        y=plot_data.rolling(7).mean().values,
        name=rule,
        mode='lines+markers',
        showlegend=False,
        legendgroup='postam',
        marker=dict(color=next(PALETTE))
    ), 1, 1)
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data.values,
        mode='markers',
        legendgroup='postam',
        showlegend=False,
        marker=dict(color=next(PALETTE_ALPHA))
    ), 1, 1)
    fig.add_annotation(x=plot_data.index[-1], y=plot_data.values[-1], text='{:.2f}'.format(plot_data.values[-1]))
    fig.add_annotation(x=plot_data.index[-1], y=plot_data.rolling(7).mean().values[-1], text='{:.2f}'.format(plot_data.rolling(7).mean().values[-1]))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, plot_data[-20:].max() + 1])
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=30, l=10, b=10, r=10),
        autosize=True,
    )
    return fig


def normalisation(data, population, rule):
    if RULE_MAP[rule] == 'percentage':
        new_data = data / population * UNITA
        new_data.name = data.name
        return new_data
    else:
        new_data = data
        new_data.name = data.name
        return new_data


def get_fmt(rule):
    if RULE_MAP[rule] == 'percentage':
        return '{:.2f}'
    else:
        return '{:.0f}'


def plot_average(plot_data, palette, fig, name, palette_alpha, fmt, start=None, stop=None, log=True):
    line = dict(color=next(palette))
    fig.add_trace(go.Line(
        x=plot_data.index, y=plot_data.rolling(7).mean().values,
        name=name,
        legendgroup=name,
        line=line,
        mode='lines'
    ), 1, 1)
    line['dash'] = 'dot'
    if start and stop:
        plot_fit(
            plot_data.rolling(7).mean(),
            fig,
            label=name,
            start=start,
            stop=stop,
            mode='lines',
            line=line,
            shift=5
        )
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data.values,
        mode='markers',
        legendgroup=name,
        showlegend=False,
        marker=dict(color=next(palette_alpha))
    ), 1, 1)
    if log is True:
        y = np.log10(plot_data.values[-1])
    else:
        y = plot_data.values[-1]
    fig.add_annotation(
        x=plot_data.index[-1],
        y=y,
        text=fmt.format(plot_data.values[-1])
    )


def plot_selection(data, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri, start_deceduti, stop_deceduti, log=True):

    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    PALETTE_ALPHA = itertools.cycle(get_matplotlib_cmap('tab10', bins=8, alpha=.3))

    data = data[country]

    maxs = []
    mins = []

    fig = make_subplots(1, 1, subplot_titles=[country])
    fmt = get_fmt(rule)

    plot_data = normalisation(data.nuovi_positivi, data.popolazione, rule)
    maxs.append(plot_data.max())
    mins.append((plot_data.rolling(7).mean()[20:] + .001).min())
    plot_average(
        plot_data,
        fig=fig,
        name='Nuovi Positivi',
        start=start_positivi,
        stop=stop_positivi,
        palette=PALETTE,
        palette_alpha=PALETTE_ALPHA,
        fmt=fmt,
        log=log,
    )

    plot_data = normalisation(data.ricoverati_con_sintomi, data.popolazione, rule)
    maxs.append(plot_data.max())
    mins.append((plot_data.rolling(7).mean()[20:] + .001).min())
    plot_average(
        plot_data,
        fig=fig,
        name='Ricoveri',
        start=start_ricoveri,
        stop=stop_ricoveri,
        palette=PALETTE,
        palette_alpha=PALETTE_ALPHA,
        fmt=fmt,
        log=log,
    )

    plot_data = normalisation(data.terapia_intensiva, data.popolazione, rule)
    maxs.append(plot_data.max())
    mins.append((plot_data.rolling(7).mean()[20:] + .001).min())
    plot_average(
        plot_data,
        fig=fig,
        name='Terapie Intensive',
        start=start_ti,
        stop=stop_ti,
        palette=PALETTE,
        palette_alpha=PALETTE_ALPHA,
        fmt=fmt,
        log=log,
    )

    plot_data = normalisation(data.deceduti, data.popolazione, rule).diff()
    maxs.append(plot_data.max())
    mins.append((plot_data.rolling(7).mean()[20:] + .001).min())
    plot_average(
        plot_data,
        fig=fig,
        name='Deceduti',
        start=start_deceduti,
        stop=stop_deceduti,
        palette=PALETTE,
        palette_alpha=PALETTE_ALPHA,
        fmt=fmt,
        log=log,
    )

    # add_events(fig)

    if log is True:
        maximum = np.nanmax(np.log10(maxs)) + .5
        minimum = np.nanmin(np.log10(mins))
        yscale = 'log'
    else:
        maximum = np.nanmax(maxs)
        minimum = np.nanmin(mins)
        yscale = 'linear'
    fig.update_xaxes(row=1, col=1, showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(row=1, col=1, type=yscale, showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[minimum, maximum], showexponent='all', exponentformat='power')
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=70, l=0, b=0, r=0),
        yaxis_title=f'{rule}',
        # width=1300,
        height=500,
        autosize=True,
        legend={
            'orientation': "h",
            'yanchor': "bottom",
            # 'y': -.15, # bottom
            'y': .9, # top
            'xanchor': "center",
            'x': .5,
        }
    )
    return fig


def summary(data, what, st):
    titles = [title for title in data if title not in ['P.A. Bolzano', 'P.A. Trento']]
    fig = make_subplots(4, 5, shared_xaxes='all', shared_yaxes='all', subplot_titles=titles,
                        vertical_spacing=.08)
    minus = 0
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    maxs = []
    for i, name in enumerate(data):
        col = (i - minus) % 5 + 1
        row = (i - minus) // 5 + 1
        region = data[name]
        if name in ['P.A. Bolzano', 'P.A. Trento']:
            minus += 1
            continue
        if what == 'Terapie Intensive':
            plot_data = region.terapia_intensiva.rolling(7).mean() / region.popolazione * UNITA
            title = "Terapie Intensive per 100.000 abitanti"
            yscale = 'log'
        elif what == 'Nuovi Positivi':
            plot_data = region.nuovi_positivi.rolling(7).mean() / region.popolazione * UNITA
            title = "Nuovi positivi per 100.000 abitanti"
            yscale = 'log'
        elif what == 'Percentuale tamponi positivi':
            plot_data = region.nuovi_positivi.rolling(7).mean() / region.tamponi.diff().rolling(
                7).mean() * 100
            title = "Percentuale tamponi positivi."
            yscale = 'linear'
        elif what == 'Deceduti':
            plot_data = region.deceduti.diff().rolling(7).mean() / region.tamponi.diff().rolling(
                7).mean() * 100
            title = "Deceduti giornalieri per 100.000 abitanti."
            yscale = 'log'
        maxs.append(plot_data.values[-90:].max())
        fig.add_trace(go.Line(x=plot_data.index[-90:], y=plot_data.values[-90:], showlegend=False,
                              name=title, marker=dict(color=next(PALETTE))), row, col)
    st.subheader(title)
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs) + 1])
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        # width=1300,
        height=500,
        autosize=True,
    )
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=15, color=next(PALETTE))
    return fig


def translate(data, days_delay=0):
    dates = data.index + np.timedelta64(days_delay, 'D')
    data_t = data.copy()
    data_t.index = dates
    return data_t


def mortality(data, offset=-7):
    italy = data['Italia']
    fig = make_subplots(1, 1, subplot_titles=['Mortalità apparente'])
    translated_cases = translate(italy.nuovi_positivi, days_delay=offset)
    plot_data = italy.deceduti.diff() / translated_cases * 100
    fig.add_trace(go.Line(
        x=plot_data.index, y=plot_data.rolling(7).mean().values,
        mode='lines'
    ))
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        yaxis_title='Percentuale di deceduti su nuovi positivi di {} giorni prima'.format(abs(offset)),
        # width=1300,
        height=500,
        autosize=True,
    )
    return fig


def comparison(data, offset=7):
    deceduti = data.deceduti.diff().rolling(7).mean()
    positivi_shift = translate(data.nuovi_positivi.rolling(7).mean(), offset) / 100 * 1.3
    fig = make_subplots(1, 1, subplot_titles=["Confronto fra deceduti e l'1.3% dei nuovi casi ritardati di 7 giorni"])
    fig.add_trace(go.Line(
        x=positivi_shift.index, y=positivi_shift.values,
        mode='lines',
        name='1.3% dei nuovi casi spostati di 7 giorni',
        line={'dash': 'dot'},
    ))
    fig.add_trace(go.Line(
        x=deceduti.index, y=deceduti.values,
        mode='lines',
        name='Deceduti'
    ))
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, range=[positivi_shift.index[-120], positivi_shift.index[-1]], gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, type='log', gridwidth=1, gridcolor='LightGrey')
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        yaxis_title='',
        # width=1300,
        height=500,
        autosize=True,
        legend={
            'orientation': "h",
            'yanchor': "bottom",
            # 'y': -.15, # bottom
            'y': .9,  # top
            'xanchor': "center",
            'x': .5,
        }
    )
    return fig
