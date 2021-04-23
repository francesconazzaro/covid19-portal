import builtins
import copy
import datetime
import itertools

import numpy as np
import plotly
import plotly.graph_objects as go
import io
# import scipy.optimize
# import scipy.stats
import streamlit as st
# from matplotlib import cm
from plotly.subplots import make_subplots
import requests

import import_data

UNITA = 100000

def exp2(t, t_0, T_d):
    return 2 ** ((t - t_0) / T_d)


def linear(t, t_0, T_d):
    return (t - t_0) / T_d


RULE_MAP = {
    'Totali': 'total',
    'Dati per 100.000 abitanti': 'percentage',
}

PERCENTAGE_RULE = {
    'Percentuale di tamponi positivi': 'tamponi',
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

CATEGORIES = [
    {'name': 'categoria_operatori_sanitari_sociosanitari', 'label': 'Operatori sanitari sociosanitari'},
    {'name': 'categoria_personale_non_sanitario', 'label': 'Personale non sanitario'},
    {'name': 'categoria_ospiti_rsa', 'label': 'Ospiti RSA'},
    {'name': 'categoria_over80', 'label': 'Over 80'},
    {'name': 'categoria_forze_armate', 'label': 'Forze armate'},
    {'name': 'categoria_personale_scolastico', 'label': 'Personale scolastico'},
    {'name': 'categoria_60_69', 'label': '60-69'},
    {'name': 'categoria_70_79', 'label': '70-79'},
    {'name': 'categoria_soggetti_fragili', 'label': 'Soggetti fragili'},
    {'name': 'categoria_altro', 'label': 'Altro'},
]


def add_events(fig, events=ITALY_EVENTS, start=None, stop=None, offset=0, **kwargs):
    PALETTE = get_default_palette()
    for event in events[start:stop]:
        label = '{x} {label}'.format(offset=offset, **event)
        fig.add_trace(go.Scatter(x=[event['x'], event['x']], y=[0, 10 ** 10], mode='lines',
                                 legendgroup='events', showlegend=False, line=dict(color=next(PALETTE), width=2),
                                 name=label))


P0 = (np.datetime64("2020-02-12", "s"), np.timedelta64(48 * 60 * 60, "s"))


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_default_palette(alpha=False):
    hex_palette = copy.deepcopy(plotly.colors.qualitative.Plotly)
    hex_palette.pop(3)
    rgb_palette = []
    for hex_color in hex_palette:
        rgb_color = hex_to_rgb(hex_color)
        if alpha:
            rgb_color.append(.3)
            rgb_palette.append('rgba({},{},{},{})'.format(*rgb_color))
        else:
            rgb_palette.append(hex_color)

    return itertools.cycle(rgb_palette)


# def linear_fit(data, start=None, stop=None, p0=P0):
#     t_0_guess, T_d_guess = p0
#     data_fit = data[start:stop]
#     x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
#     y_fit = data_fit.values

#     x_fit = x_norm[np.isfinite(y_fit)]

#     m, y, r2, _, _ = scipy.stats.linregress(x_fit, y_fit)
#     t_0_norm = -y / m
#     T_d_norm = 1 / m

#     T_d = T_d_norm * T_d_guess
#     t_0 = t_0_guess + t_0_norm * T_d_guess
#     return t_0, T_d, r2


# def fit(data, start=None, stop=None, p0=P0):
#     t_0_guess, T_d_guess = p0
#     data_fit = data[start:stop]

#     x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
#     log2_y = np.log2(data_fit.values)

#     t_fit = data_fit.index.values[np.isfinite(log2_y)]
#     x_fit = x_norm[np.isfinite(log2_y)]
#     log2_y_fit = log2_y[np.isfinite(log2_y)]

#     m, y, r2, _, _ = scipy.stats.linregress(x_fit, log2_y_fit)
#     t_0_norm = -y / m
#     T_d_norm = 1 / m

#     T_d = T_d_norm * T_d_guess
#     t_0 = t_0_guess + t_0_norm * T_d_guess
#     return t_0, T_d, r2


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


# def get_matplotlib_cmap(cmap_name, bins, alpha=1):
#     if bins is None:
#         bins = 10
#     cmap = cm.get_cmap(cmap_name)
#     h = 1.0 / bins
#     contour_colour_list = []

#     for k in range(bins):
#         C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
#         contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))

#     C = list(map(np.uint8, np.array(cmap(bins * h)[:3]) * 255))
#     contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))
#     return contour_colour_list


@st.cache(allow_output_mutation=True, show_spinner=False)
def test_positivity_rate(data_in, country, rule):
    data = data_in[country]
    PALETTE = get_default_palette() #itertools.cycle(plotly.colors.qualitative.Plotly) #get_matplotlib_cmap('tab10', bins=8))
    PALETTE_ALPHA = get_default_palette(True)
    next(PALETTE_ALPHA)
    next(PALETTE)
    fig = make_subplots(1, 1, subplot_titles=[f'{country}: {rule}'], specs=[[{"secondary_y": True}]])
    # plot_data = region.nuovi_positivi.rolling(7).mean() / region.tamponi.diff().rolling(
    #     7).mean() * 100
    if PERCENTAGE_RULE[rule] == 'tamponi':
        plot_data = data.nuovi_positivi.rolling(7).mean() / data.tamponi.diff().rolling(7).mean() * 100
        plot_data_points = data.nuovi_positivi / data.tamponi.diff() * 100
    elif PERCENTAGE_RULE[rule] == 'casi':
        plot_data = data.nuovi_positivi.rolling(7).mean() / data.casi_testati.diff().rolling(7).mean() * 100
        plot_data_points = data.nuovi_positivi / data.casi_testati.diff() * 100
    fig.add_trace(go.Line(
        x=plot_data.index,
        y=plot_data.values,
        name='Media',
        # mode='lines+markers',
        showlegend=False,
        legendgroup='postam',
        marker=dict(color=next(PALETTE)),
        fill='tozeroy',
    ), 1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=plot_data_points.index,
        y=plot_data_points.values,
        name=rule,
        mode='markers',
        legendgroup='postam',
        showlegend=False,
        marker=dict(color=next(PALETTE_ALPHA)),
    ), 1, 1, secondary_y=True)
    # fig.add_annotation(x=plot_data.index[-1], y=plot_data.values[-1], text='{:.2f}'.format(plot_data.values[-1]))
    # fig.add_annotation(x=plot_data.index[-1], y=plot_data.rolling(7).mean().values[-1], text='{:.2f}'.format(plot_data.rolling(7).mean().values[-1]))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, plot_data[-500:].max() + 1], secondary_y=True)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=30, l=10, b=10, r=10),
        autosize=True,
        height=500,
        hovermode="x unified",
    )
    return fig


@st.cache(show_spinner=False)
def normalisation(data, population, rule):
    if RULE_MAP[rule] == 'percentage':
        new_data = data / population * UNITA
        try:
            new_data.name = data.name
        except:
            pass
        return new_data
    else:
        new_data = data
        try:
            new_data.name = data.name
        except:
            pass
        return new_data


def get_fmt(rule):
    if RULE_MAP[rule] == 'percentage':
        return '{:.2f}'
    else:
        return '{:.0f}'


def plot_average(plot_data, palette, fig, name, palette_alpha, secondary_y=True, fmt=None, start=None, stop=None, log=True):
    line = dict(color=next(palette))
    fig.add_trace(go.Line(
        x=plot_data.index, y=plot_data.rolling(7).mean().values,
        name=f'Media {name}',
        legendgroup=name,
        line=line,
        mode='lines',
    ), 1, 1, secondary_y=secondary_y)
    line['dash'] = 'dot'
    # if start and stop:
    #     plot_fit(
    #         plot_data.rolling(7).mean(),
    #         fig,
    #         label=name,
    #         start=start,
    #         stop=stop,
    #         mode='lines',
    #         line=line,
    #         shift=5
    #     )
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data.values,
        mode='markers',
        legendgroup=name,
        name=f'{name}',
        showlegend=False,
        marker=dict(color=next(palette_alpha))
    ), 1, 1, secondary_y=secondary_y)
    # if log is True:
    #     y = np.log10(plot_data.values[-1])
    # else:
    #     y = plot_data.values[-1]
    # fig.add_annotation(
    #     x=plot_data.index[-1],
    #     y=y,
        # text=fmt.format(plot_data.values[-1])
    # )


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_selection(data_in, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri, start_deceduti, stop_deceduti, log=True, secondary_y=True):

    PALETTE = get_default_palette()  # get_default_palette()
    PALETTE_ALPHA = get_default_palette(True)  # get_default_palette()

    data = data_in[country]

    maxs = []
    mins = []

    fig = make_subplots(1, 1, subplot_titles=[country], specs=[[{"secondary_y": secondary_y}]])
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
        secondary_y=secondary_y,
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
        secondary_y=secondary_y,
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
        secondary_y=secondary_y,
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
        secondary_y=secondary_y,
    )

    next(PALETTE_ALPHA)
    next(PALETTE)
    plot_data = normalisation(data.ingressi_terapia_intensiva, data.popolazione, rule)
    maxs.append(plot_data.max())
    mins.append((plot_data.rolling(7).mean()[20:] + .001).min())
    plot_average(
        plot_data,
        fig=fig,
        name='Ingressi Terapie Intensive',
        start=start_ti,
        stop=stop_ti,
        palette=PALETTE,
        palette_alpha=PALETTE_ALPHA,
        fmt=fmt,
        log=log,
        secondary_y=secondary_y,
    )

    if log is True:
        maximum = np.nanmax(np.log10(maxs)) + .5
        minimum = np.nanmin(np.log10(mins))
        yscale = 'log'
    else:
        maximum = np.nanmax(maxs)
        minimum = np.nanmin(mins)
        yscale = 'linear'
    fig.update_xaxes(row=1, col=1, showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(row=1, col=1, type=yscale, showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[minimum, maximum], showexponent='all', exponentformat='power', secondary_y=secondary_y)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=70, l=0, b=0, r=0),
        yaxis_title=f'{rule}',
        # width=1300,
        height=500,
        autosize=True,
        hovermode="x unified",
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


def vaccines_summary_plot_data(region, what):
    if what == 'Dosi somministrate':
        plot_data = (region.prima_dose.cumsum() + region.seconda_dose.cumsum()) / region.popolazione * UNITA
        title = "Dosi somministrate per 100 mila abitanti"
    if what == 'Percentuale popolazione vaccinata':
        plot_data = region.seconda_dose.cumsum() / region.popolazione * 100
        title = "Percentuale popolazione vaccinata"
    if what == 'Dosi consegnate':
        plot_data = region.prima_dose.cumsum() / region.popolazione * UNITA
        title = "Dosi consegnate per 100 mila abitanti"
    return plot_data, title

@st.cache(allow_output_mutation=True, show_spinner=False)
def vaccines_summary(vaccines, what):
    titles = [title for title in np.unique(vaccines.administration.area) if title not in ['Italia', 'P.A. Bolzano', 'P.A. Trento']]
    titles = sorted(['Trentino Alto Adige'] + titles)
    fig = make_subplots(4, 5, shared_xaxes='all', shared_yaxes='all', subplot_titles=titles,
                        vertical_spacing=.08)
    minus = 0
    PALETTE = itertools.cycle(plotly.colors.qualitative.Plotly) #get_matplotlib_cmap('tab10', bins=8))
    maxs = []
    italia = vaccines.administration[vaccines.administration.area == 'Italia']
    plot_data_italy, _ = vaccines_summary_plot_data(italia, what)
    for i, name in enumerate(titles):
        col = (i - minus) % 5 + 1
        row = (i - minus) // 5 + 1
        if name == 'Trentino Alto Adige':
            bolzano = vaccines.administration[vaccines.administration.area == 'P.A. Bolzano']
            trento = vaccines.administration[vaccines.administration.area == 'P.A. Trento']
            region = bolzano + trento
        else:
            region = vaccines.administration[vaccines.administration.area == name]
        plot_data, title = vaccines_summary_plot_data(region, what)
        maxs.append(plot_data.values[-90:].max())
        fig.add_trace(go.Scatter(x=plot_data.index[-90:], y=plot_data.values[-90:], showlegend=False,
                                 name=name, marker=dict(color=next(PALETTE)), fill='tozeroy'), row, col)
        fig.add_trace(go.Scatter(x=plot_data_italy.index[-90:], y=plot_data_italy.values[-90:], showlegend=False,
                                 name='Italia', marker=dict(color='rgba(31, 119, 180, .5)')), row,
                      col)
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs)])
    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        # width=1300,
        height=600,
        autosize=True,
        hovermode="x unified",
    )
    PALETTE = itertools.cycle(plotly.colors.qualitative.Plotly) #get_matplotlib_cmap('tab10', bins=8))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color=next(PALETTE))
    return fig


def summary_plot_data(region, what):
    if what == 'Terapie Intensive':
        plot_data = region.terapia_intensiva.rolling(7).mean() / region.popolazione * UNITA
        title = "Terapie Intensive per 100.000 abitanti"
        yscale = 'log'
    if what == 'Ingressi Terapie Intensive':
        plot_data = region.ingressi_terapia_intensiva.rolling(
            7).mean() / region.popolazione * UNITA
        title = "Ingressi Terapie Intensive per 100.000 abitanti"
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
    return plot_data, title


@st.cache(allow_output_mutation=True, show_spinner=False)
def summary(data, what):
    titles = [title for title in data if title not in ['P.A. Bolzano', 'P.A. Trento', 'Italia']]
    titles += ['Trentino Alto Adige']
    titles = sorted(titles)
    fig = make_subplots(4, 5, shared_xaxes='all', shared_yaxes='all', subplot_titles=titles,
                        vertical_spacing=.08)
    minus = 0
    PALETTE = get_default_palette()  # get_default_palette()
    maxs = []
    plot_data_italy, _ = summary_plot_data(data['Italia'], what)
    for i, name in enumerate(titles):
        col = (i - minus) % 5 + 1
        row = (i - minus) // 5 + 1
        if name == 'Trentino Alto Adige':
            region = data['P.A. Bolzano'] + data['P.A. Trento']
        else:
            region = data[name]
        plot_data, title = summary_plot_data(region, what)
        maxs.append(plot_data.values[-90:].max())
        fig.add_trace(go.Scatter(x=plot_data.index[-90:], y=plot_data.values[-90:], showlegend=False,
                                 name=name, marker=dict(color=next(PALETTE)), fill='tozeroy'), row, col)
        fig.add_trace(go.Scatter(x=plot_data_italy.index[-90:], y=plot_data_italy.values[-90:], showlegend=False,
                                 name='Italia', marker=dict(color='rgba(31, 119, 180, .5)')), row,
                      col)
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs)])
    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        # width=1300,
        height=600,
        autosize=True,
        hovermode="x unified",
    )
    PALETTE = get_default_palette()  # get_default_palette()
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color=next(PALETTE))
    return fig


def translate(data, days_delay=0):
    dates = data.index + np.timedelta64(days_delay, 'D')
    data_t = data.copy()
    data_t.index = dates
    return data_t


@st.cache(show_spinner=False)
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


@st.cache(allow_output_mutation=True, show_spinner=False)
def comparison(data, offset=7):
    deceduti = data.deceduti.diff().rolling(7).mean()
    positivi_shift = translate(data.nuovi_positivi.rolling(7).mean(), offset) / 100 * 1.3
    fig = make_subplots(1, 1, subplot_titles=["Confronto fra deceduti e l'1.3% dei nuovi casi ritardati di 7 giorni"])
    fig.add_trace(go.Line(
        x=positivi_shift.index, y=positivi_shift.values,
        mode='lines',
        name='1.3% dei nuovi casi spostati di 7 giorni',
        line={'dash': 'dot'},
        fill='tozeroy',
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


@st.cache(show_spinner=False)
def mobility_data(mobility_plot_data, variable, variables):
    fig = make_subplots(1, subplot_titles=[variable], specs=[[{"secondary_y": True}]])
    PALETTE = itertools.cycle(plotly.colors.qualitative.Plotly) #get_matplotlib_cmap('tab10', bins=8))
    for v in variables:
        if v == variable:
            break
        else:
            next(PALETTE)
    ax = go.Scatter(x=mobility_plot_data.rolling(7).mean().index,
                    y=getattr(mobility_plot_data, variable).rolling(7).mean(), fill='tozeroy',
                    name='', marker=dict(color=next(PALETTE)))
    fig.add_trace(ax, secondary_y=True)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        yaxis_title='',
        height=500,
        hovermode="x unified",
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
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', secondary_y=True)

    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_vaccines(vaccines, area=None, unita=UNITA, subplot_title='Dosi somministrate per 100 mila abitanti', fill=None, height=500):
    fig = make_subplots(1, subplot_titles=[subplot_title])
    maxs = []
    if area is None:
        for a in np.unique(vaccines.area):
            data_plot = vaccines[vaccines.area == a]
            total = (data_plot.sesso_maschile + data_plot.sesso_femminile).cumsum() / data_plot.popolazione * unita
            ax = go.Scatter(
                x=data_plot.index,
                y=total,
                fill=fill,
                name=a,
                mode='lines+markers',
            )
            fig.add_trace(ax)
            maxs.append(total.max())
        fig.update_layout(legend={
            'font': dict(
                size=10,
            ),
        }
)
    else:
        plot_data = vaccines[vaccines.area == area]
        total = (plot_data.sesso_maschile + plot_data.sesso_femminile).cumsum() / plot_data.popolazione * unita
        ax = go.Scatter(
            x=plot_data.index,
            y=total,
            # fill='tonexty',
            name='Numero somministrazioni cumulate',
            mode='lines+markers',
        )
        bar = go.Bar(
            x=plot_data.index,
            y=(plot_data.sesso_maschile + plot_data.sesso_femminile) / plot_data.popolazione * unita,
            name='Numero somministrazioni',
        )
        maxs.append(total)
        fig.add_trace(ax)
        fig.add_trace(bar)
        fig.update_layout(legend={
            'orientation': "v",
            'yanchor': "bottom",
            # 'y': -.15, # bottom
            'y': .85,  # top
            'xanchor': "right",
            'x': .5,
        })
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50,  l=10, b=10, r=10),
        yaxis_title='',
        height=height,
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs) + .05])
    return fig


def fill_data(data):
    time = datetime.date.today()
    if data.index[-1] != time:
        index = import_data.pd.date_range(data.index[-1] + np.timedelta64(1, 'D'), time)
        return import_data.pd.concat([data, import_data.pd.Series(np.ones(len(index)) * data[-1], index)])
    else:
        return data


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_deliveries(deliveries, population, subplot_title, unita):
    times = []
    fornitori = []
    names = []
    population_list = []
    for fornitore in np.unique(deliveries.fornitore):
        fornitore_data = deliveries[deliveries.fornitore == fornitore]
        times.append(fornitore_data.index[-1])
        names.append(fornitore)
        fornitori.append(fornitore_data.numero_dosi)
        population_list.append(population)
    
    legend={
        'orientation': "v",
        'yanchor': "top",
        'y': 1,  # top
        'xanchor': "left",
        'x': 0,
    }
    return plot_fill(
        [fill_data(fornitore.cumsum()).resample('1D').fillna('bfill') for fornitore in fornitori],
        names, population_list=population_list, unita=unita, legend=legend, rolling=False,
        cumulate=False, subplot_title="Dosi di vaccino consegnate per 100 mila abitanti"
    )


@st.cache(allow_output_mutation=True, show_spinner=False)
def categories_timeseries(vaccines, area, cumulate=False):
    data_area = vaccines[vaccines.area == area]
    population = data_area.popolazione
    data_list = []
    populations = []
    names = []
    maxs = []
    for category in CATEGORIES:
        plot_data = getattr(data_area, category['name'])
        data_list.append(plot_data)
        maxs.append(plot_data.sum())
        populations.append(population)
        names.append(category['label'])
    order = np.argsort(maxs)[::-1]
    legend = {
        'orientation': "v",
        'yanchor': "bottom",
        'y': .3,  # top
        'xanchor': "left",
        'x': .0,
    }
    return plot_fill([data_list[i] for i in order], [names[i] for i in order], cumulate=cumulate, subplot_title=f'{area}: Somministrazioni vaccino per categoria', legend=legend)


def sum_doses(data):
    return data.prima_dose + data.seconda_dose


@st.cache(allow_output_mutation=True, show_spinner=False)
def fornitori_timeseries(vaccines, area, cumulate=False):
    plot_data = {}
    if area == 'Italia':
        for fornitore in np.unique(vaccines.fornitore.values.astype(str)):
            if fornitore != 'nan':
                plot_data[fornitore] = sum_doses(vaccines[vaccines.fornitore == fornitore].groupby('data_somministrazione').sum())
    else:
        region = vaccines[vaccines.area == area]
        for fornitore in np.unique(region.fornitore.values.astype(str)):
            plot_data[fornitore] = sum_doses(region[region.fornitore == fornitore].groupby('data_somministrazione').sum())
    legend = {
        'orientation': "v",
        'yanchor': "bottom",
        'y': .75,  # top
        'xanchor': "left",
        'x': .0,
    }
    return plot_fill(plot_data.values(), plot_data.keys(), subplot_title=f"{area}: Somministrazioni vaccino per fornitore", cumulate=cumulate, legend=legend)


def age_timeseries(vaccines, area, fascia_anagrafica, demography, dose, unita=100, cumulate=False):
    PALETTE = itertools.cycle(plotly.colors.qualitative.Plotly)#get_default_palette()
    title = f'{area}: Percentuale popolazione che ha ricevuto<br>la {dose} nella fascia {fascia_anagrafica}'
    dose = dose.replace(' ', '_')
    if area == 'Italia':
        plot_data = vaccines[vaccines.fascia_anagrafica == fascia_anagrafica]
    else:
        region = vaccines[vaccines.area == area]
        plot_data = region[region.fascia_anagrafica == fascia_anagrafica]
    def cum(data):
        if cumulate:
            return data.cumsum()
        else:
            return data
    fig = make_subplots(1, subplot_titles=[title], specs=[[{"secondary_y": True}]])
    maxs_perc = []
    for fornitore in np.unique(sorted(plot_data.fornitore)):
        fornitore_data = cum(getattr(plot_data[plot_data.fornitore == fornitore], dose).groupby('data_somministrazione').sum())
        percentage = fornitore_data / demography[area].loc[fascia_anagrafica] * unita
        bar_perc = go.Bar(
            x=percentage.index,
            y=percentage,
            name=f'{fornitore}',
            legendgroup=f'{fornitore}',
            marker_color=next(PALETTE)
        )
        fig.add_trace(bar_perc, secondary_y=True)
        maxs_perc.append(percentage.max())
    total = cum(getattr(plot_data.groupby('data_somministrazione').sum().rolling(7).mean(), dose))
    total_perc = total / demography[area].loc[fascia_anagrafica] * unita
    ax_perc = go.Scatter(
        x=total_perc.index,
        y=total_perc,
        name='Media su 7 giorni',
        legendgroup='Numero dosi consegnate cumulate',
        marker=dict(color=next(PALETTE))
    )
    fig.add_trace(ax_perc, secondary_y=True)
    fig.update_layout(
        legend={
            'orientation': "v",
            'yanchor': "bottom",
            'y': .70,  # top
            'xanchor': "left",
            'x': .0,
        }, barmode='stack', hovermode="x unified"
    )
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50,  l=10, b=10, r=10),
        yaxis_title='',
        height=500,
        autosize=True,
    )
    if unita == 100:
        primary_title = 'Percentuale'
    else:
        primary_title = f'Dati per {unita:,} abitanti'
    start = total.index[0]
    max_perc = sum(maxs_perc) + sum(maxs_perc) * .1
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[start, total.index[-1] + np.timedelta64(1, 'D')])
    fig.update_yaxes(showgrid=True, title_text=primary_title, gridwidth=1, gridcolor='LightGrey', range=[0, max_perc], secondary_y=True)
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def ages_timeseries(vaccines, area, cumulate=False):
    if area == 'Italia':
        child = vaccines[vaccines.fascia_anagrafica == '16-19'].groupby('data_somministrazione').sum()
        twentys = vaccines[vaccines.fascia_anagrafica == '20-29'].groupby('data_somministrazione').sum()
        thirtys = vaccines[vaccines.fascia_anagrafica == '30-39'].groupby('data_somministrazione').sum()
        fortys = vaccines[vaccines.fascia_anagrafica == '40-49'].groupby('data_somministrazione').sum()
        fiftys = vaccines[vaccines.fascia_anagrafica == '50-59'].groupby('data_somministrazione').sum()
        sixtys = vaccines[vaccines.fascia_anagrafica == '60-69'].groupby('data_somministrazione').sum()
        seventys = vaccines[vaccines.fascia_anagrafica == '70-79'].groupby('data_somministrazione').sum()
        eightys = vaccines[vaccines.fascia_anagrafica == '80-89'].groupby('data_somministrazione').sum()
        nineties = vaccines[vaccines.fascia_anagrafica == '90+'].groupby('data_somministrazione').sum()
    else:
        region = vaccines[vaccines.area == area]
        child = region[region.fascia_anagrafica == '16-19'].groupby('data_somministrazione').sum()
        twentys = region[region.fascia_anagrafica == '20-29'].groupby('data_somministrazione').sum()
        thirtys = region[region.fascia_anagrafica == '30-39'].groupby('data_somministrazione').sum()
        fortys = region[region.fascia_anagrafica == '40-49'].groupby('data_somministrazione').sum()
        fiftys = region[region.fascia_anagrafica == '50-59'].groupby('data_somministrazione').sum()
        sixtys = region[region.fascia_anagrafica == '60-69'].groupby('data_somministrazione').sum()
        seventys = region[region.fascia_anagrafica == '70-79'].groupby('data_somministrazione').sum()
        eightys = region[region.fascia_anagrafica == '80-89'].groupby('data_somministrazione').sum()
        nineties = region[region.fascia_anagrafica == '90+'].groupby('data_somministrazione').sum()
    data_list = [
        sum_doses(child),
        sum_doses(twentys),
        sum_doses(thirtys),
        sum_doses(fortys),
        sum_doses(fiftys),
        sum_doses(sixtys),
        sum_doses(seventys),
        sum_doses(eightys),
        sum_doses(nineties),
    ]
    names = ['16-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    legend = {
        'orientation': "v",
        'yanchor': "bottom",
        'y': .35,  # top
        'xanchor': "left",
        'x': 0,
    }
    return plot_fill(data_list, names, subplot_title=f"{area}: Somministrazioni vaccino per fascia d'età", cumulate=cumulate, legend=legend)


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_ages(vaccines, area):
    if area == 'Italia':
        url = "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/anagrafica-vaccini-summary-latest.csv"
        s=requests.get(url).content
        ages = import_data.pd.read_csv(io.StringIO(s.decode('utf-8')), index_col='fascia_anagrafica')
    else:
        vaccines_area = vaccines[vaccines.area == area]
        ages = vaccines_area.groupby('fascia_anagrafica').sum()
    pie = go.Pie(
        values=(ages.sesso_maschile + ages.sesso_femminile).values,
        labels=ages.index,
        textinfo='label+percent',
    )
    fig = make_subplots(1, subplot_titles=[f'Fasce di età di somministrazione'])
    fig.add_trace(pie)
    fig.update_layout(legend={
        'orientation': 'h',
        'yanchor': "bottom",
        'y': -.35,  # top
        'xanchor': "center",
        'x': .5,
    }, height=500,
    )
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_second_dose_percentage(vaccines, area):
    plot_data = vaccines[vaccines.area == area].sum()
    not_prima_dose = plot_data.prima_dose - plot_data.seconda_dose
    pie = go.Pie(
        values=[not_prima_dose, plot_data.seconda_dose],
        labels=['Seconda dose ancora non somministrata', 'Seconda dose'],
        textinfo='percent'
    )
    fig = make_subplots(1, subplot_titles=[f'Somministrazione seconda dose'])
    fig.add_trace(pie)
    fig.update_layout(legend={
        'orientation': 'h',
        'yanchor': "bottom",
        'y': -.2,  # top
        'xanchor': "center",
        'x': .5,
    }, height=440
    )
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_category(vaccines, area):
    plot_data = vaccines[vaccines.area == area].sum()
    pie = go.Pie(
        values=[getattr(plot_data, category['name']) for category in CATEGORIES],
        labels=[category['label'] for category in CATEGORIES],
        textinfo='percent',
    )
    fig = make_subplots(1, subplot_titles=[f'Somministrazione per categorie'])
    fig.add_trace(pie)
    fig.update_layout(legend={
        'orientation': 'v',
        'yanchor': "bottom",
        'y': -.55,  # top
        'xanchor': "center",
        'x': .5,
        }, height=600
)
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_fornitore(vaccines, area):
    plot_data = vaccines[vaccines.area == area].groupby('fornitore').sum()
    pie = go.Pie(
        values=[plot_data.numero_dosi[fornitore] for fornitore in sorted(plot_data.index)],
        labels=sorted(plot_data.index),
        textinfo='percent',
    )
    fig = make_subplots(1, subplot_titles=[f'Fornitore dosi vaccino'])
    fig.add_trace(pie)
    fig.update_layout(legend={
        'orientation': 'h',
        'yanchor': "bottom",
        'y': -.3,  # top
        'xanchor': "center",
        'x': .5,
    }, height=500
    )
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_vaccine_popolation(vaccines, area):
    vaccines_area = vaccines[vaccines.area == area]
    vaccines_pop = vaccines_area.seconda_dose.cumsum().iloc[-1]
    pie = go.Pie(
        values=[vaccines_pop, vaccines_area.popolazione.iloc[-1] - vaccines_pop],
        labels=['Popolazione vaccinata',  'Popolazione non vaccinata'],
        textinfo='percent',
    )
    fig = make_subplots(1)
    fig.add_trace(pie)
    fig.update({'layout_showlegend': False})
    fig.update_layout(legend={
        'yanchor': "bottom",
        'y': -.3,  # top
        'xanchor': "center",
        'x': .5,
    }, height=150, margin=dict(
        l=0,
        r=0,
        b=0,
        t=30,
        pad=0
    ),
    )
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_percentage(vaccines, deliveries, area):
    vaccines_area = vaccines[vaccines.area == area]
    tot_vaccines = (vaccines_area.sesso_maschile + vaccines_area.sesso_femminile).cumsum().iloc[-1]
    tot_deliveries = deliveries[deliveries.area == area].numero_dosi.cumsum().iloc[-1]
    pie = go.Pie(
        values=[tot_vaccines, (tot_deliveries - tot_vaccines)],
        labels=['Dosi somministrate',  'Dosi ancora non somministrate'],
        textinfo='percent',
    )
    fig = make_subplots(1)
    fig.add_trace(pie)
    fig.update({'layout_showlegend': False})
    fig.update_layout(legend={
        'yanchor': "bottom",
        'y': -.3,  # top
        'xanchor': "center",
        'x': .5,
    }, height=150, margin=dict(
        l=0,
        r=0,
        b=0,
        t=30,
        pad=0
    ),
    )
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_vaccines_prediction(vaccines, area, npoints=7, p0=(np.datetime64("2021-01-01", "s"), np.timedelta64(48 * 60 * 60, "s"))):
    plot_data = vaccines[vaccines.area == area]
    popolazione = plot_data.popolazione
    plot_data = (plot_data.sesso_femminile + plot_data.sesso_maschile).cumsum() / popolazione * 100
    t_0, T_d, r2 = linear_fit(plot_data, start=-npoints, stop=-1, p0=p0)
    t100 = (70 * T_d) + t_0
    subplot_title = f'Previsione vaccinazione 70% della popolazione: {import_data.pd.to_datetime(t100).date()}<br>' \
                    f'Fit sugli ultimi {npoints} giorni'
    fig = plot_vaccines(vaccines, area, unita=100, fill='tozeroy', subplot_title=subplot_title)
    ax = go.Scatter(
        x=import_data.pd.date_range(plot_data.index[0], t100),
        y=linear(import_data.pd.date_range(plot_data.index[0], t100), t_0, T_d),
        name='Fit',
        mode='lines',
    )
    fig.add_trace(ax)
    fig.update_layout(legend={
        'orientation': 'v',
        'yanchor': "bottom",
        'y': .9,  # top
        'xanchor': "center",
        'x': 0.1,
    }
    )
    return fig



def plot_fill(data_list, names, population_list=None, unita=100000, cumulate=True, rolling=True, subplot_title='', start=None, height=500, legend=None):
    PALETTE = itertools.cycle(plotly.colors.qualitative.Plotly)#get_default_palette()
    PALETTE_ALPHA = itertools.cycle(plotly.colors.qualitative.Plotly)#get_default_palette()
    fig = make_subplots(1, subplot_titles=[subplot_title], specs=[[{"secondary_y": True}]])
    maxs_perc = []
    maxs_tot = []
    min_date = []
    if not population_list:
        population_list = [None,] * len(data_list)
        primary_grid = True
    else:
        primary_grid = False
    for i, (data, population, name) in enumerate(zip(data_list, population_list, names)):
        if cumulate:
            data = data.cumsum()
        else:
            if rolling:
                data = data.rolling(7).mean()
        if population is not None:
            percentage_cumsum = data / population * unita
            ax_perc = go.Scatter(
                x=percentage_cumsum.index,
                y=percentage_cumsum,
                # mode='lines+markers',
                showlegend=False,
                stackgroup='One',
                hoverinfo='skip',
                legendgroup=name,
                marker=dict(color=next(PALETTE))
            )
            maxs_perc.append(percentage_cumsum.max())
            fig.add_trace(ax_perc)
        ax_tot = go.Scatter(
            x=data.index,
            y=data,
            name=f'{name}',
            # mode='lines+markers',
            showlegend=True if name else False,
            stackgroup='Two',
            legendgroup=name,
            marker=dict(color=next(PALETTE_ALPHA))
        )
        maxs_tot.append(data.max())
        fig.add_trace(ax_tot, secondary_y=True)
        min_date.append(data.index[0])
    if not legend:
        legend = {
            'orientation': "v",
            'yanchor': "bottom",
            'y': .85,  # top
            'xanchor': "left",
            'x': .0,
        }
    fig.update_layout(
        legend=legend
    )
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50,  l=10, b=10, r=10),
        yaxis_title='',
        height=height,
        autosize=True,
        hovermode="x unified",
    )
    if unita == 100:
        primary_title = 'Percentuale'
    else:
        primary_title = f'Dati per {unita:,} abitanti'
    if not start:
        start = min(min_date)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[start, data.index[-1] + np.timedelta64(1, 'D')])
    fig.update_yaxes(showgrid=True, title_text=primary_title, gridwidth=1, gridcolor='LightGrey') #, range=[0, max(maxs_perc)])
    fig.update_yaxes(showgrid=primary_grid, title_text='Totale', gridwidth=1, gridcolor='LightGrey', secondary_y=True)#, range=[0, max(maxs_tot)])
    return fig


# @st.cache(allow_output_mutation=True, show_spinner=False)
# def plot_deliveries(deliveries, population, unita=100, subplot_title='', start=None, height=500):
#     PALETTE_1 = get_default_palette()  # get_default_palette()
#     PALETTE_2 = get_default_palette()  # get_default_palette()
#     fig = make_subplots(1, subplot_titles=[subplot_title], specs=[[{"secondary_y": True}]])
#     maxs_perc = []
#     maxs_tot = []
#     cumsum = deliveries.numero_dosi.groupby('data_consegna').sum().cumsum()
#     percentage_cumsum = cumsum / population * unita
#     for fornitore in np.unique(sorted(deliveries.fornitore)):
#         fornitore_data = deliveries[deliveries.fornitore == fornitore].numero_dosi
#         bar_tot = go.Bar(
#             x=fornitore_data.index,
#             y=fornitore_data,
#             name=f'{fornitore}',
#             showlegend=False,
#             legendgroup=f'{fornitore}',
#             marker_color=next(PALETTE_1)
#         )
#         percentage = fornitore_data / population * unita
#         bar_perc = go.Bar(
#             x=percentage.index,
#             y=percentage,
#             name=f'{fornitore}',
#             hoverinfo='skip',
#             legendgroup=f'{fornitore}',
#             marker_color=next(PALETTE_2)
#         )
#         fig.add_trace(bar_perc)
#         fig.add_trace(bar_tot, secondary_y=True)
#     ax_perc = go.Scatter(
#         x=percentage_cumsum.index,
#         y=percentage_cumsum,
#         name='Cumulate',
#         mode='lines+markers',
#         hoverinfo='skip',
#         legendgroup='Numero dosi consegnate cumulate',
#         marker=dict(color=next(PALETTE_1))
#     )
#     ax_tot = go.Scatter(
#         x=cumsum.index,
#         y=cumsum,
#         name='Cumulate',
#         mode='lines+markers',
#         showlegend=False,
#         legendgroup='Numero dosi consegnate cumulate',
#         marker=dict(color=next(PALETTE_2))
#     )
#     fig.add_trace(ax_perc)
#     fig.add_trace(ax_tot, secondary_y=True)
#     maxs_perc.append(percentage_cumsum.max())
#     maxs_tot.append(cumsum.max())
#     fig.update_layout(
#         legend={
#             'orientation': "v",
#             'yanchor': "bottom",
#             'y': .70,  # top
#             'xanchor': "left",
#             'x': .0,
#         }, barmode='stack', hovermode="x unified"
#     )
#     fig.update_layout(
#         plot_bgcolor="white",
#         margin=dict(t=50,  l=10, b=10, r=10),
#         yaxis_title='',
#         height=height,
#         autosize=True,
#     )
#     if unita == 100:
#         primary_title = 'Percentuale'
#     else:
#         primary_title = f'Dati per {unita:,} abitanti'
#     if not start:
#         start = cumsum.index[0]
#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[start, cumsum.index[-1] + np.timedelta64(1, 'D')])
#     fig.update_yaxes(showgrid=True, title_text=primary_title, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs_perc)])
#     fig.update_yaxes(showgrid=False, title_text='Totale', gridwidth=1, gridcolor='LightGrey', secondary_y=True, range=[0, max(maxs_tot)])
#     return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def cumulate_and_not(data_list, names, population_list, unita=100, subplot_title='', start=None, height=500):
    PALETTE = get_default_palette()
    PALETTE_ALPHA = get_default_palette(True)
    fig = make_subplots(1, subplot_titles=[subplot_title], specs=[[{"secondary_y": True}]])
    maxs_perc = []
    maxs_tot = []
    for data, population, name in zip(data_list, population_list, names):
        cumsum = data.cumsum()
        percentage = data / population * unita
        percentage_cumsum = data.cumsum() / population * unita
        ax_perc = go.Scatter(
            x=percentage_cumsum.index,
            y=percentage_cumsum,
            name=f'{name} cumulate',
            mode='lines+markers',
            legendgroup=name + 'cumulate',
            marker=dict(color=next(PALETTE))
        )
        ax_tot = go.Scatter(
            x=cumsum.index,
            y=cumsum,
            name=f'{name} cumulate',
            mode='lines+markers',
            showlegend=False,
            legendgroup=name + 'cumulate',
            marker=dict(color=next(PALETTE_ALPHA))
        )
        bar_tot = go.Bar(
            x=data.index,
            y=data,
            name=f'{name}',
            showlegend=False,
            legendgroup=name,
            marker_color=next(PALETTE_ALPHA)
        )
        bar_perc = go.Bar(
            x=percentage.index,
            y=percentage,
            name=f'{name}',
            legendgroup=name,
            marker_color=next(PALETTE)
        )
        maxs_perc.append(percentage_cumsum.max())
        maxs_tot.append(cumsum.max())
        fig.add_trace(ax_perc)
        fig.add_trace(bar_perc)
        fig.add_trace(ax_tot, secondary_y=True)
        fig.add_trace(bar_tot, secondary_y=True)
    fig.update_layout(
        legend={
            'orientation': "v",
            'yanchor': "bottom",
            'y': .85,  # top
            'xanchor': "right",
            'x': .5,
        }
    )
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50,  l=10, b=10, r=10),
        yaxis_title='',
        height=height,
        autosize=True,
    )
    if unita == 100:
        primary_title = 'Percentuale'
    else:
        primary_title = f'Dati per {unita:,} abitanti'
    # if not start:
    #     start = data.index[0]
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')#, range=[start, data.index[-1]])
    fig.update_yaxes(showgrid=True, title_text=primary_title, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs_perc)])
    fig.update_yaxes(showgrid=False, title_text='Totale', gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs_tot)], secondary_y=True)
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def plot_variables(data_list, names, rule=None, popolazione=None, title='', yaxis_title='', xaxes_range=None, secondary_y=True, nrows=1):
    PALETTE = get_default_palette()
    PALETTE_ALPHA = get_default_palette(True)

    maxs = []
    mins = []

    fig = make_subplots(1, 1, subplot_titles=[title], specs=[[{"secondary_y": True}]])
    for data, name in zip(data_list, names):
        if rule is not None:
            plot_data = normalisation(data, popolazione, rule)
        else:
            plot_data = data
        plot_data = plot_data[~np.isnan(plot_data)]
        maxs.append(plot_data.max())
        mins.append(plot_data.min())
        plot_average(
            plot_data,
            fig=fig,
            name=name,
            palette=PALETTE,
            palette_alpha=PALETTE_ALPHA,
            secondary_y=secondary_y,
        )

        maximum = np.nanmax(maxs)
        minimum = np.nanmin(mins)
    yscale = 'linear'
    if xaxes_range is None:
        xaxes_range = [plot_data.index[0], plot_data.index[-1] + np.timedelta64(1, 'D')]
    fig.update_xaxes(row=1, col=1, showgrid=True, gridwidth=1, gridcolor='LightPink', range=xaxes_range)
    fig.update_yaxes(row=1, col=1, type=yscale, showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[minimum, maximum], secondary_y=secondary_y)#, showexponent='all', exponentformat='power')
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=70, l=0, b=0, r=0),
        yaxis_title=yaxis_title,
        height=500,
        autosize=True,
        legend={
            'orientation': "h",
            'yanchor': "bottom",
            'y': .9, # top
            'xanchor': "center",
            'x': .5,
        }
    )
    return fig
