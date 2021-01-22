
import datetime
import itertools
import numpy as np
import scipy.optimize
import scipy.stats
from matplotlib import cm

import import_data

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        name=rule,
        mode='lines+markers',
        showlegend=False,
        legendgroup='postam',
        marker=dict(color=next(PALETTE)),
        fill='tozeroy',
    ), 1, 1)
    fig.add_trace(go.Scatter(
        x=plot_data_points.index,
        y=plot_data_points.values,
        mode='markers',
        legendgroup='postam',
        showlegend=False,
        marker=dict(color=next(PALETTE_ALPHA)),
    ), 1, 1)
    # fig.add_annotation(x=plot_data.index[-1], y=plot_data.values[-1], text='{:.2f}'.format(plot_data.values[-1]))
    # fig.add_annotation(x=plot_data.index[-1], y=plot_data.rolling(7).mean().values[-1], text='{:.2f}'.format(plot_data.rolling(7).mean().values[-1]))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, plot_data[-20:].max() + 1])
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=30, l=10, b=10, r=10),
        autosize=True,
        height=500,
    )
    return fig


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


def plot_average(plot_data, palette, fig, name, palette_alpha, fmt, start=None, stop=None, log=True):
    line = dict(color=next(palette))
    fig.add_trace(go.Line(
        x=plot_data.index, y=plot_data.rolling(7).mean().values,
        name=name,
        legendgroup=name,
        line=line,
        mode='lines',
    ), 1, 1)
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
        showlegend=False,
        marker=dict(color=next(palette_alpha))
    ), 1, 1)
    # if log is True:
    #     y = np.log10(plot_data.values[-1])
    # else:
    #     y = plot_data.values[-1]
    # fig.add_annotation(
    #     x=plot_data.index[-1],
    #     y=y,
        # text=fmt.format(plot_data.values[-1])
    # )


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
        fig.add_trace(go.Scatter(x=plot_data.index[-90:], y=plot_data.values[-90:], showlegend=False,
                                 name=title, marker=dict(color=next(PALETTE)), fill='tozeroy'), row, col)
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs)])
    fig.update_layout(
        title=title,
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


def mobility_data(mobility_plot_data, variable):
    fig = make_subplots(1, subplot_titles=[variable])
    ax = go.Scatter(x=mobility_plot_data.index,
                    y=getattr(mobility_plot_data, variable), fill='tozeroy',
                    name='')
    fig.add_trace(ax)
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


def plot_deliveries(deliveries, area):
    fig = make_subplots(1, subplot_titles=[f'Dosi di vaccino consegnate per 100 mila abitanti'])
    plot_data = deliveries[deliveries.area == area]
    ax = go.Scatter(
        x=plot_data.index,
        y=plot_data.numero_dosi.cumsum() / plot_data.popolazione * UNITA,
        # fill='tozeroy',
        name='Numero dosi consegnate cumulate',
        mode='lines+markers',
    )
    bar = go.Bar(
        x=plot_data.index,
        y=plot_data.numero_dosi / plot_data.popolazione * UNITA,
        name='Numero dosi consegnate',
    )
    fig.add_trace(ax)
    fig.add_trace(bar)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        yaxis_title='',
        height=500,
        autosize=True,
        legend={
            'orientation': "v",
            'yanchor': "bottom",
            # 'y': -.15, # bottom
            'y': .85,  # top
            'xanchor': "right",
            'x': .5,
        }
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    return fig


def plot_ages(vaccines, area):
    if area == 'Italia':
        ages = import_data.pd.read_csv(
            import_data.os.path.join(import_data.BASE_PATH, 'covid19-opendata-vaccini/dati/anagrafica-vaccini-summary-latest.csv'),
            index_col='fascia_anagrafica',
        )
    else:
        vaccines_area = vaccines[vaccines.area == area]
        ages = vaccines_area.groupby('fascia_anagrafica').sum()
    pie = go.Pie(
        values=(ages.sesso_maschile + ages.sesso_femminile).values,
        labels=ages.index,
        textinfo='label+percent'
    )
    fig = make_subplots(1, subplot_titles=[f'Fasce di età di somministrazione'])
    fig.add_trace(pie)
    fig.update_layout(legend={
        'orientation': 'h',
        'yanchor': "bottom",
        'y': -.3,  # top
        'xanchor': "center",
        'x': .5,
    }
    )
    return fig


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
        'y': -.15,  # top
        'xanchor': "center",
        'x': .5,
    }
    )
    return fig


def plot_category(vaccines, area):
    plot_data = vaccines[vaccines.area == area].sum()
    pie = go.Pie(
        values=[
            plot_data.categoria_operatori_sanitari_sociosanitari,
            plot_data.categoria_personale_non_sanitario,
            plot_data.categoria_ospiti_rsa,
            plot_data.categoria_over80,
        ],
        labels=[
            'Categoria operatori sanitari sociosanitari',
            'Categoria personale non sanitario',
            'Categoria ospiti RSA',
            'Categoria over 80',
        ],
        textinfo='percent',
    )
    fig = make_subplots(1, subplot_titles=[f'Somministrazione per categorie'])
    fig.add_trace(pie)
    fig.update_layout(legend={
            'yanchor': "bottom",
            'y': -.3,  # top
            'xanchor': "center",
            'x': .5,
        }
)
    return fig


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


def second_dose(vaccines, area=None, unita=100, subplot_title='Percentuale popolazione vaccinata (seconda dose somministrata)', fill=None, height=500):
    fig = make_subplots(1, subplot_titles=[subplot_title])
    maxs = []
    if area is None:
        for a in np.unique(vaccines.area):
            plot_data = vaccines[vaccines.area == a]
            total = plot_data.seconda_dose.cumsum() / plot_data.popolazione * unita
            ax = go.Scatter(
                x=plot_data.index,
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
        total = plot_data.seconda_dose.cumsum() / plot_data.popolazione * unita
        ax = go.Scatter(
            x=plot_data.index,
            y=total,
            # fill='tonexty',
            name='Vaccinazioni cumulate',
            mode='lines+markers',
        )
        bar = go.Bar(
            x=plot_data.index,
            y=plot_data.seconda_dose / plot_data.popolazione * unita,
            name='Vaccinazioni',
        )
        maxs.append(total)
        fig.add_trace(ax)
        fig.add_trace(bar)
        fig.update_layout(legend={
            'orientation': "h",
            'yanchor': "bottom",
            # 'y': -.15, # bottom
            'y': .9,  # top
            'xanchor': "center",
            'x': .5,
        })
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50,  l=10, b=10, r=10),
        yaxis_title='',
        height=height,
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[np.datetime64('2021-01-16'), plot_data.index[-1]])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs) + .01])
    return fig


