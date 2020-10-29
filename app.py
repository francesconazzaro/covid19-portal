import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 16

import datetime
import pandas as pd
import itertools
import os
import scipy.optimize
import scipy.stats
import numpy as np
from glob import glob


UNITA = 100000


def exp2(t, t_0, T_d):
    return 2 ** ((t - t_0) / T_d)


def linear(t, t_0, T_d):
    return (t - t_0) / T_d


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
    x_list = []
    label_list = []
    for event in events[start:stop]:
        label = '{x} {label}'.format(offset=offset, **event)
        fig.add_trace(go.Scatter(x=[event['x'], event['x']], y=[0, 10**10], mode='lines', legendgroup='events', line=dict(color=next(PALETTE), width=2), name=label))


P0 = (np.datetime64("2020-02-12", "s"), np.timedelta64(48 * 60 * 60, "s"))
def linear_fit(data, start=None, stop=None, p0=P0):
    t_0_guess, T_d_guess = p0
    data_fit = data[start:stop]
    x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
    y_fit = data_fit.values

    t_fit = data_fit.index.values
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
        label = 'Fit {label} (T_d = {0:.2f})'.format(td / np.timedelta64(1, "D"), label=label)
    x = data[start:stop].index
    x_base = np.arange(x[0] - np.timedelta64(shift, 'D'), x[-1] + np.timedelta64(shift, 'D'), np.timedelta64(1, 'D'))
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
    contour_level_list = []
    contour_colour_list = []

    for k in range(bins):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))

    C = list(map(np.uint8, np.array(cmap(bins * h)[:3]) * 255))
    contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))
    return contour_colour_list


def import_population():
    import io
    popolazione_regioni = pd.read_csv(io.StringIO(""""regione","Età","Maschicelibi","Maschiconiugati","Maschidivorziati","Maschivedovi","TotaleMaschi","Femminenubili","Femmineconiugate","Femminedivorziate","Femminevedove","TotaleFemmine"
    Piemonte,Totale,961689,1037848,67357,62509,2129403,809839,1052873,96774,303637,2263123
    ValledAosta,Totale,30565,27466,2316,1629,61976,25981,27505,3183,8238,64907
    Lombardia,Totale,2325702,2326127,123952,118582,4894363,1993086,2354127,176807,600783,5124803
    TrentinoAltoAdige,Totale,270190,227847,14066,10495,522598,235864,229418,18640,56340,540262
    Veneto,Totale,1120357,1160940,57666,55604,2394567,960361,1175062,78773,298766,2512962
    FriuliVeneziaGiulia,Totale,268181,284558,20210,16802,589751,222707,288104,28045,89265,628121
    Liguria,Totale,334317,359828,27658,24549,746352,290172,366777,39500,122506,818955
    EmiliaRomagna,Totale,1023234,1011556,62211,60285,2157286,893827,1025469,88598,283661,2291555
    Toscana,Totale,824989,875839,49846,52210,1802884,726340,890705,72094,250414,1939553
    Umbria,Totale,191686,214378,8918,11977,426959,169452,217119,13381,61997,461949
    Marche,Totale,343304,368041,15148,18993,745486,293177,373067,21255,105070,792569
    Lazio,Totale,1352807,1358535,66050,69484,2846876,1214000,1386278,111590,339380,3051248
    Abruzzo,Totale,289180,326808,11215,16495,643698,245399,328783,15878,88489,678549
    Molise,Totale,68757,77636,1989,4395,152777,56440,76790,2619,21823,157672
    Campania,Totale,1332996,1414011,27102,69130,2843239,1190010,1439647,48064,318124,2995845
    Puglia,Totale,882357,1018982,23583,49235,1974157,798074,1028789,36194,226674,2089731
    Basilicata,Totale,126550,142327,3437,7862,280176,106513,141499,4671,37506,290189
    Calabria,Totale,445585,483425,9793,23535,962338,381021,485468,15167,121134,1002790
    Sicilia,Totale,1126909,1243950,30389,58200,2459448,995498,1249199,48221,304275,2597193
    Sardegna,Totale,397525,379494,15721,18667,811407,343246,380381,20088,98013,841728"""), index_col='regione')
    popolazione = popolazione_regioni.TotaleMaschi + popolazione_regioni.TotaleFemmine
    return popolazione


class FileReference:
    def __init__(self):
        base_path = '/app'
        self.filename_regioni = os.path.join(base_path, 'COVID-19/dati-regioni/')
        self.filename_ita = os.path.join(base_path, 'COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')


def hash_file_reference(file_reference):
    filename_regioni = file_reference.filename_regioni
    filename_ita = file_reference.filename_ita
    return (filename_regioni, filename_ita, os.path.getmtime(filename_regioni), os.path.getmtime(filename_ita))


@st.cache(hash_funcs={FileReference: hash_file_reference})
def import_data(base_path='/app'):
    popolazione = import_population()
    # data_raw = []
    # for path in glob(os.path.join(base_path, 'COVID-19/dati-regioni/dpc-covid19-ita-regioni-*.csv')):
    #     data_raw.append(pd.read_csv(path, parse_dates=['data']).set_index('data'))
    # #    data_raw.append(pd.read_csv(path))
    # data = pd.concat(data_raw)
    if not os._exists('COVID-19'):
        import git
        git.Git("./").clone("https://github.com/pcm-dpc/COVID-19.git")
    data_aggregate = pd.read_csv(os.path.join(base_path, 'COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv'), index_col='data', parse_dates=['data'])

    regioni = {}
    ita = pd.read_csv(os.path.join(base_path, 'COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'), index_col='data', parse_dates=['data'])
    ita['popolazione'] = popolazione.sum()
    regioni['Italia'] = ita
    for regione in np.unique(data_aggregate.denominazione_regione):
        try:
            popolazione_index = [index for index in popolazione.index if regione[:4].lower() in index.lower()][0]
        except:
            print(regione)
        data_in = data_aggregate[data_aggregate.denominazione_regione == regione].sort_index()
        data_in['popolazione'] = popolazione[popolazione_index]
        regioni[regione] = data_in
    italy_minus = []
    for regione in regioni.keys():
        #if regione != 'Lombardia':
        italy_minus.append(regioni[regione])
    italy_minus = pd.concat(italy_minus)
    return regioni




def normalisation(data, population, rule):
    if rule == 'per 100.000 abitanti':
        new_data = data / population * UNITA
        new_data.name = data.name
        return new_data
    else:
        new_data = data
        new_data.name = data.name
        return new_data

def get_fmt(rule):
    if rule == 'per 100.000 abitanti':
        return '{:.2f}'
    else:
        return '{:.0f}'


def plot_selection(data, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri):

    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    PALETTE_ALPHA = itertools.cycle(get_matplotlib_cmap('tab10', bins=8, alpha=.3))

    data = data[country]

    fig = make_subplots(1, 1)
    fmt = get_fmt(rule)

    line = dict(color=next(PALETTE))
    plot_data = normalisation(data.nuovi_positivi, data.popolazione, rule)
    fig.add_trace(go.Line(
        x=plot_data.index, y=plot_data.rolling(7).mean().values,
        name='Nuovi Positivi',
        legendgroup='pos',
        line=line,
        mode='lines+markers'
    ), 1, 1)

    line['dash'] = 'dot'
    plot_fit(plot_data.rolling(7).mean(), fig, label='Nuovi Positivi', start=start_positivi, stop=stop_positivi, mode='lines', line=line, shift=5)

    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values, mode='markers', legendgroup='pos', showlegend=False, marker=dict(color=next(PALETTE_ALPHA))), 1, 1)
    fig.add_annotation(x=plot_data.index[-1], y=np.log10(plot_data.values[-1]), text=fmt.format(plot_data.values[-1]))

    line = dict(color=next(PALETTE))
    plot_data = normalisation(data.ricoverati_con_sintomi, data.popolazione, rule)
    fig.add_trace(go.Line(
        x=plot_data.index,
        y=plot_data.rolling(7).mean().values,
        name='Ricoveri',
        legendgroup='ric',
        line=line,
        mode='lines+markers'
    ), 1, 1)
    line['dash'] = 'dot'
    plot_fit(plot_data.rolling(7).mean(), fig, label='Ricoveri', start=start_ricoveri, stop=stop_ricoveri, mode='lines', line=line, shift=5)
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values, mode='markers', legendgroup='ric', showlegend=False, marker=dict(color=next(PALETTE_ALPHA))), 1, 1)

    line = dict(color=next(PALETTE))
    plot_data = normalisation(data.terapia_intensiva, data.popolazione, rule)
    fig.add_trace(go.Line(
        x=plot_data.index,
        y=plot_data.rolling(7).mean().values,
        name='Terapie Intensive',
        legendgroup='ti',
        line=line,
        mode='lines+markers'
    ), 1, 1)
    line['dash'] = 'dot'
    plot_fit(plot_data.rolling(7).mean(), fig, label='Terapie Intensive', start=start_ti, stop=stop_ti, mode='lines', line=line, shift=5)
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values, mode='markers', legendgroup='ti', showlegend=False, marker=dict(color=next(PALETTE_ALPHA))), 1, 1)
    fig.add_annotation(x=plot_data.index[-1], y=np.log10(plot_data.values[-1]), text=fmt.format(plot_data.values[-1]))

    line = dict(color=next(PALETTE))
    plot_data = normalisation(data.deceduti.diff(), data.popolazione, rule)
    fig.add_trace(go.Line(
        x=plot_data.index,
        y=plot_data.rolling(7).mean().values,
        name='Deceduti',
        legendgroup='dec',
        line=line,
        mode='lines+markers'
    ), 1, 1)
    # fig.add_trace(go.Line(x=plot_data.index, y=plot_data.rolling(7).mean().values, name='Deceduti', legendgroup='dec', marker=dict(color=next(PALETTE))), 1, 1)
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values, name='Deceduti', mode='markers', legendgroup='dec', showlegend=False, marker=dict(color=next(PALETTE_ALPHA))), 1, 1)
    fig.add_annotation(x=plot_data.index[-1], y=np.log10(plot_data.values[-1]), text=fmt.format(plot_data.values[-1]))

    add_events(fig)

    plot_data = normalisation(data.nuovi_positivi, data.popolazione, rule)
    fig.update_xaxes(row=1, col=1, showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(row=1, col=1, type="log", showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[-3, np.log10(plot_data.max()) + .5], showexponent='all', exponentformat='power')
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=30,l=10,b=10,r=10),
        yaxis_title=f'Dati {rule}',
        # width=1300,
        height=500,
        autosize=True,
    )
    return fig

def test_positivity_rate(data, country):
    data = data[country]
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    PALETTE_ALPHA = itertools.cycle(get_matplotlib_cmap('tab10', bins=8, alpha=.3))
    fig = make_subplots(1, 1, subplot_titles=['Percentuale di tamponi positivi'])
    plot_data = data.nuovi_positivi / data.tamponi.diff() * 100
    fig.add_trace(go.Line(
        x=plot_data.index,
        y=plot_data.rolling(7).mean().values,
        name='Percentuale tamponi positivi',
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
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=['2020-09-01', np.datetime64(datetime.datetime.now() + datetime.timedelta(days=1))])
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, plot_data[-20:].max() + 1])
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=30,l=10,b=10,r=10),
        # width=1300,
        # height=500,
        autosize=True,
    )
    return fig

def summary(data, what):
    titles = [title for title in data if title not in ['P.A. Bolzano', 'P.A. Trento']]
    fig = make_subplots(3, 7, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles, vertical_spacing=.08)
    minus = 0
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    maxs = []
    for i, name in enumerate(data):
        col = (i - minus) % 7 + 1
        row = (i - minus) // 7 + 1
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
            plot_data = region.nuovi_positivi.rolling(7).mean() / region.tamponi.diff().rolling(7).mean() * 100
            title = "Percentuale tamponi positivi."
            yscale = 'linear'
        maxs.append(plot_data.values[-90:].max())
        fig.add_trace(go.Line(x=plot_data.index[-90:], y=plot_data.values[-90:], showlegend=False, name=title, marker=dict(color=next(PALETTE))), row, col)
    st.subheader(title)
    fig.update_xaxes(showgrid=True, gridwidth=1, tickangle=45, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, max(maxs) + 1])
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50,l=10,b=10,r=10),
        # width=1300,
        height=500,
        autosize=True,
    )
    PALETTE = itertools.cycle(get_matplotlib_cmap('tab10', bins=8))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=15, color=next(PALETTE))
    return fig
        


st.beta_set_page_config(layout='wide')
data = import_data()
st.title('COVID-19: Situazione in Italia aggiornata al {}'.format(data['Italia'].index[-1].date()))
population = import_population()


def explore_regions():
    st.header('Dati regione')
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        country = st.selectbox('Seleziona una regione', list(data.keys()))
    with col2:
        rule = st.radio('', ['per 100.000 abitanti', 'Totali'])
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        start_positivi = st.date_input('Data inizio fit Nuovi positivi', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col2:
        start_ti = st.date_input('Data inizio fit Terapie intensive', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col3:
        start_ricoveri = st.date_input('Data inizio fit Ricoveri', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col1:
        stop_positivi = st.date_input('Data fine fit Nuovi positivi', data[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    with col2:
        stop_ti = st.date_input('Data fine fit Terapie intensive', data[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    with col3:
        stop_ricoveri = st.date_input('Data fine fit Ricoveri', data[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    st.plotly_chart(plot_selection(data, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri), use_container_width=True)
    st.plotly_chart(test_positivity_rate(data, country), use_container_width=True)
    st.subheader(f'Andamento degli ultimi 5 giorni: {country} ({rule})')
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.write(normalisation(data[country].nuovi_positivi, data[country].popolazione, rule)[-5:])
    with col2:
        st.write(normalisation(data[country].terapia_intensiva, data[country].popolazione, rule)[-5:])
    with col3:
        st.write(normalisation(data[country].deceduti.diff(), data[country].popolazione, rule)[-5:])
    with col4:
        tpr = data[country].nuovi_positivi[-5:] / data[country].tamponi.diff()[-5:] * 100
        tpr.name = 'TPR (%)'
        st.write(tpr)


explore_regions()
st.header('Confronto tra regioni')
col1, col2, col3, col4, col5 = st.beta_columns(5)
with col1:
    rule = st.selectbox('Variabile', ['Nuovi Positivi', 'Terapie Intensive', 'Percentuale tamponi positivi'])
st.plotly_chart(summary(data, rule), use_container_width=True)

expander = st.beta_expander("This app is developed by Francesco Nazzaro")
expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")

