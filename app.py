
import streamlit as st

import datetime

import plot
import import_data
import plugins

plugins.google_analytics()


st.set_page_config(layout='wide')
repo_reference = import_data.RepoReference()
DATA = import_data.covid19(repo_reference)
st.title('COVID-19: Situazione in Italia aggiornata al {}'.format(DATA['Italia'].index[-1].date()))


def explore_regions():
    st.header('Dati regione')
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        country = st.selectbox('Seleziona una regione', list(DATA.keys()))
    with col2:
        rule = st.radio('', list(plot.RULE_MAP.keys()))
    col1, col2, col3, *_ = st.beta_columns(6)
    with col1:
        start_positivi = st.date_input('Data inizio fit Nuovi positivi', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col2:
        start_ti = st.date_input('Data inizio fit Terapie intensive', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col3:
        start_ricoveri = st.date_input('Data inizio fit Ricoveri', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col1:
        stop_positivi = st.date_input('Data fine fit Nuovi positivi', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    with col2:
        stop_ti = st.date_input('Data fine fit Terapie intensive', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    with col3:
        stop_ricoveri = st.date_input('Data fine fit Ricoveri', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    st.plotly_chart(plot.plot_selection(DATA, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri), use_container_width=True)
    st.plotly_chart(plot.test_positivity_rate(DATA, country), use_container_width=True)
    st.subheader(f'Andamento degli ultimi 5 giorni: {country} ({rule})')
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.write(plot.normalisation(DATA[country].nuovi_positivi, DATA[country].popolazione, rule)[-5:])
    with col2:
        st.write(plot.normalisation(DATA[country].terapia_intensiva, DATA[country].popolazione, rule)[-5:])
    with col3:
        st.write(plot.normalisation(DATA[country].deceduti.diff(), DATA[country].popolazione, rule)[-5:])
    with col4:
        tpr = DATA[country].nuovi_positivi[-5:] / DATA[country].tamponi.diff()[-5:] * 100
        tpr.name = 'TPR (%)'
        st.write(tpr)


explore_regions()
st.header('Confronto tra regioni')
col1, col2, col3, col4, col5 = st.beta_columns(5)
with col1:
    rule = st.selectbox('Variabile', ['Nuovi Positivi', 'Terapie Intensive', 'Percentuale tamponi positivi'])
st.plotly_chart(plot.summary(DATA, rule, st), use_container_width=True)

expander = st.beta_expander("This app is developed by Francesco Nazzaro")
expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")

