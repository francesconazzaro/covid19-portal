
import streamlit as st

import datetime

import plot
import import_data
import plugins

plugins.google_analytics()


st.set_page_config(layout='wide', initial_sidebar_state='collapsed')
repo_reference = import_data.RepoReference()
DATA, DATA_TI, DATA_RIC = import_data.covid19(repo_reference)
st.title('COVID-19: Situazione in Italia aggiornata al {}'.format(DATA['Italia'].index[-1].date()))

fmt = "%d-%m-%Y"

def explore_regions():
    st.header('Dati regione')
    col1, col2, *_ = st.beta_columns(5)
    with col1:
        country = st.selectbox('Seleziona una regione', list(DATA.keys()))
        log = st.checkbox('Scala logaritmica', True)
    with col2:
        rule = st.radio('', list(plot.RULE_MAP.keys()))
    # col1, col2, col3, col4, col5, col6 = st.beta_columns(6)
    col1, col2 = st.sidebar.beta_columns(2)
    with col1:
        start_positivi = st.date_input('Data inizio fit Positivi', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
        start_ricoveri = st.date_input('Data inizio fit Ricoveri', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
        start_ti = st.date_input('Data inizio fit Terapie intensive', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
        start_deceduti = st.date_input('Data inizio fit Deceduti', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col2:
        stop_positivi = st.date_input('Data fine fit Positivi', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        stop_ricoveri = st.date_input('Data fine fit Ricoveri', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        stop_ti = st.date_input('Data fine fit Terapie intensive', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        stop_deceduti = st.date_input('Data fine fit Deceduti', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    st.plotly_chart(plot.plot_selection(DATA, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri, start_deceduti, stop_deceduti, log=log), use_container_width=True)
    percentage_rule = st.radio('', list(plot.PERCENTAGE_RULE.keys()))
    st.plotly_chart(plot.test_positivity_rate(DATA, country, rule=percentage_rule), use_container_width=True)
    st.subheader(f'Andamento degli ultimi 5 giorni: {country} ({rule})')
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    data_country = DATA[country].copy()
    data_country.index = data_country.index.strftime(fmt)
    with col1:
        st.write('Nuovi Positivi')
        df = plot.normalisation(data_country.nuovi_positivi, data_country.popolazione, rule)[-5:]
        df.name = 'positivi'
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col2:
        st.write('Ricoveri')
        df = plot.normalisation(data_country.ricoverati_con_sintomi, data_country.popolazione, rule)[-5:]
        df.name = 'ricoveri'
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col3:
        st.write('Terapia Intensiva')
        df = plot.normalisation(data_country.terapia_intensiva, data_country.popolazione, rule)[-5:]
        df.name = 'TI'
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col4:
        st.write('Deceduti')
        df = plot.normalisation(data_country.deceduti.diff(), data_country.popolazione, rule)[-5:]
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col5:
        if plot.PERCENTAGE_RULE[percentage_rule] == 'tamponi':
            st.write('Percentuale Tamponi Positivi')
            tpr = data_country.nuovi_positivi[-5:] / data_country.tamponi.diff()[-5:] * 100
            tpr.name = 'TPR (%)'
        elif plot.PERCENTAGE_RULE[percentage_rule] == 'casi':
            st.write('Percentuale Casi Positivi')
            tpr = data_country.nuovi_positivi[-5:] / data_country.casi_testati.diff()[-5:] * 100
            tpr.name = 'CPR (%)'
        st.dataframe(tpr.to_frame().style.background_gradient(cmap='Reds'))

explore_regions()
st.header('Confronto tra regioni')
col1, col2, col3 = st.beta_columns([2, 2, 3])
with col1:
    st.subheader('Percentuale di posti letto in area medica occupati regione per regione')
    st.write('')
    st.dataframe(DATA_RIC.data.to_frame().style.background_gradient(cmap='Reds'), height=500)
with col2:
    st.subheader('Percentuale di Terapie Intensive occupate regione per regione')
    st.write('')
    st.dataframe(DATA_TI.data.to_frame().style.background_gradient(cmap='Reds'), height=500)
with col3:
    rule = st.selectbox('Variabile', ['Nuovi Positivi', 'Terapie Intensive', 'Percentuale tamponi positivi', 'Deceduti'])
    st.plotly_chart(plot.summary(DATA, rule, st), use_container_width=True)
st.write("Dati sul totale delle terapie intensive e dei posti letto in area medica aggiornati al 2020-10-28")

# st.plotly_chart(plot.mortality(DATA))


expander = st.beta_expander("This app is developed by Francesco Nazzaro (click to check raw data)")
expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")
expander.write("Raw data")
expander.dataframe(DATA['Italia'])
expander.plotly_chart(plot.comparison(DATA['Italia']), use_container_width=True)
