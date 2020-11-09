
import streamlit as st

import datetime

import plot
import import_data
import plugins

plugins.google_analytics()


st.set_page_config(layout='wide', initial_sidebar_state='collapsed')
repo_reference = import_data.RepoReference()
DATA = import_data.covid19(repo_reference)
st.title('COVID-19: Situazione in Italia aggiornata al {}'.format(DATA.data['Italia'].index[-1].date()))


fmt = "%d-%m-%Y"


def explore_regions():
    # col1, col2, col3, col4, col5, col6 = st.beta_columns(6)
    col1, col2 = st.sidebar.beta_columns(2)
    with col1:
        start_positivi = st.date_input('Data inizio fit Positivi', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
        start_ricoveri = st.date_input('Data inizio fit Ricoveri', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
        start_ti = st.date_input('Data inizio fit Terapie intensive', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
        start_deceduti = st.date_input('Data inizio fit Deceduti', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    with col2:
        stop_positivi = st.date_input('Data fine fit Positivi', DATA.data['Italia'].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        DATA.set_start_stop('nuovi_positivi', start_positivi, stop_positivi)
        stop_ricoveri = st.date_input('Data fine fit Ricoveri', DATA.data['Italia'].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        DATA.set_start_stop('ricoverati_con_sintomi', start_ricoveri, stop_ricoveri)
        stop_ti = st.date_input('Data fine fit Terapie intensive', DATA.data['Italia'].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        DATA.set_start_stop('terapia_intensiva', start_ti, stop_ti)
        stop_deceduti = st.date_input('Data fine fit Deceduti', DATA.data['Italia'].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
        DATA.set_start_stop('deceduti', start_deceduti, stop_deceduti)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        country = st.selectbox('Seleziona una regione', list(DATA.data.keys()))
        log = st.checkbox('Scala logaritmica', True)
    with col2:
        rule = st.radio('', list(plot.stats.RULE_MAP.keys()))
    st.header('Dati regione')
    selection_plot = plot.plot_selection(DATA, country, rule, log=log)
    st.plotly_chart(selection_plot, use_container_width=True)
    percentage_rule = st.radio('', list(plot.PERCENTAGE_RULE.keys()))
    st.plotly_chart(plot.test_positivity_rate(DATA.data, country, rule=percentage_rule), use_container_width=True)
    st.subheader(f'Andamento degli ultimi 5 giorni: {country} ({rule})')
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    data_country = DATA.data[country].copy()
    data_country.index = data_country.index.strftime(fmt)
    with col1:
        st.write('Nuovi Positivi')
        df = plot.stats.normalisation(data_country.nuovi_positivi, data_country.popolazione, rule)[-5:]
        df.name = 'positivi'
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col2:
        st.write('Ricoveri')
        df = plot.stats.normalisation(data_country.ricoverati_con_sintomi, data_country.popolazione, rule)[-5:]
        df.name = 'ricoveri'
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col3:
        st.write('Terapia Intensiva')
        df = plot.stats.normalisation(data_country.terapia_intensiva, data_country.popolazione, rule)[-5:]
        df.name = 'TI'
        st.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col4:
        st.write('Deceduti')
        df = plot.stats.normalisation(data_country.deceduti.diff(), data_country.popolazione, rule)[-5:]
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
    st.dataframe(DATA.posti_letto.data.to_frame().style.background_gradient(cmap='Reds'), height=500)
with col2:
    st.subheader('Percentuale di Terapie Intensive occupate regione per regione')
    st.write('')
    st.dataframe(DATA.terapie_intensive.data.to_frame().style.background_gradient(cmap='Reds'), height=500)
with col3:
    rule = st.selectbox('Variabile', ['Nuovi Positivi', 'Terapie Intensive', 'Percentuale tamponi positivi'])
    st.plotly_chart(plot.summary(DATA.data, rule, st), use_container_width=True)
st.write("Dati sul totale delle terapie intensive e dei posti letto in area medica aggiornati al 2020-10-28")

# st.plotly_chart(plot.mortality(DATA))


expander = st.beta_expander("This app is developed by Francesco Nazzaro (click to check raw data)")
expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")
expander.write("Raw data")
expander.dataframe(DATA.data['Italia'])
