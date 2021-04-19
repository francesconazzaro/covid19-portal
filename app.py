import traceback

import streamlit as st

import datetime

import plot
import import_data
import plugins

plugins.google_analytics()
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title='COVID-19 Dashboard', page_icon=":chart_with_upwards_trend:", layout='wide', initial_sidebar_state='collapsed')
try:
    DATA, DATA_TI, DATA_RIC = import_data.covid19()
except:
    st.error(
        "L'applicazione è in fase di aggiornamento. Prova a [riaggiornare](/) la pagina tra qualche secondo.")
    error = st.beta_expander("Dettagli dell'errore")
    error.error(traceback.format_exc())
    st.stop()

fmt = "%d-%m-%Y"


def mobility_expander():
    expander_mobility = st.beta_expander('Dati sulla mobilità')
    col1, col2, col3, col4 = expander_mobility.beta_columns([1, 4, 4, 8])
    with col1:
        country = col1.selectbox('Country', ['IT'] + sorted(import_data.get_list_of_regions()))
    mobility_country = import_data.get_mobility_country(country)
    with col2:
        sub_region_1 = col2.selectbox('sub_region_1', mobility_country.get_sub_region_1())
    with col3:
        sub_region_2 = col3.selectbox('sub_region_2', mobility_country.get_sub_region_2(sub_region_1))
    with col4:
        variable = col4.selectbox('Variabile', mobility_country.get_variables())
    mobility_plot_data = mobility_country.select(sub_region_1, sub_region_2)
    expander_mobility.plotly_chart(plot.mobility_data(mobility_plot_data, variable, variables=mobility_country.get_variables()), use_container_width=True)


def explore_regions(country):
    st.header('Dati sul contagio aggiornati al {}'.format(DATA['Italia'].index[-1].date()))
    col1, line, col3, col4, col5, col5bis, col6 = st.beta_columns([10, 1, 8, 8, 8, 8, 8])
    line.markdown(LINE, unsafe_allow_html=True)
    with col1:
        rule = st.radio('', list(plot.RULE_MAP.keys()))
        st.write('')
        log = st.checkbox('Scala logaritmica', True)
    with col3:
        st.markdown("<h3 style='text-align: center;'>Nuovi positivi</h2>",
                    unsafe_allow_html=True)
        nuovi_positivi = plot.normalisation(DATA[country].iloc[-1].nuovi_positivi, DATA[country].iloc[-1].popolazione, rule)
        text = f'{nuovi_positivi:.2f}' if plot.RULE_MAP[rule] == 'percentage' else f'{int(nuovi_positivi):,}'
        st.markdown(f"<h1 style='text-align: center; color: red;'>{text}</h1>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='text-align: center;'>Deceduti oggi</h2>",
                    unsafe_allow_html=True)
        deceduti = plot.normalisation(DATA[country].deceduti.diff().iloc[-1], DATA[country].iloc[-1].popolazione, rule)
        text = f'{deceduti:.2f}' if plot.RULE_MAP[rule] == 'percentage' else f'{int(deceduti):,}'
        # print(plot.RULE_MAP[rule] == 'percentage')
        st.markdown(f"<h1 style='text-align: center; color: red;'>{text}</h1>", unsafe_allow_html=True)
    with col5:
        st.markdown("<h3 style='text-align: center;'>Persone in terapia intensiva</h2>",
                    unsafe_allow_html=True)
        terapia_intensiva = plot.normalisation(DATA[country].iloc[-1].terapia_intensiva, DATA[country].iloc[-1].popolazione, rule)
        text = f'{terapia_intensiva:.2f}' if plot.RULE_MAP[rule] == 'percentage' else f'{int(terapia_intensiva):,}'
        st.markdown(f"<h1 style='text-align: center; color: red;'>{text}</h1>", unsafe_allow_html=True)
    with col5bis:
        st.markdown("<h3 style='text-align: center;'>Ingressi in terapia intensiva</h2>",
                    unsafe_allow_html=True)
        ingressi = plot.normalisation(DATA[country].iloc[-1].ingressi_terapia_intensiva, DATA[country].iloc[-1].popolazione, rule)
        text = f'{ingressi:.2f}' if plot.RULE_MAP[rule] == 'percentage' else f'{int(ingressi):,}'
        st.markdown(f"<h1 style='text-align: center; color: red;'>{text}</h1>", unsafe_allow_html=True)
    with col6:
        st.markdown("<h3 style='text-align: center;'>Persone ricoverate</h2>",
                    unsafe_allow_html=True)
        ricoverati_con_sintomi = plot.normalisation(DATA[country].iloc[-1].ricoverati_con_sintomi, DATA[country].iloc[-1].popolazione, rule)
        text = f'{ricoverati_con_sintomi:.2f}' if plot.RULE_MAP[rule] == 'percentage' else f'{int(ricoverati_con_sintomi):,}'
        st.markdown(f"<h1 style='text-align: center; color: red;'>{text}</h1>", unsafe_allow_html=True)
    # col1, col2, col3, col4, col5, col6 = st.beta_columns(6)
    # fit_expander = st.beta_expander('Personalizza Fit')
    # col1, col2, col3, col4 = fit_expander.beta_columns(4)
    # with col1:
    #     start_positivi = col1.date_input('Data inizio fit Positivi', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    #     stop_positivi = col1.date_input('Data fine fit Positivi', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    # with col2:
    #     start_ricoveri = col2.date_input('Data inizio fit Ricoveri', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    #     stop_ricoveri = col2.date_input('Data fine fit Ricoveri', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    # with col3:
    #     start_ti = col3.date_input('Data inizio fit Terapie intensive', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    #     stop_ti = col3.date_input('Data fine fit Terapie intensive', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    # with col4:
    #     start_deceduti = col4.date_input('Data inizio fit Deceduti', datetime.date(2020, 10, 15), min_value=datetime.date(2020, 3, 1), max_value=datetime.date.today())
    #     stop_deceduti = col4.date_input('Data fine fit Deceduti', DATA[country].index[-1], min_value=datetime.date(2020, 3, 2), max_value=datetime.date.today())
    start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri, start_deceduti, stop_deceduti = [0] * 8


    st.plotly_chart(plot.plot_selection(DATA, country, rule, start_positivi, start_ti, start_ricoveri, stop_positivi, stop_ti, stop_ricoveri, start_deceduti, stop_deceduti, log=log), use_container_width=True)
    col1, col2, col3 = st.beta_columns([1, 2, 2])
    with col1:
        percentage_rule = col1.radio('', list(plot.PERCENTAGE_RULE.keys()))
    if plot.PERCENTAGE_RULE[percentage_rule] == 'tamponi':
        perc = DATA[country].nuovi_positivi.rolling(7).mean() / DATA[
            country].tamponi.diff().rolling(7).mean() * 100
        perc_points = DATA[country].nuovi_positivi / DATA[country].tamponi.diff() * 100
    else:
        perc = DATA[country].nuovi_positivi.rolling(7).mean() / DATA[country].casi_testati.diff().rolling(7).mean() * 100
        perc_points = DATA[country].nuovi_positivi / DATA[country].casi_testati.diff() * 100
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{percentage_rule}</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{perc_points.iloc[-1]:.2f}</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h3 style='text-align: center;'>Media su 7 giorni della {percentage_rule}</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{perc.iloc[-1]:.2f}</h1>", unsafe_allow_html=True)
    st.plotly_chart(plot.test_positivity_rate(DATA, country, rule=percentage_rule), use_container_width=True)
    try:
        mobility_expander()
    except:
        pass
    table_expander = st.beta_expander('Tabelle andamento')
    col1, _, col2 = table_expander.beta_columns([1, 3, 7])
    with col1:
        days_before = col1.selectbox("Lunghezza tabelle", [5, 10, 25])
    with col2:
        col2.subheader(f'Andamento degli ultimi {days_before} giorni: {country} ({rule})')
    col1, col2, col3, col3bis, col4, col5 = table_expander.beta_columns(6, )#[2, 2, 2, 2, 3, 3])
    data_country = DATA[country].copy()
    data_country.index = data_country.index.strftime(fmt)
    with col1:
        col1.write('Nuovi Positivi')
        df = plot.normalisation(data_country.nuovi_positivi, data_country.popolazione, rule)[-days_before:]
        df.name = 'positivi'
        col1.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col2:
        col2.write('Ricoveri')
        df = plot.normalisation(data_country.ricoverati_con_sintomi, data_country.popolazione, rule)[-days_before:]
        df.name = 'ricoveri'
        col2.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col3:
        col3.write('Terapia Intensiva')
        df = plot.normalisation(data_country.terapia_intensiva, data_country.popolazione, rule)[-days_before:]
        df.name = 'TI'
        col3.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col3bis:
        col3bis.write('Ingressi Terapia Intensiva')
        df = plot.normalisation(data_country.ingressi_terapia_intensiva, data_country.popolazione, rule)[-days_before:]
        df.name = 'TI'
        col3bis.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col4:
        col4.write('Deceduti')
        df = plot.normalisation(data_country.deceduti.diff(), data_country.popolazione, rule)[-days_before:]
        col4.dataframe(df.to_frame().style.background_gradient(cmap='Reds'))
    with col5:
        if plot.PERCENTAGE_RULE[percentage_rule] == 'tamponi':
            col5.write('Percentuale Tamponi Positivi')
            tpr = data_country.nuovi_positivi[-days_before:] / data_country.tamponi.diff()[-days_before:] * 100
            tpr.name = 'TPR (%)'
        elif plot.PERCENTAGE_RULE[percentage_rule] == 'casi':
            col5.write('Percentuale Casi Positivi')
            tpr = data_country.nuovi_positivi[-days_before:] / data_country.casi_testati.diff()[-days_before:] * 100
            tpr.name = 'CPR (%)'
        col5.dataframe(tpr.to_frame().style.background_gradient(cmap='Reds'))
    test_expander = st.beta_expander('Dettaglio sui Test')
    col1, col2 = test_expander.beta_columns(2)
    plot_variables = [
        DATA[country].tamponi_test_molecolare.diff(),
        DATA[country].tamponi_test_antigenico_rapido.diff(),
    ]
    col1.plotly_chart(plot.plot_variables(
        plot_variables,
        ['Test molecolare', 'Test antigenico'],
        popolazione=DATA[country].popolazione,
        rule=rule,
        title=f'Andamento tamponi: {country} ({rule})',
    ), use_container_width=True)
    plot_variables = [
        DATA[country].totale_positivi_test_molecolare.diff() / DATA[country].tamponi_test_molecolare.diff() * 100,
        DATA[country].totale_positivi_test_antigenico_rapido.diff() / DATA[country].tamponi_test_antigenico_rapido.diff() * 100,
    ]
    col2.plotly_chart(plot.plot_variables(
        plot_variables,
        ['Test molecolare', 'Test antigenico'],
        title=f'Percentuale tamponi positivi: {country}',
    ), use_container_width=True)
    # ti_expander = st.beta_expander('Dettaglio sulle Terapie Intensive')
    # col1, col2 = ti_expander.beta_columns(2)
    # plot_variables = [
    #     DATA[country].ingressi_terapia_intensiva,
    #     DATA[country].ingressi_terapia_intensiva - DATA[country].terapia_intensiva.diff(),
    # ]
    # col1.plotly_chart(plot.plot_variables(
    #     plot_variables,
    #     ['Ingressi', 'Uscite'],
    #     title=f'Terapia intensiva: {country} ({rule})',
    # ), use_container_width=True)
    # plot_variables = [
    #     DATA[country].terapia_intensiva.diff(),
    # ]
    # ingressi_not_nan = DATA[country].ingressi_terapia_intensiva[~plot.np.isnan(DATA[country].ingressi_terapia_intensiva)]
    # col2.plotly_chart(plot.plot_variables(
    #     plot_variables,
    #     ['Uscite terapia intensiva'],
    #     title=f'Percentuale tamponi positivi: {country}',
    #     xaxes_range=[ingressi_not_nan.index[0], ingressi_not_nan.index[-1]],
    # ), use_container_width=True)

LINE = """<style>
.vl {
  border-left: 2px solid black;
  height: 200px;
  position: absolute;
  left: 50%;
  margin-left: -3px;
  top: 0;
}
</style>

<div class="vl"></div>"""

st.title('COVID-19: Situazione in Italia')
st.text("")
try:
    vaccines = import_data.vaccines(DATA)
    demography = import_data.demography(vaccines)
except:
    st.error("L'applicazione è in fase di aggiornamento. Prova a [riaggiornare](/) la pagina tra qualche secondo.")
    error = st.beta_expander("Dettagli dell'errore")
    error.error(traceback.format_exc())
    st.stop()

default_what_map = {'infection': 0, 'vaccines': 1, 'contagio': 0, 'vaccini': 1}


def sanitize(string):
    return string.replace(' ', '-')


col1, col2, _, col3 = st.beta_columns([2, 2, 8, 1,])
query_params = st.experimental_get_query_params()

what = col1.radio('Seleziona un dato', ['Contagio', 'Vaccini'],
                      index=default_what_map[query_params.get('dato', ['contagio'])[0].lower()])

default_area_index = -1
for i, region_id in enumerate(import_data.REGIONS_MAP.keys()):
    if region_id.lower() == query_params.get('area', ['IT'])[0].lower():
        default_area_index = i
        break

area = col2.selectbox("Seleziona un'area", ['Italia'] + list(import_data.REGIONS_MAP.values()),
                      index=default_area_index + 1)

col3.write("**[Twitter](https://twitter.com/effenazzaro)<br>[Linkedin](https://www.linkedin.com/in/fnazzaro/)<br>[:beer:](https://www.buymeacoffee.com/francesconazzar)**", unsafe_allow_html=True)
if what == 'Vaccini':
    st.header(f"Dati sulle vaccinazioni aggiornati al {vaccines.administration[vaccines.administration.area == 'Italia'].index[-1].date()}")
    pie1, title1, line, pie2, title2, title3 = st.beta_columns([2, 4, 1, 2, 4, 4])
    line.markdown(LINE, unsafe_allow_html=True)
    with title3:
        st.markdown("<h3 style='text-align: center;'>Dosi somministrate fino ad ora</h2>",
                    unsafe_allow_html=True)
        tot_somm = vaccines.administration[vaccines.administration.area == area].cumsum().iloc[-1].sesso_maschile + vaccines.administration[vaccines.administration.area == area].cumsum().iloc[-1].sesso_femminile
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_somm:,}</h1>", unsafe_allow_html=True)
    with title2:
        st.markdown("<h3 style='text-align: center;'>Dosi consegnate fino ad ora</h2>",
                    unsafe_allow_html=True)
        tot_cons = vaccines.deliveries[vaccines.deliveries.area == area].cumsum().iloc[-1].numero_dosi
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_cons:,}</h1>", unsafe_allow_html=True)
    with pie2:
        st.plotly_chart(plot.plot_percentage(vaccines.administration, vaccines.deliveries, area), use_container_width=True)
    with title1:
        st.markdown("<h3 style='text-align: center;'>Popolazione vaccinata (seconda dose)</h2>",
                    unsafe_allow_html=True)
        tot_somm = vaccines.administration[vaccines.administration.area == area].cumsum().iloc[-1].seconda_dose
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_somm:,}</h1>", unsafe_allow_html=True)
    with pie1:
        st.plotly_chart(plot.plot_vaccine_popolation(vaccines.administration, area), use_container_width=True)


    col1, col2 = st.beta_columns(2)
    with col2:
        vacc_area = vaccines.deliveries[vaccines.deliveries.area == area]
        fornitori = plot.np.unique(vacc_area.fornitore)
        data_list = [vacc_area[vacc_area.fornitore == fornitori[0]].numero_dosi]
        population = vaccines.administration.popolazione[vaccines.administration.area == area]
        names = ['Numero dosi consegnate']
        col2.plotly_chart(plot.plot_deliveries(
            vacc_area,
            population=population,
            subplot_title='Dosi di vaccino consegnate',
            unita=100000
        ), use_container_width=True)
    with col1:
        data_list = [vaccines.administration.prima_dose[vaccines.administration.area == area], vaccines.administration.seconda_dose[vaccines.administration.area == area]]
        population = [vaccines.administration.popolazione[vaccines.administration.area == area]] * 2
        names = ['Prima dose', 'Seconda dose']
        col1.plotly_chart(plot.plot_fill(data_list, names, population_list=population, subplot_title='Somministrazioni vaccino'), use_container_width=True)

    st.subheader(f"Dettaglio andamenti {area}")
    col1, _, col2, col3, _ = st.beta_columns([1, 2, 1, 1, 1])
    status = col1.selectbox('', ['Giornaliero', 'Cumulato'])
    fascia_anagrafica = col2.selectbox('Seleziona fascia anagrafica', ['16-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+'], index=7)
    dose = col3.selectbox('Seleziona dose', ['prima dose', 'seconda dose'])
    if status == 'Cumulato':
        cumulate = True
    else:
        cumulate = False
    col1, col2 = st.beta_columns(2)
    col1.plotly_chart(plot.ages_timeseries(vaccines.raw, area, cumulate=cumulate), use_container_width=True)
    col2.plotly_chart(plot.age_timeseries(vaccines.raw, area, fascia_anagrafica, demography, dose=dose, cumulate=cumulate), use_container_width=True)

    col1, col2 = st.beta_columns(2)
    col2.plotly_chart(plot.fornitori_timeseries(vaccines.raw, area, cumulate=cumulate), use_container_width=True)
    col1.plotly_chart(plot.categories_timeseries(vaccines.administration, area, cumulate=cumulate), use_container_width=True)
    # col2.plotly_chart(plot.plot_fill(data_list, [''], population, cumulate=cumulate, unita=100, subplot_title=names[0]), use_container_width=True)

    col1, col3, col4 = st.beta_columns(3)
    with col1:
        col1.plotly_chart(plot.plot_ages(vaccines.raw, area), use_container_width=True)
    # with col2:
    #     col2.plotly_chart(plot.plot_second_dose_percentage(vaccines.administration, area), use_container_width=True)
    with col3:
        col3.plotly_chart(plot.plot_category(vaccines.administration, area), use_container_width=True)
    with col4:
        col4.plotly_chart(plot.plot_fornitore(vaccines.deliveries, area), use_container_width=True)

    st.header('Confronto tra regioni')
    col1, _, col2 = st.beta_columns([4, 1, 10])
    with col1:
        st.subheader('Percentuale popolazione vaccinata')
        st.write('')
        data = vaccines.administration.groupby('area').sum().seconda_dose / vaccines.administration.groupby('area').mean().popolazione
        st.dataframe(data.to_frame().style.background_gradient(cmap='Reds').format("{:.2%}"), height=700)
    rule = col2.selectbox('Variabile', ['Percentuale popolazione vaccinata', 'Dosi somministrate', 'Dosi consegnate'])
    col2.plotly_chart(plot.vaccines_summary(vaccines, rule), use_container_width=True)
    st.write("**:beer: Buy me a [beer](https://www.buymeacoffee.com/francesconazzar)**")
    expander = st.beta_expander("This app is developed by Francesco Nazzaro")
    expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
    expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")
    expander.write("Raw data")
    expander.dataframe(vaccines.raw)

elif what == 'Contagio':
    explore_regions(area)
    st.header('Confronto tra regioni')
    col1, col2, col3 = st.beta_columns([2, 2, 3])
    import_data.pd.options.display.float_format = '{:.2f}'.format
    with col1:
        st.subheader('Percentuale di posti letto in area medica occupati regione per regione')
        st.write('')
        ric = DATA_RIC.data.to_frame()
        st.dataframe(ric.style.background_gradient(cmap='Reds').format("{:.2%}"), height=700)
    with col2:
        st.subheader('Percentuale di Terapie Intensive occupate regione per regione')
        st.write('')
        ti = DATA_TI.data.to_frame()
        st.dataframe(ti.style.background_gradient(cmap='Reds').format("{:.2%}"), height=700)
    with col3:
        rule = st.selectbox('Variabile', ['Nuovi Positivi', 'Terapie Intensive', 'Ingressi Terapie Intensive', 'Percentuale tamponi positivi', 'Deceduti'])
        st.plotly_chart(plot.summary(DATA, rule), use_container_width=True)
    st.write("*Dati sul totale delle terapie intensive e dei posti letto in area medica aggiornati al 2020-10-28.")
    st.write("**Dati per P.A. Bolzano e P.A. Trento non disponibili.")

    # st.plotly_chart(plot.mortality(DATA))


    st.write("**:beer: Buy me a [beer](https://www.buymeacoffee.com/francesconazzar)**")
    expander = st.beta_expander("This app is developed by Francesco Nazzaro.")
    expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
    expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")
    expander.write("Raw data")
    expander.dataframe(DATA['Italia'])
    expander.plotly_chart(plot.comparison(DATA['Italia']), use_container_width=True)
