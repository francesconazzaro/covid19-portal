
import streamlit as st

import datetime

import plot
import import_data
import plugins

plugins.google_analytics()


st.set_page_config(layout='wide', initial_sidebar_state='collapsed')
repo_reference = import_data.RepoReference()
DATA, DATA_TI, DATA_RIC = import_data.covid19(repo_reference)

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
    expander_mobility.plotly_chart(plot.mobility_data(mobility_plot_data, variable), use_container_width=True)


def explore_regions():
    st.header('Dati sul contagio aggiornati al {}'.format(DATA['Italia'].index[-1].date()))
    col1, col2, col3, col4, col5, col6 = st.beta_columns([1, 1, 1, 1, 1, 1])
    with col1:
        country = st.selectbox("Seleziona un'area", list(DATA.keys()))
        log = st.checkbox('Scala logaritmica', True)
    with col2:
        rule = st.radio('', list(plot.RULE_MAP.keys()))
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
        print(plot.RULE_MAP[rule] == 'percentage')
        st.markdown(f"<h1 style='text-align: center; color: red;'>{text}</h1>", unsafe_allow_html=True)
    with col5:
        st.markdown("<h3 style='text-align: center;'>Persone in terapia intensiva</h2>",
                    unsafe_allow_html=True)
        terapia_intensiva = plot.normalisation(DATA[country].iloc[-1].terapia_intensiva, DATA[country].iloc[-1].popolazione, rule)
        text = f'{terapia_intensiva:.2f}' if plot.RULE_MAP[rule] == 'percentage' else f'{int(terapia_intensiva):,}'
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
    mobility_expander()
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


st.title('COVID-19: Situazione in Italia')
vaccine_repo = import_data.RepoReference(
    repo_path='covid19-opendata-vaccini',
    repo_url='https://github.com/italia/covid19-opendata-vaccini.git'
)
vaccines = import_data.vaccines(vaccine_repo, DATA)

what = st.radio('', ['Dati contagio', 'Dati somministrazione vaccini'])

if what == 'Dati somministrazione vaccini':
    st.header(f"Dati sulle vaccinazioni aggiornati al {vaccines.administration[vaccines.administration.area == 'Italia'].index[-1].date()}")
    col1, col2, col3 = st.beta_columns([2, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align: center;'>Dosi somministrate fino ad ora in Italia</h2>",
                    unsafe_allow_html=True)
        tot_somm = vaccines.administration[vaccines.administration.area == 'Italia'].cumsum().iloc[-1].sesso_maschile + vaccines.administration[vaccines.administration.area == 'Italia'].cumsum().iloc[-1].sesso_femminile
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_somm:,}</h1>", unsafe_allow_html=True)
    with col1:
        st.markdown("<h3 style='text-align: center;'>Dosi consegnate fino ad ora in Italia</h2>",
                    unsafe_allow_html=True)
        tot_cons = vaccines.deliveries[vaccines.deliveries.area == 'Italia'].cumsum().iloc[-1].numero_dosi
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_cons:,}</h1>", unsafe_allow_html=True)
    with col3:
        col3.plotly_chart(plot.plot_percentage(vaccines.administration, vaccines.deliveries, 'Italia'), use_container_width=True)
    st.subheader('Dosi somministrate')
    st.write('Per visualizzare una sola regione fare doppio click sul nome della regione')
    st.plotly_chart(plot.plot_vaccines(vaccines.administration), use_container_width=True)

    second_dose_expander = st.beta_expander("Percentuale popolazione vaccinata (seconda dose)")
    second_dose_expander.plotly_chart(plot.second_dose(vaccines.administration), use_container_width=True)

    st.subheader('Dettaglio per area')
    col1, col2, col3, col4, col5 = st.beta_columns([1, 2, 2, 2, 2])
    with col1:
        area = col1.selectbox("Seleziona un'area", ['Italia'] + list(import_data.REGIONS_MAP.values()))
    with col3:
        st.markdown("<h3 style='text-align: center;'>Dosi somministrate fino ad ora</h2>",
                    unsafe_allow_html=True)
        tot_somm = vaccines.administration[vaccines.administration.area == area].cumsum().iloc[-1].sesso_maschile + vaccines.administration[vaccines.administration.area == area].cumsum().iloc[-1].sesso_femminile
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_somm:,}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h3 style='text-align: center;'>Dosi consegnate fino ad ora</h2>",
                    unsafe_allow_html=True)
        tot_cons = vaccines.deliveries[vaccines.deliveries.area == area].cumsum().iloc[-1].numero_dosi
        st.markdown(f"<h1 style='text-align: center; color: red;'>{tot_cons:,}</h1>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='text-align: center;'>Dosi somministrate al giorno</h2>",
                    unsafe_allow_html=True)
        avg_somm = tot_somm / len(vaccines.administration[vaccines.administration.area == area].cumsum().index)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{avg_somm:,.2f}</h1>", unsafe_allow_html=True)
    with col5:
        col5.plotly_chart(plot.plot_percentage(vaccines.administration, vaccines.deliveries, area), use_container_width=True)
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        col1.plotly_chart(plot.plot_ages(vaccines.raw, area), use_container_width=True)
    with col2:
        col2.plotly_chart(plot.plot_second_dose_percentage(vaccines.administration, area), use_container_width=True)
    with col3:
        col3.plotly_chart(plot.plot_category(vaccines.administration, area), use_container_width=True)
    col1, col2 = st.beta_columns(2)
    with col1:
        col1.plotly_chart(plot.plot_deliveries(vaccines.deliveries, area), use_container_width=True)
    with col2:
        col2.plotly_chart(plot.plot_vaccines(
            vaccines.administration,
            area,
            unita=plot.UNITA,
            subplot_title='Dosi somministrate per 100 mila abitanti',
            fill='tozeroy',
            height=500,
        ), use_container_width=True
        )
    st.plotly_chart(plot.second_dose(vaccines.administration, area), use_container_width=True)
    expander = st.beta_expander("This app is developed by Francesco Nazzaro (click to check raw data)")
    expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
    expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")
    expander.write("Raw data")
    expander.dataframe(vaccines.administration)

elif what == 'Dati contagio':
    explore_regions()
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
        st.plotly_chart(plot.summary(DATA, rule, st), use_container_width=True)
    st.write("*Dati sul totale delle terapie intensive e dei posti letto in area medica aggiornati al 2020-10-28.")
    st.write("**Dati per P.A. Bolzano e P.A. Trento non disponibili.")

    # st.plotly_chart(plot.mortality(DATA))


    expander = st.beta_expander("This app is developed by Francesco Nazzaro (click to check raw data)")
    expander.write("Contact me on [Twitter](https://twitter.com/effenazzaro)")
    expander.write("The source code is on [GitHub](https://github.com/francesconazzaro/covid19-portal)")
    expander.write("Raw data")
    expander.dataframe(DATA['Italia'])
    expander.plotly_chart(plot.comparison(DATA['Italia']), use_container_width=True)
