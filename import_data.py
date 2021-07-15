from dateutil import parser
# import git
import requests
import zipfile
import glob
import yaml
import io
import os
import time
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import datetime

CWD = os.path.abspath(os.path.dirname(__file__))
try:
    config = yaml.safe_load(open(os.path.join(CWD, 'config.yaml')))
except FileNotFoundError:
    config = {}
BASE_PATH = config.get('base_path', '.')


REGIONS_MAP = {
    'ABR': 'Abruzzo',
    'BAS': 'Basilicata',
    'CAL': 'Calabria',
    'CAM': 'Campania',
    'EMR': 'Emilia-Romagna',
    'FVG': 'Friuli Venezia Giulia',
    'LAZ': 'Lazio',
    'LIG': 'Liguria',
    'LOM': 'Lombardia',
    'MAR': 'Marche',
    'MOL': 'Molise',
    'PAB': 'P.A. Bolzano',
    'PAT': 'P.A. Trento',
    'PIE': 'Piemonte',
    'PUG': 'Puglia',
    'SAR': 'Sardegna',
    'SIC': 'Sicilia',
    'TOS': 'Toscana',
    'UMB': 'Umbria',
    'VDA': "Valle d'Aosta",
    'VEN': 'Veneto',
}

FASCE_ETA = ['12-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']


ISTAT_REGION_MAP = {
    "Valle d'Aosta": "Valle d'Aosta / Vallée d'Aoste",
    "P.A. Bolzano": "Provincia Autonoma Bolzano / Bozen",
    "P.A. Trento": "Provincia Autonoma Trento",
    "Friuli Venezia Giulia": "Friuli-Venezia Giulia",
}


def demography(vaccines):
    try:
        return pd.read_pickle(os.path.join(CWD, 'resources/demography'))
    except:
        pass
    dem_in = pd.read_csv(os.path.join(CWD, 'resources/demografia.csv'))
    dem_in = dem_in[dem_in.STATCIV2 == 99]
    dem_in = dem_in[dem_in.SEXISTAT1 == 9]
    dem_out = pd.DataFrame(index=FASCE_ETA)
    for area in np.unique(vaccines.raw.area):
        region = dem_in[dem_in.Territorio == ISTAT_REGION_MAP.get(area, area)]
        for eta in range(20, 90, 10):
            value = 0
            for i in range(10):
                eta_id = f'Y{eta + i}'
                value += region[region.ETA1 == eta_id].Value.values[0]
            fascia_id = f'{eta}-{eta + 9}'
            try:
                dem_out[area].loc[fascia_id] = value
            except KeyError:
                dem_out[area] = 0
                dem_out[area].loc[fascia_id] = value
        value = 0
        for i in range(12, 20):
            eta_id = f'Y{i}'
            value += region[region.ETA1 == eta_id].Value.values[0]
        fascia_id = '12-19'
        dem_out[area].loc[fascia_id] = value

        value = 0
        for i in range(90, 100):
            eta_id = f'Y{i}'
            value += region[region.ETA1 == eta_id].Value.values[0]
        eta_id = 'Y_GE100'
        value += region[region.ETA1 == eta_id].Value.values[0]
        fascia_id = '90+'
        dem_out[area].loc[fascia_id] = value
    # dem_out.to_pickle(os.path.join(CWD, 'resources/demography'))
    return dem_out


def population():
    popolazione_regioni = pd.read_csv(os.path.join(CWD, 'resources/popolazione_regioni_italiane.csv'), index_col='regione')
    popolazione = popolazione_regioni.TotaleMaschi + popolazione_regioni.TotaleFemmine
    return popolazione


def intensive_care():
    terapie_intensive = pd.read_csv(os.path.join(CWD, 'resources/posti_terapie_intensive.csv'), sep='\t', index_col='regioni')
    return terapie_intensive


def beds():
    posti_letto = pd.read_csv(os.path.join(CWD, 'resources/posti_letto.csv'), index_col='regioni')
    return posti_letto


def process_data(data, covid_data, date_label, drop_ages=False, deliveries=False):
    result = pd.DataFrame()
    for region_name in REGIONS_MAP.keys():
        if drop_ages is True:
            region = data[data.area == region_name].groupby(
                date_label).sum()
        else:
            region = data[data.area == region_name]
        region['area'] = REGIONS_MAP[region_name]
        region['popolazione'] = covid_data[REGIONS_MAP[region_name]].popolazione[0]
        result = result.append(region)
    if deliveries:
        ita = pd.DataFrame()
        for fornitore in np.unique(data.fornitore):
            fornitore_data = data[data.fornitore == fornitore].groupby(date_label).sum()
            fornitore_data['fornitore'] = fornitore
            ita = ita.append(fornitore_data)
        ita = ita.sort_index()
    else:
        ita = result.groupby(date_label).sum()
    ita['area'] = 'Italia'
    ita['popolazione'] = covid_data['Italia'].popolazione[0]
    result = result.append(ita)
    return result


class Vaccines:
    def __init__(self, vaccines, deliveries, covid_data):
        self.raw = process_data(vaccines, covid_data, date_label='data_somministrazione')
        self.administration = process_data(vaccines, covid_data, drop_ages=True, date_label='data_somministrazione')
        self.deliveries = process_data(deliveries, covid_data, date_label='data_consegna', deliveries=True)


@st.cache(show_spinner=False, ttl=60*60)
def vaccines(covid_data):
    url = "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/somministrazioni-vaccini-latest.csv"
    s=requests.get(url).content
    vaccine_data = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col='data_somministrazione', parse_dates=['data_somministrazione'])
    url = "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/consegne-vaccini-latest.csv"
    s=requests.get(url).content
    deliveries = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col='data_consegna', parse_dates=['data_consegna'])
    return Vaccines(vaccine_data, deliveries, covid_data)


def get_list_of_regions():
    mobility_data_path = os.path.join(BASE_PATH, 'mobility')
    if not os.path.exists(mobility_data_path):
        response = requests.get('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip')
        mobility_zip_path = os.path.join(BASE_PATH, 'mobility_data.zip')
        with open(mobility_zip_path, 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(mobility_zip_path, 'r') as zip_ref:
            zip_ref.extractall(mobility_data_path)
    list_of_regions = []
    for path in glob.glob(os.path.join(mobility_data_path, '2020*.csv')):
        list_of_regions.append(os.path.basename(path)[5:7])
    return list_of_regions


class Mobility:
    def __init__(self, data):
        self.data = data

    def get_sub_region_1(self):
        return ['Totale'] + list(np.unique(self.data.sub_region_1.fillna('')))[1:]

    def get_sub_region_2(self, sub_region_1):
        data_sel = self.data[self.data.sub_region_1 == sub_region_1]
        return ['Totale'] + list(np.unique(data_sel.sub_region_2.fillna('')))[1:]

    def get_variables(self):
        return [col for col in self.data.columns if 'from_baseline' in col]

    def select(self, sub_region_1, sub_region_2):
        if sub_region_2 is not 'Totale':
            return self.data[self.data.sub_region_2 == sub_region_2]
        elif sub_region_1 is not 'Totale':
            iso_3166_2_code = self.data[self.data.sub_region_1 == sub_region_1].iso_3166_2_code[0]
            return self.data[self.data.iso_3166_2_code == iso_3166_2_code]
        else:
            return self.data[self.data.iso_3166_2_code.fillna('') == '']


def get_mobility_country(country):
    mobility_data_path = os.path.join(BASE_PATH, 'mobility')
    mobility_country = pd.DataFrame()
    for mobility_country_path in glob.glob(os.path.join(mobility_data_path, f'202*_{country}_Region_Mobility_Report.csv')):
        mobility_country = mobility_country.append(pd.read_csv(mobility_country_path, index_col='date'))
    return Mobility(mobility_country.sort_index())


# class RepoReference:
#     def __init__(self, base_path=BASE_PATH, repo_path='COVID-19', repo_url="https://github.com/pcm-dpc/COVID-19.git"):
#         path = os.path.join(BASE_PATH, repo_path)
#         if not os.path.exists(path):
#             git.Git(BASE_PATH).clone(repo_url)
#         repo = git.Repo(path)
#         o = repo.remotes.origin
#         try:
#             o.pull()
#         except:
#             pass
#         self.path = path
#         self.hexsha = repo.head.commit.hexsha
#         self.regions_path = os.path.join(base_path, 'COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv')
#         self.italy_path = os.path.join(base_path, 'COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')


# def cache(func, cache_time=60*60):
#     def wrapper(*args, **kwargs):
#         now = time.time()
#         cache_file = f'./{func.__name__}.pk'
#         if os.path.exists(cache_file):
#             with open(cache_file, 'rb') as f:
#                 cached_time, data = pickle.load(f)
#         else:
#             data = func(*args, **kwargs)
#             with open(cache_file, 'wb') as f:
#                 pickle.dump((now, data), f)
#             return data
#         if now - cached_time < cache_time:
#             return data
#         else:
#             data = func(*args, **kwargs)
#             with open(cache_file, 'wb') as f:
#                 pickle.dump((now, data), f)
#             return data
#     return wrapper


@st.cache(show_spinner=False, ttl=60*60)
def covid19():
    popolazione = population()
    terapie_intensive = intensive_care()
    posti_letto = beds()
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
    s=requests.get(url).content
    data_aggregate = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col='data', parse_dates=['data'])
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    s=requests.get(url).content
    ita = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col='data', parse_dates=['data'])

    regioni = {}
    ita['popolazione'] = popolazione.sum()
    ita['terapie_intensive_disponibili'] = terapie_intensive.posti_attuali.sum()
    ita['posti_letto_disponibili'] = posti_letto.posti_attuali.sum()
    terapie_intensive['data'] = 0
    posti_letto['data'] = 0
    regioni['Italia'] = ita
    for regione in np.unique(data_aggregate.denominazione_regione):
        try:
            popolazione_index = [index for index in popolazione.index if regione[:5].lower() in index.lower()][0]
            terapie_intensive_index = [index for index in terapie_intensive.index if regione[:5].lower() in index.lower()][0]
            posti_letto_index = [index for index in posti_letto.index if regione[:5].lower() in index.lower()][0]
        except:
            popolazione_index = [index for index in popolazione.index if regione[-5:].lower() in index.lower()][0]
            terapie_intensive_index = [index for index in terapie_intensive.index if regione[-5:].lower() in index.lower()][0]
            posti_letto_index = [index for index in posti_letto.index if regione[-5:].lower() in index.lower()][0]
        data_in = data_aggregate[data_aggregate.denominazione_regione == regione].sort_index()
        data_in['popolazione'] = popolazione[popolazione_index]
        data_in['terapie_intensive_disponibili'] = terapie_intensive.posti_attuali[terapie_intensive_index]
        data_in['posti_letto_disponibili'] = posti_letto.posti_attuali[posti_letto_index]
        ti_perc = data_in.terapia_intensiva[-1] / terapie_intensive.posti_attuali[terapie_intensive_index]
        beds_perc = data_in.ricoverati_con_sintomi[-1] / posti_letto.posti_attuali[posti_letto_index]
        if np.isfinite(ti_perc) and np.isfinite(beds_perc):
            terapie_intensive.data.loc[terapie_intensive_index] = ti_perc
            posti_letto.data.loc[posti_letto_index] = beds_perc
        regioni[regione] = data_in
    return regioni, terapie_intensive, posti_letto
