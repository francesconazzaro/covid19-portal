import git
import requests
import zipfile
import glob
import yaml
import os
import pandas as pd
import numpy as np
import streamlit as st

CWD = os.path.abspath(os.path.dirname(__file__))
try:
    config = yaml.safe_load(open(os.path.join(CWD, 'config.yaml')))
except FileNotFoundError:
    config = {}
BASE_PATH = config.get('base_path', '/app')


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


def process_data(data, covid_data, date_label, drop_ages=False):
    administration = pd.DataFrame()
    for region_name in REGIONS_MAP.keys():
        if drop_ages is True:
            region = data[data.area == region_name].groupby(
                date_label).sum()
        else:
            region = data[data.area == region_name]
        region['area'] = REGIONS_MAP[region_name]
        region['popolazione'] = covid_data[REGIONS_MAP[region_name]].popolazione[0]
        administration = administration.append(region)
    ita = administration.groupby(date_label).sum()
    ita['area'] = 'Italia'
    ita['popolazione'] = covid_data['Italia'].popolazione[0]
    administration = administration.append(ita)
    return administration


class Vaccines:
    def __init__(self, vaccines, deliveries, covid_data):
        self.raw = process_data(vaccines, covid_data, date_label='data_somministrazione')
        self.administration = process_data(vaccines, covid_data, drop_ages=True, date_label='data_somministrazione')
        self.deliveries = process_data(deliveries, covid_data, date_label='data_consegna')


def vaccines(repo_reference, covid_data):
    vaccine_data = pd.read_csv(os.path.join(BASE_PATH, 'covid19-opendata-vaccini/dati/somministrazioni-vaccini-latest.csv'), index_col='data_somministrazione', parse_dates=['data_somministrazione'])
    deliveries = pd.read_csv(os.path.join(BASE_PATH, 'covid19-opendata-vaccini/dati/consegne-vaccini-latest.csv'), index_col='data_consegna', parse_dates=['data_consegna'])
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
    mobility_country_path = os.path.join(mobility_data_path, f'2020_{country}_Region_Mobility_Report.csv')
    mobility_country = pd.read_csv(mobility_country_path, index_col='date')
    return Mobility(mobility_country)


class RepoReference:
    def __init__(self, base_path=BASE_PATH, repo_path='COVID-19', repo_url="https://github.com/pcm-dpc/COVID-19.git"):
        path = os.path.join(BASE_PATH, repo_path)
        if not os.path.exists(path):
            git.Git(BASE_PATH).clone(repo_url)
        repo = git.Repo(path)
        o = repo.remotes.origin
        try:
            o.pull()
        except:
            pass
        self.path = path
        self.hexsha = repo.head.commit.hexsha
        self.regions_path = os.path.join(base_path, 'COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv')
        self.italy_path = os.path.join(base_path, 'COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')


@st.cache(show_spinner=False)
def covid19(repo_reference):
    popolazione = population()
    terapie_intensive = intensive_care()
    posti_letto = beds()
    data_aggregate = pd.read_csv(repo_reference.regions_path, index_col='data', parse_dates=['data'])
    ita = pd.read_csv(repo_reference.italy_path, index_col='data', parse_dates=['data'])

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
