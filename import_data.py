import git
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


class RepoReference:
    def __init__(self, base_path=BASE_PATH):
        path = os.path.join(BASE_PATH, 'COVID-19')
        if not os.path.exists(path):
            git.Git(BASE_PATH).clone("https://github.com/pcm-dpc/COVID-19.git")
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


@st.cache()
def covid19(repo_reference):
    popolazione = population()
    terapie_intensive = intensive_care()
    posti_letto = beds()
    print('CACHE MISS')
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
            print('Unable to find', regione)
            continue
        data_in = data_aggregate[data_aggregate.denominazione_regione == regione].sort_index()
        data_in['popolazione'] = popolazione[popolazione_index]
        data_in['terapie_intensive_disponibili'] = terapie_intensive.posti_attuali[terapie_intensive_index]
        data_in['posti_letto_disponibili'] = posti_letto.posti_attuali[posti_letto_index]
        terapie_intensive.data[terapie_intensive_index] = float(data_in.terapia_intensiva[-1]) / terapie_intensive.posti_attuali[terapie_intensive_index] * 100
        posti_letto.data[posti_letto_index] = float(data_in.ricoverati_con_sintomi[-1]) / posti_letto.posti_attuali[posti_letto_index] * 100
        regioni[regione] = data_in
    return regioni, terapie_intensive, posti_letto
