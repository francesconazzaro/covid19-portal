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
    print('CACHE MISS')
    data_aggregate = pd.read_csv(repo_reference.regions_path, index_col='data', parse_dates=['data'])
    ita = pd.read_csv(repo_reference.italy_path, index_col='data', parse_dates=['data'])

    regioni = {}
    ita['popolazione'] = popolazione.sum()
    regioni['Italia'] = ita
    for regione in np.unique(data_aggregate.denominazione_regione):
        try:
            popolazione_index = [index for index in popolazione.index if regione[:4].lower() in index.lower()][0]
        except:
            print('Unable to find', regione)
        data_in = data_aggregate[data_aggregate.denominazione_regione == regione].sort_index()
        data_in['popolazione'] = popolazione[popolazione_index]
        regioni[regione] = data_in
    return regioni
