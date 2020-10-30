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


class FileReference:
    def __init__(self):
        base_path = BASE_PATH
        self.filename_regioni = os.path.join(base_path, 'COVID-19/dati-regioni/')
        self.filename_ita = os.path.join(base_path, 'COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')


class RepoReference:
    def __init__(self):
        path = 'COVID-19'
        repo = git.Repo(path)
        o = repo.remotes.origin
        o.pull()
        self.hexsha = repo.head.commit.hexsha


def hash_file_reference(file_reference):
    filename_regioni = file_reference.filename_regioni
    filename_ita = file_reference.filename_ita
    return (filename_regioni, filename_ita, os.path.getmtime(filename_regioni), os.path.getmtime(filename_ita))


def hash_repo_reference(repo_reference):
    return (repo_reference.hexsha)


@st.cache(suppress_st_warning=True, hash_funcs={FileReference: hash_file_reference, RepoReference: hash_repo_reference})
def covid19(base_path=BASE_PATH):
    popolazione = population()
    if not os.path.exists('COVID-19'):
        git.Git("./").clone("https://github.com/pcm-dpc/COVID-19.git")
    data_aggregate = pd.read_csv(os.path.join(base_path, 'COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv'), index_col='data', parse_dates=['data'])
    ita = pd.read_csv(os.path.join(base_path, 'COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'), index_col='data', parse_dates=['data'])

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
