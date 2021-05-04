import os
import pathlib
import bs4
import streamlit as st
import re

CWD = os.path.abspath(os.path.dirname(__file__))
try:
    GA_TAG = open(os.path.join(CWD, 'gtag.js')).read()
except FileNotFoundError:
    GA_TAG = ''

try:
    GA_TAG_2 = open(os.path.join(CWD, 'gtag2.js')).read()
except FileNotFoundError:
    GA_TAG_2 = ''

SRC = "https://www.googletagmanager.com/gtag/js?id=G-KY9BZNJJT3"
# print(GA_TAG_2)


def google_analytics():
    # Insert the script in the head tag of the static template inside your virtual environement
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    # Open the file
    with open(index_path, 'r') as index_file:
        data = index_file.read()

        # Check whether there is GA script
        if len(re.findall('UA-', data))==0:

            # Insert Script for Google Analytics
            with open(index_path, 'w') as index_file_f:

                # The Google Analytics script should be pasted in the header of the HTML file
                newdata = re.sub('<head>','<head>' + GA_TAG, data)

                index_file_f.write(newdata)


def google_analytics_old():
    # Insert the script in the head tag of the static template inside your virtual environement
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = bs4.BeautifulSoup(index_path.read_text(), features="lxml")
    if not soup.find(id='custom-js'):
        script_tag_1 = soup.new_tag("script", id='custom-js', src=SRC, attrs={'async': None})
        # script_tag_1.string = GA_TAG
        soup.head.append(script_tag_1)
        script_tag_2 = soup.new_tag("script", id='custom-js-2')
        script_tag_2.string = GA_TAG_2
        soup.head.append(script_tag_2)
        index_path.write_text(str(soup))
