import os
import pathlib
import bs4
import streamlit as st

CWD = os.path.abspath(os.path.dirname(__file__))
try:
    GA_TAG_1 = open(os.path.join(CWD, 'gtag1.js')).read()
except FileNotFoundError:
    GA_TAG_1 = ''

try:
    GA_TAG_2 = open(os.path.join(CWD, 'gtag2.js')).read()
except FileNotFoundError:
    GA_TAG_2 = ''

SRC = "https://www.googletagmanager.com/gtag/js?id=G-KY9BZNJJT3"
# print(GA_TAG_2)


def google_analytics():
    # Insert the script in the head tag of the static template inside your virtual environement
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    print(f"---- INDEX PATH: {index_path}")
    soup = bs4.BeautifulSoup(index_path.read_text(), features="lxml")
    if not soup.find(id='custom-js'):
        script_tag_1 = soup.new_tag("script", id='custom-js', src=SRC, attrs={'async': None})
        # script_tag_1.string = GA_TAG_1
        soup.head.append(script_tag_1)
        script_tag_2 = soup.new_tag("script", id='custom-js-2')
        script_tag_2.string = GA_TAG_2
        soup.head.append(script_tag_2)
        index_path.write_text(str(soup))
