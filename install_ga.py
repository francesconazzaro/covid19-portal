import os
import pathlib
import bs4
import streamlit as st

CWD = os.path.abspath(os.path.dirname(__file__))
try:
    GA_TAG = open(os.path.join(CWD, 'gtag.js')).read()
except FileNotFoundError:
    GA_TAG = ''

# Insert the script in the head tag of the static template inside your virtual environement
index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
soup = bs4.BeautifulSoup(index_path.read_text(), features="lxml")
if not soup.find(id='custom-js'):
    script_tag = soup.new_tag("script", id='custom-js')
    script_tag.string = GA_TAG
    soup.head.append(script_tag)
    index_path.write_text(str(soup))

print(open(index_path).read())
