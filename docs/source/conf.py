# Configuration file for the Sphinx documentation builder.

# -- For some crazy reason this is a requirement to import the package
import os
from re import A
import sys
import shutil
import glob

sys.path.insert(0, os.path.abspath("../.."))
project_root = os.path.abspath("../..")
print("Running doc generation, project root at {}".format(project_root))

# -- Project information

project = "Moonshine"
copyright = "2023, Moonshine Labs"
author = "Moonshine Labs"

release = "0.1.5"
version = "0.1.5"

# -- Theme options
html_static_path = ['_static']
html_title = ' Moonshine'

# Customize CSS
html_css_files = ['css/custom.css']

# -- Options for HTML output
html_theme = "furo"
html_theme_options = {
    'light_logo': 'logo-light-mode.png',
    'dark_logo': 'logo-light-mode.png',
    'light_css_variables': {
        'color-brand-primary': '#373737',
        'color-brand-content': '#373737',
    },
    'dark_css_variables': {
        'color-brand-primary': '#f9f9f9',
        'color-brand-content': '#f9f9f9',
    },
}

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
]
exclude_patterns = ['_build', '**.ipynb_checkpoints']

pygments_style = 'manni'
pygments_dark_style = 'monokai'

autosummary_generate = True

nbsphinx_execute = 'never'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Move the example notebooks into the doc source tree
print("Copy example notebooks into docs/_examples")
examples_root = os.path.join(project_root, "docs/source/_examples")
shutil.rmtree(examples_root, ignore_errors=True)
os.makedirs(examples_root)
pynb_files = glob.glob(os.path.join(project_root, "examples/**/*.ipynb"))

for f in pynb_files:
    new_path = os.path.join(examples_root, os.path.basename(f))
    shutil.copyfile(f, new_path)