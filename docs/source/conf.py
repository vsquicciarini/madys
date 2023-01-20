# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'MADYS'
copyright = '2022, Vito Squicciarini'
author = 'Vito Squicciarini'

release = '0.5'
version = '0.5.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'madys_logo_black_small.png'

html_theme = 'sphinx_rtd_theme'

html_favicon = 'clock2.ico'

# -- Options for EPUB output
epub_show_urls = 'footnote'
