# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
project = 'hsi-wizard'
copyright = '2024, Felix Wühler'
author = 'Felix Wühler'
release = '0.0.7'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # To generate autodocs
    'sphinx.ext.mathjax',           # autodoc with maths
    'sphinx.ext.napoleon',          # For auto-doc configuration
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx_autodoc_typehints'      # Type hints support
]

napoleon_google_docstring = False   # Turn off googledoc strings
napoleon_numpy_docstring = True     # Turn on numpydoc strings
napoleon_use_ivar = True            # For maths symbology

# Autodoc configuration
autodoc_default_options = {
    'members': True,                # Include all class members
    'undoc-members': True,          # Include members without docstrings
    'private-members': False,       # Don't include private members (e.g., _module)
    'special-members': '__str__, __repr__',  # Explicitly list only desired special members
    'show-inheritance': True,       # Show class inheritance information
}

# Exclude __init__.py files and other patterns from documentation
exclude_patterns = ['docs', 'Thumbs.db', '.DS_Store', '**/__init__.py']

# Enable autosummary generation
autosummary_generate = True

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def skip_init(app, what, name, obj, skip, options):
    # Skip __init__ files from documentation
    if what == "module" and (name == "__init__" or name.endswith("__init__")):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_init)
