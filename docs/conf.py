import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'PhilanthroPy'
copyright = '2026, PhilanthroPy Contributors'
author = 'PhilanthroPy Contributors'

import philanthropy
version = philanthropy.__version__
release = philanthropy.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.linkcode',
    'sphinx_gallery.gen_gallery',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PhilanthroPy-Project/PhilanthroPy",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_prev_next": False,
    "navbar_align": "content",
    "search_bar_text": "Search the docs...",
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

# -- Extension configuration -------------------------------------------------

# Numpydoc settings
numpydoc_show_class_members = False

# Sphinx-Gallery settings
sphinx_gallery_conf = {
     'examples_dirs': 'examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'reference_url': {
         # The module you locally document uses None
         'philanthropy': None,
     },
}

# Intersphinx settings removed to prevent dbm errors in GitHub Actions.


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != 'py':
        return None
    
    modname = info['module']
    fullname = info['fullname']
    
    if not modname:
        return None
    
    return "https://github.com/PhilanthroPy-Project/PhilanthroPy/blob/main/%s.py" % modname.replace('.', '/')
