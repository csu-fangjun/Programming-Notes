# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'Programming Notes'
copyright = '2020, <fangjun dot kuang at gmail dot com>'
author = 'fangjun'

# The full version, including alpha/beta/rc tags
release = 'v0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'breathe',
    'exhale',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'
pygments_style = 'sphinx'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
smartquotes = False
html_show_sourcelink = True

html_theme_options = {
    'collapse_navigation': False,
    'analytics_id': 'UA-160691436-1',
}

# refer to https://docs.readthedocs.io/en/latest/guides/vcs.html#github
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "csu-fangjun",  # Username
    "github_repo": "Programming-Notes",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/",  # Path in the checkout to the docs root
}

# ==============================
# Setup for exhale started
# ------------------------------
# refer to https://exhale.readthedocs.io/en/latest/usage.html#quickstart-guide

# Setup the breathe extension
breathe_projects = {"Notes": "./doxyoutput/xml"}
breathe_default_project = "Notes"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder": "./example_exhale_api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": "..",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ./sphinx/code"
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

# ==============================
# Setup for exhale ended
# ------------------------------
