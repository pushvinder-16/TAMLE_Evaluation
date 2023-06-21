# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import os.path as op
import sys

HERE = op.dirname(op.abspath(__file__))
LIB_PKG_PATH = op.abspath(op.join(HERE, "..", "..", "src"))
sys.path.insert(0, LIB_PKG_PATH)

# NOTE: This is needed for jupyter-sphinx to be able to build docs
os.environ["PYTHONPATH"] = ":".join((LIB_PKG_PATH, os.environ.get("PYTHONPATH", "")))


# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

# -- Project information -----------------------------------------------------

project = "House price prediction"
copyright = "2023, Pushvinder Kumar"
author = "Pushvinder Kumar"
release = "1.0.0"
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     # "sphinxcontrib.napoleon",
#     "sphinx.ext.napoleon",
#     "sphinx.ext.autodoc",
#     "sphinx.ext.autosummary",
#     "sphinx_rtd_theme",
#     "sphinx.ext.todo",
#     "sphinx.ext.extlinks",
#     "nbsphinx",
#     "jupyter_sphinx",
#     # "sphinx.ext.viewcode",
# ]

autosummary_generate = True
autosummary_imported_members = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]

autodoc_member_order = "bysource"
autodoc_mock_imports = ["theano", "past"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
source_suffix = ".rst"
todo_include_todos = True
html_css_files = ["_static/css/custom.css"]

imgmath_font_size = 12

numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}
