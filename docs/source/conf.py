# type: ignore
# Configuration file for the Sphinx documentation builder.

# -- For some crazy reason this is a requirement to import the package
from typing import Any, List, Tuple, Type

import types
import os
from re import A
import sys
import shutil
import glob

# import torch

import moonshine.models
import moonshine.preprocessing

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
html_static_path = ["_static"]
html_title = " Moonshine"

# Favicon
html_favicon = "https://moonshine-assets.s3.us-west-2.amazonaws.com/favicon.ico"

# Customize CSS
html_css_files = ["css/custom.css"]

# -- Options for HTML output
html_theme = "furo"
html_theme_options = {
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-light-mode.png",
    "light_css_variables": {
        "color-brand-primary": "#373737",
        "color-brand-content": "#373737",
    },
    "dark_css_variables": {
        "color-brand-primary": "#f9f9f9",
        "color-brand-content": "#f9f9f9",
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
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

pygments_style = "manni"
pygments_dark_style = "monokai"

autosummary_generate = True

nbsphinx_execute = "never"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

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


def _auto_rst_for_module(module: types.ModuleType, exclude_members: List[Any]) -> str:
    """Generate the content of an rst file documenting a module.
    Includes the module docstring, followed by tables for the functions,
    classes, and exceptions
    Args:
        module: The module object to document
        exclude_members: A list of Python objects to exclude from the
            documentation. Providing objects that are not imported in
            ``module`` are ignored.
    Returns:
        The rst content for the module
    """
    name = module.__name__
    lines = []

    functions: List[Tuple[str, types.FunctionType]] = []
    exceptions: List[Tuple[str, Type[BaseException]]] = []
    classes: List[Tuple[str, Type[object]]] = []
    methods: List[Tuple[str, types.MethodType]] = []
    attributes: List[Tuple[str, object]] = []

    # add title and module docstring
    lines.append(f"{name}")
    lines.append(f'{"=" * len(name)}\n')
    lines.append(f".. automodule:: {name}\n")

    # set prefix so that we can use short names in the autosummaries
    lines.append(f".. currentmodule:: {name}")

    try:
        all_members = list(module.__all__)
    except AttributeError:
        all_members = list(vars(module).keys())

    for item_name, val in vars(module).items():
        if val in exclude_members:
            continue

        if item_name.startswith("_"):
            # Skip private members
            continue

        if item_name not in all_members:
            # Skip members not in `__all__``
            continue

        if isinstance(val, types.ModuleType):
            # Skip modules; those are documented by autosummary
            continue

        if isinstance(val, types.FunctionType):
            functions.append((item_name, val))
        elif isinstance(val, types.MethodType):
            methods.append((item_name, val))
        elif isinstance(val, type) and issubclass(val, BaseException):
            exceptions.append((item_name, val))
        elif isinstance(val, type):
            assert issubclass(val, object)
            classes.append((item_name, val))
        else:
            attributes.append((item_name, val))
            continue

    # Sort by the reimported name
    functions.sort(key=lambda x: x[0])
    exceptions.sort(key=lambda x: x[0])
    classes.sort(key=lambda x: x[0])
    attributes.sort(key=lambda x: x[0])

    for category, category_name in (
        (functions, "Functions"),
        (classes, "Classes"),
        (exceptions, "Exceptions"),
    ):
        sphinx_lines = []
        for item_name, _ in category:
            sphinx_lines.append(f"      {item_name}")
        if len(sphinx_lines) > 0:
            lines.append(f"\n.. rubric:: {category_name}\n")
            lines.append(".. autosummary::")
            lines.append("      :toctree: generated")
            lines.append("      :nosignatures:")
            if category_name in ("Classes"):
                lines.append("      :template: classtemplate.rst")
            elif category_name == "Functions":
                lines.append("      :template: functemplate.rst")
            lines.append("")
            lines.extend(sphinx_lines)
            lines.append("")

    lines.append(".. This file autogenerated by docs/source/conf.py\n")

    return "\n".join(lines)


def _modules_to_rst() -> List[types.ModuleType]:
    """Return the list of modules for which to generate API reference rst files."""
    document_modules: List[types.Module] = [
        moonshine.models,
        moonshine.preprocessing,
    ]
    exclude_modules: List[types.Module] = [moonshine]
    for name in moonshine.__dict__:
        obj = moonshine.__dict__[name]
        if isinstance(obj, types.ModuleType) and obj not in exclude_modules:
            document_modules.append(obj)

    return document_modules


def _generate_rst_files_for_modules() -> None:
    """Generate .rst files for each module to include in the API reference.
    These files contain the module docstring followed by tables listing all the functions, classes,
    etc.
    """
    docs_dir = os.path.abspath(os.path.dirname(__file__))
    module_rst_save_dir = os.path.join(docs_dir, "api_reference")
    # gather up modules to generate rst files for
    document_modules = _modules_to_rst()

    # rip out types that are duplicated in top-level moonshine module
    moonshine_imported_types = []
    for name in moonshine.__all__:
        obj = moonshine.__dict__[name]
        if not isinstance(obj, types.ModuleType):
            moonshine_imported_types.append(obj)

    document_modules = sorted(document_modules, key=lambda x: x.__name__)
    os.makedirs(module_rst_save_dir, exist_ok=True)
    for module in document_modules:
        saveas = os.path.join(module_rst_save_dir, module.__name__ + ".rst")
        print(f"Generating rst file {saveas} for module: {module.__name__}")

        # avoid duplicate entries in docs. We add torch's _LRScheduler to
        # types, so we get a ``WARNING: duplicate object description`` if we
        # don't exclude it
        # exclude_members = [torch.optim.lr_scheduler._LRScheduler]
        exclude_members = []
        if module is not moonshine:
            exclude_members += moonshine_imported_types

        content = _auto_rst_for_module(module, exclude_members=exclude_members)

        with open(saveas, "w") as f:
            f.write(content)


_generate_rst_files_for_modules()
