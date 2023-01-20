Welcome to MADYS' documentation!
===================================

**MADYS** (/lu'make/)  (/ˈmɑːdɪs/) is a *flexible* Python tool for age and mass determination of young *stellar and substellar objects*.

MADYS automatically *retrieves* and *cross-matches* photometry from several catalogs, estimates interstellar extinction, and derives age and mass estimates for individual objects through isochronal fitting.

Harmonising the heterogeneity of publicly-available isochrone grids, the tool allows to choose amongst several models, many of which with customisable astrophysical parameters. Particular attention has been dedicated to the categorization of these models, labeled through a four-level taxonomical classification.

At the moment of writing, MADYS includes 17 models, more than 120 isochrone grids, and more then 200 photometric filters (a thorough description each of them is provided). However, despite our efforts, the model list is far from being complete. If you wish a new model to be included in a new version of MADYS, or a new set of photometric filters to be added to the current list, feel free to get in contact with us.

Finally, several dedicated plotting functions are included to allow a visual perception of the numerical output.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

Check  the :doc:`instructions` section.

.. note::

   This project is under active development.
   
   MADYS has its documentation hosted on Read the Docs.
   


Contents
--------

.. toctree::

   usage
   api
   instructions
