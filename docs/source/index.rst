Rationale
===================================

Thanks to the unrivalled astrometric and photometric capabilities of the Gaia mission, new impetus has been given to the study of young stars: both from an *environmental* perspective, as members of comoving star-forming regions, and from an *individual* perspective, as targets amenable to planet-hunting direct-imaging observations.

In view of the large availability of theoretical evolutionary models, both fields would benefit from a unified framework that allows a straightforward comparison of physical parameters obtained by different stellar and substellar models.

To this aim, we developed the Manifold Age Determination for Young Stars (**MADYS**, IPA: /ˈmɑːdɪs/), a flexible Python tool for the age and mass determination of young stellar and substellar objects. MADYS automatically *retrieves* and *cross-matches* photometry from several catalogs, estimates interstellar extinction, and derives age and mass estimates for individual objects through isochronal fitting.

Harmonising the heterogeneity of publicly-available isochrone grids, our tool allows to choose amongst 17 models, more than 120 isochrone grids, and more then 200 photometric filters. Several dedicated plotting functions are included to allow a visual perception of the numerical output.

Check out the :doc:`usage` section for further information, including how to perform the :ref:`installation` of the project.

Check  the :doc:`instructions` section.

Features
--------

Bugs
--------

Attribution
--------
Please cite `Squicciarini & Bonavita (2022) <https://ui.adsabs.harvard.edu/abs/2022A%26A...666A..15S/abstract>`_ whenever you publish results obtained with MADYS.

.. note::

   If you wish a new model to be included in a new version of MADYS, or a new set of photometric filters to be added to the current list, feel free to get in contact with us.
 


Contents
--------

.. toctree::

   usage
   api
   instructions
