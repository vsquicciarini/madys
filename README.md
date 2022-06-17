
Manifold Age Determination for Young Stars (MADYS) 
==========

Description
-----------
This repository includes the first version of `MADYS`: the Manifold Age Determination for Young Stars, a flexible Python tool for age and mass determination of young stellar and substellar objects.

`MADYS` automatically retrieves and cross-matches photometry from several catalogs, estimates interstellar extinction, and derives age and mass estimates for individual objects through isochronal fitting.

Harmonising the heterogeneity of publicly-available isochrone grids, the tool allows to choose amongst several models, many of which with customisable astrophysical parameters. Particular attention has been dedicated to the categorization of these models, labeled through a four-level taxonomical classification.

Note that, since this is an alpha version, it only includes a sub-set of isochrones.
The full set, with 16 models and more than 100 isochrone grids, will be included the first full release, expected for July 2022.

Despite our efforts, the model list is far from being complete. If you wish a new model to be included in a new version of `MADYS`, feel free to get in contact with us.

A thorough description of photometric filters featured in `MADYS` is provided; finally, several dedicated plotting functions are included to allow a visual perception of the numerical output.


Latest news:
------------
Jun 17, 2022 - Gaia DR3 now available! The new catalog replaces, for all intents and purposes, Gaia EDR3.


Installation:
------------

The current repository can be installed using pip:

```sh
pip install madys
```

Note that, when using for the first time an extinction map, `MADYS` will download the relevant file (0.7 GB or 2.2 GB, depending on the map).

Requirements
------------

This package relies on usual packages for data science and astronomy: [numpy](https://numpy.org/) (v1.18.1), [scipy](https://www.scipy.org/) (v1.6.1), [pandas](https://pandas.pydata.org/) (v1.1.4), [matplotlib](https://matplotlib.org/) (v3.3.4), [astropy](https://www.astropy.org/) (v4.3.1) and [h5py](https://www.h5py.org/) (v3.2.1).

In addition, it also requires [astroquery](https://github.com/astropy/astroquery/) (v0.4.2.dev0), [tabulate](https://pypi.org/project/tabulate/) (v0.8.9) and [TAP Gaia Query](https://github.com/mfouesneau/tap) (v0.1). The last package might require the installation of [lxml](https://lxml.de/) (v4.6.3).


Known issues
------------
Catalog queries are mediated by the [TAP Gaia Query](https://github.com/mfouesneau/tap) package (tap). The command 

```sh
pip install tap
```

must not be executed, because it would download **a different**, although eponymous, package. The right command to install the package is instead:

```sh
pip install  git+https://github.com/mfouesneau/tap
```


Examples
--------

The package is fully documented and a detailed description of its features, together with several examples of the kind of scientific results that can be obtained with it, is provided in [Squicciarini & Bonavita 2022 arXiv:2206.02446](https://arxiv.org/abs/2206.02446)

However, we recommend you check out the [examples](https://github.com/vsquicciarini/madys/blob/main/examples/) provided, for a better understanding of its usage.

If you find a bug or want to suggest improvements, please create a [ticket](https://github.com/vsquicciarini/madys/issues).


Recent papers using `MADYS`:
-----------------------

`MADYS` has already been used, in its various preliminary forms, for several publications, including: 

* `A scaled-up planetary system around a supernova progenitor`, [Squicciarini et al. 2022 arXiv220502279S](https://ui.adsabs.harvard.edu/abs/2022arXiv220502279S/abstract)
* `Results from The COPAINS Pilot Survey: four new brown dwarfs and a high companion detection rate for accelerating stars`, [Bonavita et al. 2022 MNRAS.513.5588B](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.5588B/abstract)
* `A wide-orbit giant planet in the high-mass b Centauri binary system`, [Janson et al. 2021 Natur.600..231J](https://ui.adsabs.harvard.edu/abs/2021Natur.600..231J/abstract)
* `Unveiling the star formation history of the Upper Scorpius association through its kinematics`, [Squicciarini et al. 2021 MNRAS.507.1381S](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.1381S/abstract)
* `New binaries from the SHINE survey`, [Bonavita et al. 2021 arXiv210313706B](https://ui.adsabs.harvard.edu/abs/2021arXiv210313706B/abstract)
* `BEAST begins: sample characteristics and survey performance of the B-star Exoplanet Abundance Study`,[Janson, Squicciarini et al. 2021 A&A...646A.164J](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.164J/abstract)

Authors
-----------------------
[Vito Squicciarini](https://orcid.org/0000-0002-3122-6809), University of Padova, IT

[Mariangela Bonavita](https://orcid.org/0000-0002-7520-8389), The Open University, UK

We are grateful for your effort, and hope that these tools will contribute to your scientific work and discoveries. Please feel free to report any bug or possible improvement to the authors.

Attribution
-----------------------
Please cite [Squicciarini & Bonavita 2022 arXiv:2206.02446](https://arxiv.org/abs/2206.02446) whenever you publish results obtained with `MADYS`.


