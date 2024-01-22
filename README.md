
Manifold Age Determination for Young Stars (MADYS) 
==========

Description
-----------
This repository hosts the code of `MADYS`: the Manifold Age Determination for Young Stars, a flexible Python tool for parameter determination of young stellar and substellar objects.

`MADYS` automatically retrieves and cross-matches photometry from several catalogs, estimates interstellar extinction, and derives parameter (age, mass, radius, Teff, logg, logL) estimates for individual objects through isochronal fitting.

Harmonising the heterogeneity of publicly-available isochrone grids, the tool allows to choose amongst several models, many of which with customisable astrophysical parameters. Particular attention has been dedicated to the categorization of these models, labeled through a four-level taxonomical classification.

At the moment of writing, MADYS includes 20 models, 140 isochrone grids, and more then 250 photometric filters (a thorough description of each of them is provided). However, despite our efforts, the model list is far from being complete. If you wish a new model to be included in a new version of `MADYS`, or a new set of photometric filters to be added to the current list, feel free to get in contact with us.

Five classes are defined to handle a large variety of possible applications, spanning from the characterization of directly-imaged planets to the study of stellar associations. Notably, large direct imaging survey will benefit from `MADYS`' capability to compute planetary masses corresponding to detection limits of direct imaging observations.

Finally, several dedicated plotting functions are included to allow a visual perception of the numerical output.

Latest news:
------------
Jan 22, 2023 - Version v1.2.0 published! A more accurate parameter derivation was introduced when providing [age_opt, age_min, age_max] triplets; SampleObject instances can now be merged; a higher degree of control of plotting options is now possible; fixed minor bugs.

Oct 18, 2023 - Version v1.1.0 published! Several new functionalities added: a class to handle the conversion of direct imaging contrast curves into mass curves; functions to easily export/import SampleObject and FitParams instances; a new attribute of SampleObject containing information on photometric quality. Better exploitation of log files; better handling of output files and objects; general improvement of computational performances. 

Oct 10, 2023 - MADYS has now a full documentation on [readthedocs](https://madys.readthedocs.io/en/latest/). Have a look at it!

Sep 09, 2023 - Version v1.0.0 published! Newly added models: Dartmouth (magnetic and non-magnetic, Feiden 2016), solar-metallicity PARSEC v2.0 (Nguyen et al. 2022), latest version of ATMO (Chabrier et al. 2023); added JWST filters to PARSEC (v1.2 and v2.0) isochrones, and Gaia, 2MASS, Panstarrs and HST filters to ATMO 2020. Inserted possibility to estimate synthetic photometry for fitted objects in bands not employed when deriving their parameters.

Feb 17, 2023 - Changed default queried catalog from Gaia DR2 to Gaia DR3 when providing a list of stars with non-Gaia identifiers (i.e., with 'id_type'='other').

Jan 19, 2023 - Added the possibility to take into account uncertainties on E(B-V) values, which can now be provided at inizialization through a dedicated keyword 'ebv_err'.

Aug 03, 2022 - Sloan Digital Sky Survey added to the list of automatically searchable surveys. Its filters are now available with the following models: PARSEC, MIST, AMES-Dusty, AMES-Cond, BT-Settl, NextGen.

Jun 20, 2022 - BEX models (Linder et al. 2019, Marleau et al. 2019) added to the list of available models.

Jun 17, 2022 - Gaia DR3 now available! The new catalog replaces, for all intents and purposes, Gaia EDR3.


Installation:
------------
Catalog queries are mediated by the [TAP Gaia Query](https://github.com/mfouesneau/tap) package (tap). If you import madys from the command line, the module is automatically installed if not found. However, **this does not work from Jupyter Notebook**. We suggest to manually install the package from pip, through:

```sh
pip install git+https://github.com/mfouesneau/tap
```
Please make sure you use the command above, as just using `pip install tap` will download **a different**, although eponymous, package. 

Note that TAP Gaia Query might require the installation of [lxml](https://lxml.de/) (v4.6.3).

Once TAP Gaia Query is installed, the current `MADYS` repository can be installed using pip:

```sh
pip install madys
```
Note that, when using for the first time an extinction map, `MADYS` will download the relevant file (0.7 GB or 2.2 GB, depending on the map).


Requirements
------------

This package relies on usual packages for data science and astronomy: [numpy](https://numpy.org/) (v1.18.1), [scipy](https://www.scipy.org/) (v1.6.1), [pandas](https://pandas.pydata.org/) (v1.1.4), [matplotlib](https://matplotlib.org/) (v3.3.4), [astropy](https://www.astropy.org/) (v4.3.1) and [h5py](https://www.h5py.org/) (v3.2.1).

In addition, it also requires [astroquery](https://github.com/astropy/astroquery/) (v0.4.2.dev0) and [TAP Gaia Query](https://github.com/mbonav/tapGaia) (v0.1). The latter package might require the installation of [lxml](https://lxml.de/) (v4.6.3).


Examples
--------

The package is fully documented on readthedocs.io:

[https://madys.readthedocs.io/en/latest/](https://madys.readthedocs.io/en/latest/) 

and a detailed description of its features, together with several examples of the kind of scientific results that can be obtained with it, is provided in [Squicciarini & Bonavita (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...666A..15S/abstract).

We recommend you check out the [examples](https://github.com/vsquicciarini/madys/blob/main/examples/) provided and the [docs](https://madys.readthedocs.io/en/latest/), for a better understanding of its usage.

If you find a bug or want to suggest improvements, please create a [ticket](https://github.com/vsquicciarini/madys/issues).


Recent papers using `MADYS`:
-----------------------

`MADYS` has already been employed, starting from its preliminary forms, in several publications, including: 

* `An imaged 15 MJup companion within a hierarchical quadruple system`, [Chomez et al. 2023, A&A 676, L10](https://ui.adsabs.harvard.edu/abs/2023A%26A...676L..10C/abstract)
* `BEAST detection of a brown dwarf and a low-mass stellar companion around the young bright B star HIP 81208`, [Viswanath et al. 2023, A&A 675, A54](https://ui.adsabs.harvard.edu/abs/2023A%26A...676L..10C/abstract)
* `Resolved near-UV hydrogen emission lines at 40-Myr super-Jovian protoplanet Delorme 1 (AB)b. Indications of magnetospheric accretion`, [Ringqvist et al. 2023, A&A 669, L12](https://ui.adsabs.harvard.edu/abs/2023A%26A...669L..12R/abstract)
* `Detecting planetary mass companions near the water frost-line using JWST interferometry`, [Ray et al. 2023, MNRAS 519, 2718](https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.2718R/abstract)
* `A scaled-up planetary system around a supernova progenitor`, [Squicciarini et al. 2022, A&A 664, A9](https://ui.adsabs.harvard.edu/abs/2022A%26A...664A...9S/abstract)
* `Results from The COPAINS Pilot Survey: four new brown dwarfs and a high companion detection rate for accelerating stars`, [Bonavita et al. 2022, MNRAS 513, 5588](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.5588B/abstract)
* `A wide-orbit giant planet in the high-mass b Centauri binary system`, [Janson et al. 2021, Natur.600..231J](https://ui.adsabs.harvard.edu/abs/2021Natur.600..231J/abstract)
* `Unveiling the star formation history of the Upper Scorpius association through its kinematics`, [Squicciarini et al. 2021, MNRAS 507, 1381](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.1381S/abstract)
* `New binaries from the SHINE survey`, [Bonavita et al. 2021, arXiv210313706B](https://ui.adsabs.harvard.edu/abs/2021arXiv210313706B/abstract)
* `BEAST begins: sample characteristics and survey performance of the B-star Exoplanet Abundance Study`, [Janson, Squicciarini et al. 2021, A&A 646, A164](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.164J/abstract)

Authors
-----------------------
[Vito Squicciarini](https://orcid.org/0000-0002-3122-6809), LESIA - Observatoire de Paris, FR (vito.squicciarini@obspm.fr)

[Mariangela Bonavita](https://orcid.org/0000-0002-7520-8389), The Open University, UK

We are grateful for your effort, and hope that these tools will contribute to your scientific work and discoveries. Please feel free to report any bug or possible improvement to the authors.

Attribution
-----------------------
Please cite [Squicciarini & Bonavita (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...666A..15S/abstract) whenever you publish results obtained with `MADYS`.


