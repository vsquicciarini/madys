
Manifold Age Determination for Young Stars (MADYS) 
==========

Information
-----------
This repository includes the first version of `MADYS`: the Manifold Age Determination for Young Stars, a flexible Python tool for age and mass determination of young stellar and substellar objects. 

In this first release, `MADYS` automatically retrieves and cross-matches photometry from several catalogs, estimates interstellar extinction, and derives age and mass estimates for individual stars through isochronal fitting.

Harmonizing the heterogeneity of publicly-available isochrone grids, the tool allows to choose amongst 17 models, many of which with customizable astrophysical parameters, for a total of $\sim 110$ isochrone grids. Several dedicated plotting function are provided to allow a visual perception of the numerical output.

Requirements
------------

This package relies on usual packages for data science and astronomy: [numpy](https://numpy.org/), [scipy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/) and [astropy](https://www.astropy.org/).

In addition, it also requires [astroquery](https://github.com/astropy/astroquery/) and [TAP Gaia Query](https://github.com/mfouesneau/tap).

# Installation: 
Download the current repository, unpack it and install the package manually:

```sh
gunzip madys.v0.1.0.zip
cd madys/
python setup.py install
```

Then download the latest version of the isochrones and extinction map from zenodo, using [zenodo_get](https://github.com/dvolgyes/zenodo_get) 

(N.B. you will need at least 4.5GB of space for these files):
```sh
pip install zenodo-get
zenodo_get RECORD_ID_OR_DOI
```

and unpack them: 
```sh
gunzip extinction.zip
gunzip isochrones.zip 
```

Please make sure that once you have unpacked the files, the madys directory has to include the following files/directories:
```sh
madys/
└── isochrones/
└── extinction/
└── evolution.py
└── madys.py 
```

Examples
--------

The package is not fully documented, but [examples]() are provided.

If you find a bug or want to suggest improvements, please create a [ticket](https://github.com/vsquicciarini/madys/issues)


Recent papers using `MADYS`: 
-----------------------
* `A scaled-up planetary system around a supernova progenitor`, [Squicciarini et al. 2022 arXiv220502279S](https://ui.adsabs.harvard.edu/abs/2022arXiv220502279S/abstract)
* `A wide-orbit giant planet in the high-mass b Centauri binary system`, [Janson et al. 2022 2021Natur.600..231J](https://ui.adsabs.harvard.edu/abs/2021Natur.600..231J/abstract)
* `Unveiling the star formation history of the Upper Scorpius association through its kinematics`, [Squicciarini et al. 2022 2021MNRAS.507.1381S](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.1381S/abstract)
* `New binaries from the SHINE survey`, [Bonavita et al. 2021arXiv210313706B](https://ui.adsabs.harvard.edu/abs/2021arXiv210313706B/abstract)
* `BEAST begins: sample characteristics and survey performance of the B-star Exoplanet Abundance Study`,[Janson, Squicciarini et al. 2021A&A...646A.164J](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.164J/abstract)

Author and contributors
-----------------------
[Vito Squicciarini](https://orcid.org/0000-0002-3122-6809), University of Padova, IT

[Mariangela Bonavita](https://orcid.org/0000-0002-7520-8389), The Open University, UK 

We are grateful for your effort, and hope that these tools will contribute to your scientific work and discoveries. Please feel free to report any bug or possible improvement to the author(s).

Attribution
-----------------------
Please cite [Squicciarini & Bonavita 2022]() whenever you publish results obtained with the Exo-DMC. 
Astrophysics Source Code Library reference [ascl:XXXX.XXXX]()

