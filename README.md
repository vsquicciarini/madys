
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

