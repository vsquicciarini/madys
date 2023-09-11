Release notes
=====

Version 1.0.0
 * Added Dartmouth (magnetic and non magnetic) models (Feiden 2016).
 * Added new version of ATMO 2020 with a new EOS (Chabrier 2023), called ATMO 2023.
 * Added new PARSEC v2.0 isochrones (Nguyen et al. 2022) including rotation; only solar metallicity tracks are currently supported.
 * Added Gaia, 2MASS, Panstarrs and HST filters to ATMO 2020 grids.
 * Added JWST filters to PARSEC grids.
 * Fixed syntax of Gaia ADQL queries after latest modifications by the Gaia team.
 * An error is now raised if the syntax of a SQL query is wrong (before it used to enter an infinite loop).
 * Inserted new optional keyword ``additional_columns`` in method :py:func:`SampleObject.get_params` that allows computation of synthetic photometry in different bands from those used for the fit.
 * Deleted optional keyword ``phys_params`` in method :py:func:`SampleObject.get_params`; now physical parameters (logL, logg, radius, Teff) are always computed.
 * Renamed some attributes of the ``FitParams`` class using the singular form: e.g., "ages" -> "age", "masses" -> "mass".
 * Added a keyword ``show_plot`` in :py:func:`IsochroneGrid.plot_iso_grid` not to show the plot (useful when saving the output to a file).
 * Edited default file name and inserted possibility to save several outputs at once in the function :py:func:`FitParams.plot_maps`.
 * Modified the function :py:func:`info_filters` to allow for an easier visualization of all available photometric filters.
 * Function :py:func:`FitParams.to_table` now inherits keywords of :py:func:`astropy.table.Table`.
 * Fixed wrong label in colorbar of weight map created by the function :py:func:`FitParams.plot_maps` .
 * Fixed a bug which resulted in an error if one tried to use models with a single set of astrophysical parameters with different values of the same parameters (e.g., BHAC15 with an array of FeH values).
 * Fixed a bug that impeded using starevol model in particular conditions due to its default metallicity being -0.01 and not 0.00.
 * Renamed model family of pm13 suite: from PM13 to empirical.
 * Fixed a bug in :py:func:`IsochroneGrid.plot_iso_grid` and :py:func:`IsochroneGrid.plot_isochrones` that did not allow plot of models with undefined age (i.e., pm13).

Version 0.5.0-beta
 * Added the possibility to take into account uncertainties on E(B-V) values, which can now be provided at inizialization through a dedicated keyword 'ebv_err'.
 * Fixed bug preventing in a few cases to provide as input an array of multiple FeH.

Version 0.4.1-beta
 * Sloan Digital Sky Survey added to the list of queryable surveys. Its filters are now available with the following models: PARSEC, MIST, AMES-Dusty, AMES-Cond, BT-Settl, NextGen.
 * Inserted possibility to obtain information about available filters for a certain model.
 * Fixed bug preventing overplotting of tracks upon isochrones in the function plot_isochrones().
 * Fixed bug impeding the correct handling of missing PANSTARRS filters.

Version 0.3.1-beta
 * BEX models (Linder et al. 2019, Marleau et al. 2019) added to the list of available models.
 * Gaia DR3 now available! The new catalog replaces, for all intents and purposes, Gaia EDR3.
