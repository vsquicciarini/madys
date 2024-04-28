Release notes
=====

Version 1.3.0
 * Modified CurveObject class:

   - input .fits header is now inherited by output file;
   - a verbose description of the input setting is stored in the output file's header;
   - possibility to add coronagraphic mask; if set, mask radius is automatically recovered for SPHERE and GPI;
   - age range can now be defined as a Gaussian (specified by 'age' and 'age_error') or a box [age_opt, age_min, age_max];
   - added possibility to compute extinction as usually done in MADYS (integration of 3D maps).
 * Added Gaia DR3 ra and dec error to the automatic research done by SampleObject at initialization.  
 * Added Hipparcos catalog (van Leeuwen et al. 2007) to the automatic research done by SampleObject at initialization.
 * Added long-term Gaia DR3 - Hipparcos proper motion to SampleObject's instance.phot_table computed as in Kervella et al. (2022).
 * Added a new type of 'id_type' for SampleObject instances: 'HIP'. Use it if ALL input stars have input names from the Hipparcos catalog (e.g., HIP 31414). In such cases, setting to 'other' will work, but some cross-matches between Gaia and Hipparcos might not be recovered. 
 * Improved ADQL catalog queries, with higher completeness rates of catalog cross-matches.
 * Improved general readability of the code following Docstring Conventions (PEP 8).

Version 1.2.0
 * A more accurate parameter derivation was introduced when providing [age_opt, age_min, age_max] triplets.
 * SampleObject: default value for keyword 'allwise_cross_match' is now False unless n_objects < 100. 
 * The program now raises a ValueError if no Gaia data are available for all queried objects.
 * Better control of plotting options (transparency, symsize) in SampleObject.CMD.
 * Added verbose description of minimum_error and cuts to FitParams.average_results().
 * Created a new classmethod to SampleObject that allows for merging several instances into a single one.
 * Fixed minor bugs caused by the v1.0.0 -> v1.1.0 upgrade.

Version 1.1.2
 * Added '__eq__' dunder method (equality) to the SampleObject class. Two SampleObject instances are considered equal if the queried objects (as specified by the attribute ID) are the same and have the same ordering.
 * Fixed minor bugs (related to the '__repr__' method) caused by the v1.0.0 -> v1.1.0 upgrade.

Version 1.1.0
 * Created new class, CurveObject, to derive mass limits from contrast curves.
 * Created an attribute of type astropy.Table for SampleObject instances, named quality_table; it contains information on whether and why a photometric measurement has been retained or discarded.
 * Added two methods in FitParams and SampleObject classes that handle import/export of a FitParams instance into a .h5 file.
 * Renamed 'isochrone_grid' attribute in FitParams class as 'exec_command'; added new attribute named 'model_grid', equivalent to that of the ModelHandler class.
 * Speeded up SampleObject.get_params() and ModelHandler.available().
 * Added 'additional_columns' specified in SampleObject.get_params() to output saved to file by FitParams.to_file().
 * Improved the quality of the logs.
 * Changed default value of keywords 'x_log' and 'y_log' from False to True in function IsochroneGrid.plot_iso_grid().
 * Added dundler methods __repr__ and __eq__ to class FitParams.
 * Automatically set keyword 'ext_map' to None if a manual E(B-V) vector is provided in SampleObject initialization.
 * Fixed minor bugs caused by the v0.5.0-beta -> v1.0.0 upgrade.

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
