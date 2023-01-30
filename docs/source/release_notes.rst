Release notes
=====

Version 0.5.0-beta
 * Added the possibility to take into account uncertainties on E(B-V) values, which can now be provided at inizialization through a dedicated keyword 'ebv_err'.
 * Fixed bug preventing in a few cases to provide as input an array of multiple FeH.

Version 0.4.1-beta
 * Sloan Digital Sky Survey added to the list of queryable surveys. Its filters are now available with the following models: PARSEC, MIST, AMES-Dusty, AMES-Cond, BT-Settl, NextGen.
 * Inserted possibility to obtain information about available filters for a certain model.
 * Fixed bug preventing overplotting of tracks upon isochrones in the function plot_isochrones().
 * Fixed bug impeding the correct handling of missing PANSTARRS filters.

Version 0.3.1
 * BEX models (Linder et al. 2019, Marleau et al. 2019) added to the list of available models.
 * Gaia DR3 now available! The new catalog replaces, for all intents and purposes, Gaia EDR3.
