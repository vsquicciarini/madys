Available filters
=====

More than 250 photometric filters are currently available in ``MADYS``. The full set of filters that can be currently handled by the ``SampleObject`` class is reported in the Table below and accessible by calling the function :py:func:`info_filters`:


.. code-block:: python

   madys.info_filters()

while information on a specific filter X can be returned by calling the same function with argument 'X':

.. code-block:: python

   madys.info_filters('J')

.. code-block::

  Quantity name: 'J'
  Description: 2MASS J-band filter
  Reference: Skrutskie et al., AJ 131, 1163 (2006)
  Available in the following models: BHAC15, Geneva, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SB12, Sonora Bobcat, SPOTS, STAREVOL, PM13
  Wavelength: 1.234 micron
  Absolute extinction A(l)/A(V): 0.243

Please notice the case-sensitive spelling.

.. note::

   In order for a filter to be employed for parameter determination, it has to be supported by the model of interest. Please check it beforehand.

.. list-table:: Weather forecast
   :header-rows: 1
   :widths: 7 7 7 7 60 40 80
   :stub-columns: 1

   *  -  Filter
      -  Survey
      -  Mean wvl (MICRON)
      -  Abs. coeff.
      -  Description
      -  Reference
      -  Models
   *  -  B
      -  bessell
      -  0.4525
      -  1.317
      -  Johnson-Cousins B-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  Geneva, MIST, PARSEC, BT-Settl, SPOTS, STAREVOL, PM13
   *  -  B_H
      -  bessell
      -  1.63
      -  0.1354
      -  Johnson-Cousins H-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, SB12, BHAC15
   *  -  B_J
      -  bessell
      -  1.22
      -  0.2466
      -  Johnson-Cousins J-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, SB12, BHAC15
   *  -  B_K
      -  bessell
      -  2.19
      -  0.0752
      -  Johnson-Cousins K-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, SB12, BHAC15
   *  -  Bx
      -  bessell
      -  0.4537
      -  1.268
      -  Johnson-Cousins B_X-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  PARSEC
   *  -  CFHT_H
      -  cfht
      -  1.624
      -  0.1364
      -  WIRCAM @ Canada-France-Hawaii Telescope H-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  CFHT_J
      -  cfht
      -  1.252
      -  0.2338
      -  WIRCAM @ Canada-France-Hawaii Telescope J-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  CFHT_K
      -  cfht
      -  2.143
      -  0.07813
      -  WIRCAM @ Canada-France-Hawaii Telescope Ks-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  CFHT_Y
      -  cfht
      -  1.024
      -  0.3543
      -  WIRCAM @ Canada-France-Hawaii Telescope Y-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  CFHT_Z
      -  cfht
      -  0.8793
      -  0.4707
      -  WIRCAM @ Canada-France-Hawaii Telescope Z-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  CFHT_CH4ON
      -  cfht
      -  1.691
      -  0.1255
      -  WIRCAM @ Canada-France-Hawaii Telescope CH4_on-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  CFHT_CH4OFF
      -  cfht
      -  1.588
      -  0.1428
      -  WIRCAM @ Canada-France-Hawaii Telescope CH4_off-band filter
      -  Puget et al., Proceedings of the SPIE, 5492, 978 (2004)
      -  BHAC15
   *  -  D51
      -  kepler
      -  0.51
      -  1.103
      -  Dunlap Observatory DD51 filter (510 nm)
      -  Brown et al., AJ 142, 112 (2011)
      -  MIST
   *  -  g
      -  panstarrs
      -  0.4957
      -  1.155
      -  Panstarrs DR1 g-band filter
      -  Magnier et al., ApJS, 251, 6 (2020)
      -  BHAC15, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  G
      -  gaia
      -  0.6419
      -  0.789
      -  Gaia (E)DR3 G-band filter
      -  Riello et al., A&A, 649, A3 (2021)
      -  MIST, PARSEC
   *  -  G2
      -  gaia
      -  0.6419
      -  0.789
      -  Gaia DR2 G-band filter
      -  Evans et al., A&A, 616, A4 (2018)
      -  BHAC15, Geneva, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SPOTS, STAREVOL, PM13
   *  -  Gbp
      -  gaia
      -  0.5387
      -  1.002
      -  Gaia (E)DR3 Gbp-band filter
      -  Riello et al., A&A, 649, A3 (2021)
      -  MIST, PARSEC
   *  -  Gbp2
      -  gaia
      -  0.5387
      -  1.002
      -  Gaia DR2 Gbp-band filter
      -  Evans et al., A&A, 616, A4 (2018)
      -  BHAC15, Geneva, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SPOTS, STAREVOL, PM13
   *  -  Grp
      -  gaia
      -  0.7667
      -  0.589
      -  Gaia (E)DR3 Grp-band filter
      -  Riello et al., A&A, 649, A3 (2021)
      -  MIST, PARSEC
   *  -  Grp2
      -  gaia
      -  0.7667
      -  0.589
      -  Gaia DR2 Grp-band filter
      -  Evans et al., A&A, 616, A4 (2018)
      -  BHAC15, Geneva, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SPOTS, STAREVOL, PM13
   *  -  H
      -  2mass
      -  1.639
      -  0.131
      -  2MASS H-band filter
      -  Skrutskie et al., AJ 131, 1163 (2006)
      -  BHAC15, Geneva, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SB12, Sonora Bobcat, SPOTS, STAREVOL, PM13
   *  -  H_F090M
      -  hst
      -  0.9035
      -  0.4481
      -  NICMOS1 @ Hubble Space Telescope F090M filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F110W
      -  hst
      -  1.117
      -  0.2966
      -  NICMOS1/NICMOS2/NICMOS3/WFC3_IR @ Hubble Space Telescope F110W filter
      -  Viana et al., HST Instrument Handbook (2009), Kimble et al., SPIE 7010, 70101E (2008)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F160W
      -  hst
      -  1.526
      -  0.1556
      -  NICMOS1/NICMOS2/NICMOS3 @ Hubble Space Telescope F160W filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F165M
      -  hst
      -  1.648
      -  0.1323
      -  NICMOS1 @ Hubble Space Telescope F165M filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F187W
      -  hst
      -  1.871
      -  0.1017
      -  NICMOS2 @ Hubble Space Telescope F187W filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F205W
      -  hst
      -  2.064
      -  0.08368
      -  NICMOS2 @ Hubble Space Telescope F205W filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F207M
      -  hst
      -  2.082
      -  0.08233
      -  NICMOS2 @ Hubble Space Telescope F207M filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F222M
      -  hst
      -  2.218
      -  0.07357
      -  NICMOS2/NICMOS3 @ Hubble Space Telescope F222M filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F237M
      -  hst
      -  2.369
      -  0.06569
      -  NICMOS2 @ Hubble Space Telescope F237M filter
      -  Viana et al., HST Instrument Handbook (2009)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F253M
      -  hst
      -  0.2549
      -  2.242
      -  FOC (f/96 detector) @ Hubble Space Telescope F253M filter
      -  Paresce, Hubble Space Telescope: Faint object camera instrument handbook. Version 2.0 (1990)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F300W
      -  hst
      -  0.2985
      -  1.825
      -  WFPC2-PC @ Hubble Space Telescope F300W filter
      -  Holtzman et al., PASP 107, 1065 (1995)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F336W
      -  hst
      -  0.3344
      -  1.665
      -  WFC3_UVIS1/WFC3_UVIS2/WFPC1-PC/WFPC1-WF/WFPC2-PC @ Hubble Space Telescope F336W filter
      -  Kimble et al., SPIE 7010, 70101E (2008); Holtzman et al., PASP 107, 1065 (1995); Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F346M
      -  hst
      -  0.3475
      -  1.618
      -  FOC (f/96 detector) @ Hubble Space Telescope F346M filter
      -  Paresce, Hubble Space Telescope: Faint object camera instrument handbook. Version 2.0 (1990)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F439W
      -  hst
      -  0.4312
      -  1.338
      -  WFPC1-PC/WFPC1-WF @ Hubble Space Telescope F439W filter
      -  Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F555W
      -  hst
      -  0.5443
      -  1.013
      -  ACS_HRC/WFC3_UVIS1/WFC3_UVIS2/WFPC1-PC/WFPC1-WF/WFPC2-PC @ Hubble Space Telescope F555W filter
      -  Ford et al., Proc. SPIE 2807, 184 (1996); Kimble et al., SPIE 7010, 70101E (2008); Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990); Holtzman et al., PASP 107, 1065 (1995)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F606W
      -  hst
      -  0.6001
      -  0.8825
      -  ACS_HRC/ACS_WFC/WFC3_UVIS1/WFC3_UVIS2/WFPC1-PC/WFPC1-WF/WFPC2-PC @ Hubble Space Telescope F606W filter
      -  Ford et al., Proc. SPIE 2807, 184 (1996); Kimble et al., SPIE 7010, 70101E (2008); Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990); Holtzman et al., PASP 107, 1065 (1995)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F675W
      -  hst
      -  0.6718
      -  0.7431
      -  WFPC1-PC/WFPC1-WF/WFPC2-PC @ Hubble Space Telescope F675W filter
      -  Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990); Holtzman et al., PASP 107, 1065 (1995)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F785LP
      -  hst
      -  0.8687
      -  0.4811
      -  WFPC1-PC/WFPC1-WF/WFPC2-PC @ Hubble Space Telescope F785LP filter
      -  Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990); Holtzman et al., PASP 107, 1065 (1995)
      -  AMES-Cond, AMES-Dusty
   *  -  H_F814W
      -  hst
      -  0.7996
      -  0.5568
      -  ACS_HRC/ACS_WFC/WFC3_UVIS1/WFC3_UVIS2/WFPC1-PC/WFPC1-WF @ Hubble Space Telescope F675W filter
      -  Ford et al., Proc. SPIE 2807, 184 (1996); Kimble et al., SPIE 7010, 70101E (2008); Griffiths, Hubble Space Telescope: Wide field and planetary camera instrument handbook. Version 2.1 (1990)
      -  AMES-Cond, AMES-Dusty
   *  -  Hp
      -  hipparcos
      -  0.5025
      -  1.124
      -  Hipparcos Hp-band filter
      -  Perryman et al., A&A 323, 49 (1997)
      -  MIST
   *  -  I
      -  bessell
      -  0.8028
      -  0.5529
      -  Johnson-Cousins I-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  Geneva, MIST, PARSEC, BT-Settl, SPOTS, STAREVOL, PM13, BHAC15, BEX
   *  -  i
      -  panstarrs
      -  0.7522
      -  0.628
      -  Panstarrs DR1 i-band filter
      -  Magnier et al., ApJS, 251, 6 (2020)
      -  BHAC15, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  I_c
      -  tess
      -  0.7698
      -  0.5943
      -  TESS I_c-band filter
      -  Ricker et al., JATIS 1, 014003 (2015)
      -  MIST
   *  -  IRAC1
      -  spitzer
      -  3.538
      -  0.03723
      -  Spitzer IRAC-Channel 1 filter (3.6 micron)
      -  Fazio et al., ApJS, 154, 10 (2004)
      -  ATMO2020, BHAC15, PARSEC
   *  -  IRAC2
      -  spitzer
      -  4.478
      -  0.02958
      -  Spitzer IRAC-Channel 2 filter (4.5 micron)
      -  Fazio et al., ApJS, 154, 10 (2004)
      -  ATMO2020, BHAC15, PARSEC
   *  -  IRAC3
      -  spitzer
      -  5.696
      -  0.02553
      -  Spitzer IRAC-Channel 3 filter (5.8 micron)
      -  Fazio et al., ApJS, 154, 10 (2004)
      -  BHAC15, PARSEC
   *  -  IRAC4
      -  spitzer
      -  7.798
      -  0.02743
      -  Spitzer IRAC-Channel 4 filter (8.0 micron)
      -  Fazio et al., ApJS, 154, 10 (2004)
      -  BHAC15, PARSEC
   *  -  IRSblue
      -  spitzer
      -  15.77
      -  0.02775
      -  Spitzer IRS blue channel
      -  Houck, ApJS 154, 18 (2004)
      -  BHAC15
   *  -  IRSred
      -  spitzer
      -  22.48
      -  0.0338
      -  Spitzer IRS red channel
      -  Houck, ApJS 154, 18 (2004)
      -  BHAC15
   *  -  J
      -  2mass
      -  1.234
      -  0.243
      -  2MASS J-band filter
      -  Skrutskie et al., AJ 131, 1163 (2006)
      -  BHAC15, Geneva, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SB12, Sonora Bobcat, SPOTS, STAREVOL, PM13
   *  -  K
      -  2mass
      -  2.176
      -  0.078
      -  2MASS K-band filter
      -  Skrutskie et al., AJ 131, 1163 (2006)
      -  BHAC15, Geneva, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, SB12, Sonora Bobcat, SPOTS, STAREVOL, PM13
   *  -  Kp
      -  kepler
      -  0.6303
      -  0.8202
      -  Kepler K_p-band filter
      -  Koch et al., ApJ 713, 79 (2010)
      -  MIST
   *  -  L
      -  bessell
      -  3.45
      -  0.03833
      -  Johnson-Cousins L-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, BHAC15
   *  -  logg
      -  hr
      -  -
      -  -
      -  surface gravity of the object [log10[gcm-2]]
      -  -
      -  all
   *  -  logL
      -  hr
      -  -
      -  -
      -  bolometric luminosity of the object [log10[L_sun]]
      -  -
      -  all
   *  -  logR
      -  hr
      -  -
      -  -
      -  log10(radius) of the object (unit: R_sun or R_jup)
      -  -
      -  all
   *  -  logT
      -  hr
      -  -
      -  -
      -  log10(effective temperature) of the object (unit: K)
      -  -
      -  all
   *  -  Lp
      -  bessell
      -  3.8
      -  0.03444
      -  Johnson-Cousins L'-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, SB12, BHAC15
   *  -  M
      -  bessell
      -  4.75
      -  0.02827
      -  Johnson-Cousins K-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, SB12, BHAC15
   *  -  MIPS24
      -  spitzer
      -  23.59
      -  0.03239
      -  Spitzer MIPS band at 24 micron
      -  Rieke et al., ApJS 154, 25 (2004)
      -  BHAC15, PARSEC
   *  -  MIPS70
      -  spitzer
      -  70.89
      -  -
      -  Spitzer MIPS band at 70 micron
      -  Rieke et al., ApJS 154, 25 (2004)
      -  BHAC15, PARSEC
   *  -  MIPS160
      -  spitzer
      -  155.0
      -  -
      -  Spitzer MIPS band at 160 micron
      -  Rieke et al., ApJS 154, 25 (2004)
      -  BHAC15, PARSEC
   *  -  MIRI_c_F1065C
      -  jwst_miri_c
      -  10.56
      -  0.06661
      -  MIRI @ James Webb Space Telescope F1065C coronagraphic filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020
   *  -  MIRI_c_F1140C
      -  jwst_miri_c
      -  11.31
      -  0.04762
      -  MIRI @ James Webb Space Telescope F1140C coronagraphic filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020
   *  -  MIRI_c_F1550C
      -  jwst_miri_c
      -  15.52
      -  0.02682
      -  MIRI @ James Webb Space Telescope F1550C coronagraphic filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020
   *  -  MIRI_c_F2300C
      -  jwst_miri_c
      -  22.64
      -  0.03358
      -  MIRI @ James Webb Space Telescope F2300C coronagraphic filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020
   *  -  MIRI_p_F560W
      -  jwst_miri_p
      -  5.635
      -  0.02564
      -  MIRI @ James Webb Space Telescope F560W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F770W
      -  jwst_miri_p
      -  7.639
      -  0.02627
      -  MIRI @ James Webb Space Telescope F770W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F1000W
      -  jwst_miri_p
      -  9.953
      -  0.08093
      -  MIRI @ James Webb Space Telescope F1000W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F1130W
      -  jwst_miri_p
      -  11.31
      -  0.04765
      -  MIRI @ James Webb Space Telescope F1130W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15
   *  -  MIRI_p_F1280W
      -  jwst_miri_p
      -  12.81
      -  0.02974
      -  MIRI @ James Webb Space Telescope F1280W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F1500W
      -  jwst_miri_p
      -  15.06
      -  0.02556
      -  MIRI @ James Webb Space Telescope F1500W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F1800W
      -  jwst_miri_p
      -  17.98
      -  0.03657
      -  MIRI @ James Webb Space Telescope F1800W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F2100W
      -  jwst_miri_p
      -  20.8
      -  0.03604
      -  MIRI @ James Webb Space Telescope F2100W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MIRI_p_F2550W
      -  jwst_miri_p
      -  25.36
      -  0.03036
      -  MIRI @ James Webb Space Telescope F2550W photometric filter
      -  Rieke et al., PASP 127, 584 (2015)
      -  ATMO2020, BHAC15, BEX
   *  -  MKO_H
      -  mko
      -  1.623
      -  0.1366
      -  Mauna Key Observatory NIR H-band filter
      -  Tokunaga et al., PASP, 114, 180 (2002)
      -  ATMO2020
   *  -  MKO_J
      -  mko
      -  1.246
      -  0.2362
      -  Mauna Key Observatory NIR J-band filter
      -  Tokunaga et al., PASP, 114, 180 (2002)
      -  ATMO2020
   *  -  MKO_K
      -  mko
      -  2.194
      -  0.07497
      -  Mauna Key Observatory NIR K-band filter
      -  Tokunaga et al., PASP, 114, 180 (2002)
      -  ATMO2020
   *  -  MKO_Lp
      -  mko
      -  3.757
      -  0.03485
      -  Mauna Key Observatory NIR L'-band filter
      -  Tokunaga et al., PASP, 114, 180 (2002)
      -  ATMO2020
   *  -  MKO_Mp
      -  mko
      -  4.683
      -  0.02857
      -  Mauna Key Observatory NIR M'-band filter
      -  Tokunaga et al., PASP, 114, 180 (2002)
      -  ATMO2020
   *  -  MKO_Y
      -  mko
      -  1.02
      -  0.3573
      -  Mauna Key Observatory NIR Y-band filter
      -  Hillenbrand et al., PASP, 114, 708 (2002)
      -  ATMO2020
   *  -  N
      -  bessell
      -  10.4
      -  0.07139
      -  Johnson-Cousins K-band filter
      -  Bessell & Brett, PASP 100, 1134 (1988)
      -  PARSEC, SB12
   *  -  NIRCAM_c210_F182M
      -  jwst_nircam_c210
      -  1.839
      -  0.1058
      -  NIRCAM @ James Webb Space Telescope F182M filter with 210R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c210_F187N
      -  jwst_nircam_c210
      -  1.874
      -  0.1018
      -  NIRCAM @ James Webb Space Telescope F187N filter with 210R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c210_F200W
      -  jwst_nircam_c210
      -  1.969
      -  0.0919
      -  NIRCAM @ James Webb Space Telescope F200W filter with 210R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c210_F210M
      -  jwst_nircam_c210
      -  2.092
      -  0.08166
      -  NIRCAM @ James Webb Space Telescope F210M filter with 210R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c210_F212N
      -  jwst_nircam_c210
      -  2.121
      -  0.07962
      -  NIRCAM @ James Webb Space Telescope F212N filter with 210R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F250M
      -  jwst_nircam_c335
      -  2.501
      -  0.0601
      -  NIRCAM @ James Webb Space Telescope F250M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F300M
      -  jwst_nircam_c335
      -  2.982
      -  0.04618
      -  NIRCAM @ James Webb Space Telescope F300M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F322W2
      -  jwst_nircam_c335
      -  3.075
      -  0.04429
      -  NIRCAM @ James Webb Space Telescope F322W2 filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F335M
      -  jwst_nircam_c335
      -  3.354
      -  0.03965
      -  NIRCAM @ James Webb Space Telescope F335M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F356W
      -  jwst_nircam_c335
      -  3.529
      -  0.03734
      -  NIRCAM @ James Webb Space Telescope F356W filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F360M
      -  jwst_nircam_c335
      -  3.615
      -  0.03634
      -  NIRCAM @ James Webb Space Telescope F360M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F410M
      -  jwst_nircam_c335
      -  4.072
      -  0.03216
      -  NIRCAM @ James Webb Space Telescope F410M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F430M
      -  jwst_nircam_c335
      -  4.278
      -  0.03075
      -  NIRCAM @ James Webb Space Telescope F430M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F444W
      -  jwst_nircam_c335
      -  4.344
      -  0.03034
      -  NIRCAM @ James Webb Space Telescope F444W filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F460M
      -  jwst_nircam_c335
      -  4.627
      -  0.02883
      -  NIRCAM @ James Webb Space Telescope F460M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c335_F480M
      -  jwst_nircam_c335
      -  4.812
      -  0.02801
      -  NIRCAM @ James Webb Space Telescope F480M filter with 335R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F250M
      -  jwst_nircam_c430
      -  2.501
      -  0.0601
      -  NIRCAM @ James Webb Space Telescope F250M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F300M
      -  jwst_nircam_c430
      -  2.982
      -  0.04618
      -  NIRCAM @ James Webb Space Telescope F300M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F322W2
      -  jwst_nircam_c430
      -  3.075
      -  0.04429
      -  NIRCAM @ James Webb Space Telescope F322W2 filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F335M
      -  jwst_nircam_c430
      -  3.354
      -  0.03965
      -  NIRCAM @ James Webb Space Telescope F335M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F356W
      -  jwst_nircam_c430
      -  3.529
      -  0.03734
      -  NIRCAM @ James Webb Space Telescope F356W filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F360M
      -  jwst_nircam_c430
      -  3.615
      -  0.03634
      -  NIRCAM @ James Webb Space Telescope F360M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F410M
      -  jwst_nircam_c430
      -  4.072
      -  0.03216
      -  NIRCAM @ James Webb Space Telescope F410M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F430M
      -  jwst_nircam_c430
      -  4.278
      -  0.03075
      -  NIRCAM @ James Webb Space Telescope F430M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F444W
      -  jwst_nircam_c430
      -  4.344
      -  0.03034
      -  NIRCAM @ James Webb Space Telescope F444W filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F460M
      -  jwst_nircam_c430
      -  4.627
      -  0.02883
      -  NIRCAM @ James Webb Space Telescope F460M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_c430_F480M
      -  jwst_nircam_c430
      -  4.812
      -  0.02801
      -  NIRCAM @ James Webb Space Telescope F480M filter with 430R mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F250M
      -  jwst_nircam_clwb
      -  2.501
      -  0.0601
      -  NIRCAM @ James Webb Space Telescope F250M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F277W
      -  jwst_nircam_clwb
      -  2.729
      -  0.05247
      -  NIRCAM @ James Webb Space Telescope F277W filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F300M
      -  jwst_nircam_clwb
      -  2.982
      -  0.04618
      -  NIRCAM @ James Webb Space Telescope F300M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F335M
      -  jwst_nircam_clwb
      -  3.354
      -  0.03965
      -  NIRCAM @ James Webb Space Telescope F335M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F356W
      -  jwst_nircam_clwb
      -  3.529
      -  0.03734
      -  NIRCAM @ James Webb Space Telescope F356W filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F360M
      -  jwst_nircam_clwb
      -  3.615
      -  0.03634
      -  NIRCAM @ James Webb Space Telescope F360M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F410M
      -  jwst_nircam_clwb
      -  4.072
      -  0.03216
      -  NIRCAM @ James Webb Space Telescope F410M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F430M
      -  jwst_nircam_clwb
      -  4.278
      -  0.03075
      -  NIRCAM @ James Webb Space Telescope F430M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F444W
      -  jwst_nircam_clwb
      -  4.344
      -  0.03034
      -  NIRCAM @ James Webb Space Telescope F444W filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F460M
      -  jwst_nircam_clwb
      -  4.627
      -  0.02883
      -  NIRCAM @ James Webb Space Telescope F460M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cLWB_F480M
      -  jwst_nircam_clwb
      -  4.812
      -  0.02801
      -  NIRCAM @ James Webb Space Telescope F480M filter with LWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cSWB_F182M
      -  jwst_nircam_cswb
      -  1.839
      -  0.1058
      -  NIRCAM @ James Webb Space Telescope F182M filter with SWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cSWB_F187N
      -  jwst_nircam_cswb
      -  1.874
      -  0.1018
      -  NIRCAM @ James Webb Space Telescope F187N filter with SWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cSWB_F200W
      -  jwst_nircam_cswb
      -  1.969
      -  0.0919
      -  NIRCAM @ James Webb Space Telescope F200W filter with SWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cSWB_F210M
      -  jwst_nircam_cswb
      -  2.092
      -  0.08166
      -  NIRCAM @ James Webb Space Telescope F210M filter with SWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_cSWB_F212N
      -  jwst_nircam_cswb
      -  2.121
      -  0.07962
      -  NIRCAM @ James Webb Space Telescope F212N filter with SWB mask
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020
   *  -  NIRCAM_p_F070Wa
      -  jwst_nircam_pa
      -  0.704
      -  0.6919
      -  NIRCAM @ James Webb Space Telescope F070W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F070Wab
      -  jwst_nircam_pab
      -  0.704
      -  0.6919
      -  NIRCAM @ James Webb Space Telescope F070W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F070Wb
      -  jwst_nircam_pb
      -  0.704
      -  0.6919
      -  NIRCAM @ James Webb Space Telescope F070W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F090Wa
      -  jwst_nircam_pa
      -  0.9004
      -  0.4523
      -  NIRCAM @ James Webb Space Telescope F090W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F090Wab
      -  jwst_nircam_pab
      -  0.9004
      -  0.4523
      -  NIRCAM @ James Webb Space Telescope F090W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F090Wb
      -  jwst_nircam_pb
      -  0.9004
      -  0.4523
      -  NIRCAM @ James Webb Space Telescope F090W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F115Wa
      -  jwst_nircam_pa
      -  1.15
      -  0.2785
      -  NIRCAM @ James Webb Space Telescope F115W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15, BEX
   *  -  NIRCAM_p_F115Wab
      -  jwst_nircam_pab
      -  1.15
      -  0.2785
      -  NIRCAM @ James Webb Space Telescope F115W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F115Wb
      -  jwst_nircam_pb
      -  1.15
      -  0.2785
      -  NIRCAM @ James Webb Space Telescope F115W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F140Ma
      -  jwst_nircam_pa
      -  1.404
      -  0.1849
      -  NIRCAM @ James Webb Space Telescope F140M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F140Mab
      -  jwst_nircam_pab
      -  1.404
      -  0.1849
      -  NIRCAM @ James Webb Space Telescope F140M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F140Mb
      -  jwst_nircam_pb
      -  1.404
      -  0.1849
      -  NIRCAM @ James Webb Space Telescope F140M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F150Wa
      -  jwst_nircam_pa
      -  1.494
      -  0.1618
      -  NIRCAM @ James Webb Space Telescope F150W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15, BEX
   *  -  NIRCAM_p_F150Wab
      -  jwst_nircam_pab
      -  1.494
      -  0.1618
      -  NIRCAM @ James Webb Space Telescope F150W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F150Wb
      -  jwst_nircam_pb
      -  1.494
      -  0.1618
      -  NIRCAM @ James Webb Space Telescope F150W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F150W2a
      -  jwst_nircam_pa
      -  1.542
      -  0.1519
      -  NIRCAM @ James Webb Space Telescope F150W2 filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F150W2ab
      -  jwst_nircam_pab
      -  1.542
      -  0.1519
      -  NIRCAM @ James Webb Space Telescope F150W2 filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F150W2b
      -  jwst_nircam_pb
      -  1.542
      -  0.1519
      -  NIRCAM @ James Webb Space Telescope F150W2 filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F162Ma
      -  jwst_nircam_pa
      -  1.625
      -  0.1361
      -  NIRCAM @ James Webb Space Telescope F162M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F162Mab
      -  jwst_nircam_pab
      -  1.625
      -  0.1361
      -  NIRCAM @ James Webb Space Telescope F162M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F162Mb
      -  jwst_nircam_pb
      -  1.625
      -  0.1361
      -  NIRCAM @ James Webb Space Telescope F162M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F164Na
      -  jwst_nircam_pa
      -  1.645
      -  0.1329
      -  NIRCAM @ James Webb Space Telescope F164N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F164Nab
      -  jwst_nircam_pab
      -  1.645
      -  0.1329
      -  NIRCAM @ James Webb Space Telescope F164N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F164Nb
      -  jwst_nircam_pb
      -  1.645
      -  0.1329
      -  NIRCAM @ James Webb Space Telescope F164N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F182Ma
      -  jwst_nircam_pa
      -  1.839
      -  0.1058
      -  NIRCAM @ James Webb Space Telescope F182M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F182Mab
      -  jwst_nircam_pab
      -  1.839
      -  0.1058
      -  NIRCAM @ James Webb Space Telescope F182M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F182Mb
      -  jwst_nircam_pb
      -  1.839
      -  0.1058
      -  NIRCAM @ James Webb Space Telescope F182M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F187Na
      -  jwst_nircam_pa
      -  1.874
      -  0.1018
      -  NIRCAM @ James Webb Space Telescope F187N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F187Nab
      -  jwst_nircam_pab
      -  1.874
      -  0.1018
      -  NIRCAM @ James Webb Space Telescope F187N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F187Nb
      -  jwst_nircam_pb
      -  1.874
      -  0.1018
      -  NIRCAM @ James Webb Space Telescope F187N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F200Wa
      -  jwst_nircam_pa
      -  1.969
      -  0.0919
      -  NIRCAM @ James Webb Space Telescope F200W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15, BEX
   *  -  NIRCAM_p_F200Wab
      -  jwst_nircam_pab
      -  1.969
      -  0.0919
      -  NIRCAM @ James Webb Space Telescope F200W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F200Wb
      -  jwst_nircam_pb
      -  1.969
      -  0.0919
      -  NIRCAM @ James Webb Space Telescope F200W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F210Ma
      -  jwst_nircam_pa
      -  2.092
      -  0.08166
      -  NIRCAM @ James Webb Space Telescope F210M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F210Mab
      -  jwst_nircam_pab
      -  2.092
      -  0.08166
      -  NIRCAM @ James Webb Space Telescope F210M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F210Mb
      -  jwst_nircam_pb
      -  2.092
      -  0.08166
      -  NIRCAM @ James Webb Space Telescope F210M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F212Na
      -  jwst_nircam_pa
      -  2.121
      -  0.07962
      -  NIRCAM @ James Webb Space Telescope F212N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F212Nab
      -  jwst_nircam_pab
      -  2.121
      -  0.07962
      -  NIRCAM @ James Webb Space Telescope F212N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F212Nb
      -  jwst_nircam_pb
      -  2.121
      -  0.07962
      -  NIRCAM @ James Webb Space Telescope F212N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F250Ma
      -  jwst_nircam_pa
      -  2.501
      -  0.0601
      -  NIRCAM @ James Webb Space Telescope F250M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F250Mab
      -  jwst_nircam_pab
      -  2.501
      -  0.0601
      -  NIRCAM @ James Webb Space Telescope F250M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F250Mb
      -  jwst_nircam_pb
      -  2.501
      -  0.0601
      -  NIRCAM @ James Webb Space Telescope F250M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F277Wa
      -  jwst_nircam_pa
      -  2.729
      -  0.05247
      -  NIRCAM @ James Webb Space Telescope F277W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15, BEX
   *  -  NIRCAM_p_F277Wab
      -  jwst_nircam_pab
      -  2.729
      -  0.05247
      -  NIRCAM @ James Webb Space Telescope F277W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F277Wb
      -  jwst_nircam_pb
      -  2.729
      -  0.05247
      -  NIRCAM @ James Webb Space Telescope F277W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F300Ma
      -  jwst_nircam_pa
      -  2.982
      -  0.04618
      -  NIRCAM @ James Webb Space Telescope F300M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F300Mab
      -  jwst_nircam_pab
      -  2.982
      -  0.04618
      -  NIRCAM @ James Webb Space Telescope F300M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F300Mb
      -  jwst_nircam_pb
      -  2.982
      -  0.04618
      -  NIRCAM @ James Webb Space Telescope F300M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F322W2a
      -  jwst_nircam_pa
      -  3.075
      -  0.04429
      -  NIRCAM @ James Webb Space Telescope F322W2 filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F322W2ab
      -  jwst_nircam_pab
      -  3.075
      -  0.04429
      -  NIRCAM @ James Webb Space Telescope F322W2 filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F322W2b
      -  jwst_nircam_pb
      -  3.075
      -  0.04429
      -  NIRCAM @ James Webb Space Telescope F322W2 filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F323Na
      -  jwst_nircam_pa
      -  3.237
      -  0.04143
      -  NIRCAM @ James Webb Space Telescope F323N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F323Nab
      -  jwst_nircam_pab
      -  3.237
      -  0.04143
      -  NIRCAM @ James Webb Space Telescope F323N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F323Nb
      -  jwst_nircam_pb
      -  3.237
      -  0.04143
      -  NIRCAM @ James Webb Space Telescope F323N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F335Ma
      -  jwst_nircam_pa
      -  3.354
      -  0.03965
      -  NIRCAM @ James Webb Space Telescope F335M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F335Mab
      -  jwst_nircam_pab
      -  3.354
      -  0.03965
      -  NIRCAM @ James Webb Space Telescope F335M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F335Mb
      -  jwst_nircam_pb
      -  3.354
      -  0.03965
      -  NIRCAM @ James Webb Space Telescope F335M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F356Wa
      -  jwst_nircam_pa
      -  3.529
      -  0.03734
      -  NIRCAM @ James Webb Space Telescope F356W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15, BEX
   *  -  NIRCAM_p_F356Wab
      -  jwst_nircam_pab
      -  3.529
      -  0.03734
      -  NIRCAM @ James Webb Space Telescope F356W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F356Wb
      -  jwst_nircam_pb
      -  3.529
      -  0.03734
      -  NIRCAM @ James Webb Space Telescope F356W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F360Ma
      -  jwst_nircam_pa
      -  3.615
      -  0.03634
      -  NIRCAM @ James Webb Space Telescope F360M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F360Mab
      -  jwst_nircam_pab
      -  3.615
      -  0.03634
      -  NIRCAM @ James Webb Space Telescope F360M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F360Mb
      -  jwst_nircam_pb
      -  3.615
      -  0.03634
      -  NIRCAM @ James Webb Space Telescope F360M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F405Na
      -  jwst_nircam_pa
      -  4.052
      -  0.03231
      -  NIRCAM @ James Webb Space Telescope F405N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F405Nab
      -  jwst_nircam_pab
      -  4.052
      -  0.03231
      -  NIRCAM @ James Webb Space Telescope F405N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F405Nb
      -  jwst_nircam_pb
      -  4.052
      -  0.03231
      -  NIRCAM @ James Webb Space Telescope F405N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F410Ma
      -  jwst_nircam_pa
      -  4.072
      -  0.03216
      -  NIRCAM @ James Webb Space Telescope F410M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F410Mab
      -  jwst_nircam_pab
      -  4.072
      -  0.03216
      -  NIRCAM @ James Webb Space Telescope F410M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F410Mb
      -  jwst_nircam_pb
      -  4.072
      -  0.03216
      -  NIRCAM @ James Webb Space Telescope F410M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F430Ma
      -  jwst_nircam_pa
      -  4.278
      -  0.03075
      -  NIRCAM @ James Webb Space Telescope F430M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F430Mab
      -  jwst_nircam_pab
      -  4.278
      -  0.03075
      -  NIRCAM @ James Webb Space Telescope F430M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F430Mb
      -  jwst_nircam_pb
      -  4.278
      -  0.03075
      -  NIRCAM @ James Webb Space Telescope F430M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F444Wa
      -  jwst_nircam_pa
      -  4.344
      -  0.03034
      -  NIRCAM @ James Webb Space Telescope F444W filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15, BEX
   *  -  NIRCAM_p_F444Wab
      -  jwst_nircam_pab
      -  4.344
      -  0.03034
      -  NIRCAM @ James Webb Space Telescope F444W filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F444Wb
      -  jwst_nircam_pb
      -  4.344
      -  0.03034
      -  NIRCAM @ James Webb Space Telescope F444W filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F460Ma
      -  jwst_nircam_pa
      -  4.627
      -  0.02883
      -  NIRCAM @ James Webb Space Telescope F460M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F460Mab
      -  jwst_nircam_pab
      -  4.627
      -  0.02883
      -  NIRCAM @ James Webb Space Telescope F460M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F460Mb
      -  jwst_nircam_pb
      -  4.627
      -  0.02883
      -  NIRCAM @ James Webb Space Telescope F460M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F466Na
      -  jwst_nircam_pa
      -  4.654
      -  0.0287
      -  NIRCAM @ James Webb Space Telescope F466N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F466Nab
      -  jwst_nircam_pab
      -  4.654
      -  0.0287
      -  NIRCAM @ James Webb Space Telescope F466N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F466Nb
      -  jwst_nircam_pb
      -  4.654
      -  0.0287
      -  NIRCAM @ James Webb Space Telescope F466N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F470Na
      -  jwst_nircam_pa
      -  4.708
      -  0.02846
      -  NIRCAM @ James Webb Space Telescope F470N filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F470Nab
      -  jwst_nircam_pab
      -  4.708
      -  0.02846
      -  NIRCAM @ James Webb Space Telescope F470N filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F470Nb
      -  jwst_nircam_pb
      -  4.708
      -  0.02846
      -  NIRCAM @ James Webb Space Telescope F470N filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F480Ma
      -  jwst_nircam_pa
      -  4.812
      -  0.02801
      -  NIRCAM @ James Webb Space Telescope F480M filter, module A
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F480Mab
      -  jwst_nircam_pab
      -  4.812
      -  0.02801
      -  NIRCAM @ James Webb Space Telescope F480M filter, module AB
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRCAM_p_F480Mb
      -  jwst_nircam_pb
      -  4.812
      -  0.02801
      -  NIRCAM @ James Webb Space Telescope F480M filter, module B
      -  Horner & Rieke, Proc. SPIE 5487, 628 (2004)
      -  ATMO2020, BHAC15
   *  -  NIRISS_c_F277W
      -  jwst_niriss_c
      -  2.729
      -  0.05247
      -  NIRISS @ James Webb Space Telescope F277W coronagraphic filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_c_F380M
      -  jwst_niriss_c
      -  3.828
      -  0.03418
      -  NIRISS @ James Webb Space Telescope F380M coronagraphic filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_c_F430M
      -  jwst_niriss_c
      -  4.278
      -  0.03075
      -  NIRISS @ James Webb Space Telescope F430M coronagraphic filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_c_F480M
      -  jwst_niriss_c
      -  4.812
      -  0.02801
      -  NIRISS @ James Webb Space Telescope F480M coronagraphic filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F090W
      -  jwst_niriss_p
      -  0.9004
      -  0.4523
      -  NIRISS @ James Webb Space Telescope F090W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F115W
      -  jwst_niriss_p
      -  1.15
      -  0.2785
      -  NIRISS @ James Webb Space Telescope F115W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F140M
      -  jwst_niriss_p
      -  1.404
      -  0.1849
      -  NIRISS @ James Webb Space Telescope F140M filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F150W
      -  jwst_niriss_p
      -  1.494
      -  0.1618
      -  NIRISS @ James Webb Space Telescope F150W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F158M
      -  jwst_niriss_p
      -  1.587
      -  0.1431
      -  NIRISS @ James Webb Space Telescope F158M filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F200W
      -  jwst_niriss_p
      -  1.969
      -  0.0919
      -  NIRISS @ James Webb Space Telescope F200W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F277W
      -  jwst_niriss_p
      -  2.729
      -  0.05247
      -  NIRISS @ James Webb Space Telescope F277W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F356W
      -  jwst_niriss_p
      -  3.529
      -  0.03734
      -  NIRISS @ James Webb Space Telescope F356W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F380M
      -  jwst_niriss_p
      -  3.828
      -  0.03418
      -  NIRISS @ James Webb Space Telescope F380M filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F430M
      -  jwst_niriss_p
      -  4.278
      -  0.03075
      -  NIRISS @ James Webb Space Telescope F430M filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F444W
      -  jwst_niriss_p
      -  4.344
      -  0.03034
      -  NIRISS @ James Webb Space Telescope F444W filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  NIRISS_p_F480M
      -  jwst_niriss_p
      -  4.812
      -  0.02801
      -  NIRISS @ James Webb Space Telescope F480M filter
      -  Doyon et al., SPIE 844, 84422R (2012) 
      -  ATMO2020
   *  -  R
      -  bessell
      -  0.6535
      -  0.7759
      -  Johnson-Cousins R-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  Geneva, MIST, PARSEC, BT-Settl, SPOTS, STAREVOL, PM13, BHAC15, BEX
   *  -  r
      -  panstarrs
      -  0.6211
      -  0.843
      -  Panstarrs DR1 r-band filter
      -  Magnier et al., ApJS, 251, 6 (2020)
      -  BHAC15, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  SDSS_g
      -  sdss
      -  0.4751
      -  1.203
      -  Sloan Digital Sky Survey g-band filter
      -  Fukugita et al., AJ, 111, 1748 (1996)
      -  PARSEC, MIST, Ames-Cond, Ames-Dusty, BT-Settl, NextGen
   *  -  SDSS_i
      -  sdss
      -  0.7519
      -  0.6183
      -  Sloan Digital Sky Survey i-band filter
      -  Fukugita et al., AJ, 111, 1748 (1996)
      -  PARSEC, MIST, Ames-Cond, Ames-Dusty, BT-Settl, NextGen
   *  -  SDSS_r
      -  sdss
      -  0.6204
      -  0.84
      -  Sloan Digital Sky Survey r-band filter
      -  Fukugita et al., AJ, 111, 1748 (1996)
      -  PARSEC, MIST, Ames-Cond, Ames-Dusty, BT-Settl, NextGen
   *  -  SDSS_u
      -  sdss
      -  0.3572
      -  1.584
      -  Sloan Digital Sky Survey u-band filter
      -  Fukugita et al., AJ, 111, 1748 (1996)
      -  PARSEC, MIST, Ames-Cond, Ames-Dusty, BT-Settl, NextGen
   *  -  SDSS_z
      -  sdss
      -  0.8992
      -  0.452
      -  Sloan Digital Sky Survey z-band filter
      -  Fukugita et al., AJ, 111, 1748 (1996)
      -  PARSEC, MIST, Ames-Cond, Ames-Dusty, BT-Settl, NextGen
   *  -  SM_g
      -  skymapper
      -  0.5075
      -  1.11
      -  SkyMapper g-band filter
      -  Keller et al., PASA, 24, 1 (2007)
      -  PARSEC
   *  -  SM_i
      -  skymapper
      -  0.7768
      -  0.5851
      -  SkyMapper i-band filter
      -  Keller et al., PASA, 24, 1 (2007)
      -  PARSEC
   *  -  SM_r
      -  skymapper
      -  0.6138
      -  0.8535
      -  SkyMapper r-band filter
      -  Keller et al., PASA, 24, 1 (2007)
      -  PARSEC
   *  -  SM_u
      -  skymapper
      -  0.3493
      -  1.611
      -  SkyMapper u-band filter
      -  Keller et al., PASA, 24, 1 (2007)
      -  PARSEC
   *  -  SM_v
      -  skymapper
      -  0.3836
      -  1.494
      -  SkyMapper v-band filter
      -  Keller et al., PASA, 24, 1 (2007)
      -  PARSEC
   *  -  SM_z
      -  skymapper
      -  0.9146
      -  0.4382
      -  SkyMapper z-band filter
      -  Keller et al., PASA, 24, 1 (2007)
      -  PARSEC
   *  -  SPH_H
      -  sphere
      -  1.625
      -  0.1362
      -  SPHERE IRDIS @ VLT H-band filter
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_H2
      -  sphere
      -  1.593
      -  0.142
      -  SPHERE IRDIS @ VLT H2 filter (dual band: H2-H3)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_H3
      -  sphere
      -  1.667
      -  0.1292
      -  SPHERE IRDIS @ VLT H3 filter (dual band: H2-H3)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_H4
      -  sphere
      -  1.733
      -  0.1193
      -  SPHERE IRDIS @ VLT H4 filter (dual band: H3-H4)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  SPH_J
      -  sphere
      -  1.245
      -  0.2365
      -  SPHERE IRDIS @ VLT J-band filter
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_J2
      -  sphere
      -  1.19
      -  0.2597
      -  SPHERE IRDIS @ VLT J2 filter (dual band: J2-J3)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_J3
      -  sphere
      -  1.273
      -  0.2258
      -  SPHERE IRDIS @ VLT J3 filter (dual band: J2-J3)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_K
      -  sphere
      -  2.182
      -  0.07569
      -  SPHERE IRDIS @ VLT Ks-band filter
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_K1
      -  sphere
      -  2.11
      -  0.08037
      -  SPHERE IRDIS @ VLT K1 filter (dual band: K1-K2)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_K2
      -  sphere
      -  2.251
      -  0.07167
      -  SPHERE IRDIS @ VLT K2 filter (dual band: K1-K2)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_NDH
      -  sphere
      -  1.593
      -  0.142
      -  SPHERE IRDIS @ VLT ND_H filter (dual band: ND_H-H23)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  SPH_Y
      -  sphere
      -  1.043
      -  0.3411
      -  SPHERE IRDIS @ VLT Y-band filter
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, BEX
   *  -  SPH_Y2
      -  sphere
      -  1.022
      -  0.3558
      -  SPHERE IRDIS @ VLT Y2 filter (dual band: Y2-Y3)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  SPH_Y3
      -  sphere
      -  1.076
      -  0.3198
      -  SPHERE IRDIS @ VLT Y3 filter (dual band: Y2-Y3)
      -  Krol et al., Proc. SPIE 8168 (2011)
      -  BHAC15, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  T_B
      -  tycho
      -  0.428
      -  1.349
      -  Tycho B-band filter
      -  HÃ¸g et al., A&A 355, L27 (2000)
      -  MIST
   *  -  T_V
      -  tycho
      -  0.534
      -  1.039
      -  Tycho V-band filter
      -  HÃ¸g et al., A&A 355, L27 (2000)
      -  MIST
   *  -  U
      -  bessell
      -  0.3656
      -  1.555
      -  Johnson-Cousins U-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  Geneva, MIST, BT-Settl, STAREVOL, PM13
   *  -  Ux
      -  bessell
      -  0.3656
      -  1.555
      -  Johnson-Cousins U_X-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  PARSEC
   *  -  UKIDSS_h
      -  ukirt
      -  1.631
      -  0.1352
      -  UKIRT Infrared Deep Sky Survey h-band filter
      -  Lawrence et al., MNRAS 379, 1599 (2007)
      -  BHAC15
   *  -  UKIDSS_j
      -  ukirt
      -  1.248
      -  0.2352
      -  UKIRT Infrared Deep Sky Survey j-band filter
      -  Lawrence et al., MNRAS 379, 1599 (2007)
      -  BHAC15
   *  -  UKIDSS_k
      -  ukirt
      -  2.201
      -  0.07454
      -  UKIRT Infrared Deep Sky Survey k-band filter
      -  Lawrence et al., MNRAS 379, 1599 (2007)
      -  BHAC15
   *  -  UKIDSS_y
      -  ukirt
      -  1.03
      -  0.3498
      -  UKIRT Infrared Deep Sky Survey h-band filter
      -  Lawrence et al., MNRAS 379, 1599 (2007)
      -  BHAC15
   *  -  UKIDSS_z
      -  ukirt
      -  0.8817
      -  0.4684
      -  UKIRT Infrared Deep Sky Survey h-band filter
      -  Lawrence et al., MNRAS 379, 1599 (2007)
      -  BHAC15
   *  -  V
      -  bessell
      -  0.5525
      -  1.0
      -  Johnson-Cousins V-band filter
      -  Bessell, PASP 102, 1181 (1990)
      -  Geneva, MIST, PARSEC, BT-Settl, SPOTS, STAREVOL, PM13, BHAC15
   *  -  W1
      -  wise
      -  3.317
      -  0.04018
      -  Wide-field Infrared Survey Explorer (WISE) W1-band filter
      -  Wright et al., AJ 140, 1868 (2010)
      -  ATMO2020, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, Sonora Bobcat, SPOTS, PM13, BEX
   *  -  W2
      -  wise
      -  4.55
      -  0.02921
      -  Wide-field Infrared Survey Explorer (WISE) W2-band filter
      -  Wright et al., AJ 140, 1868 (2010)
      -  ATMO2020, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, Sonora Bobcat, PM13, BEX
   *  -  W3
      -  wise
      -  11.73
      -  0.04046
      -  Wide-field Infrared Survey Explorer (WISE) W1-band filter
      -  Wright et al., AJ 140, 1868 (2010)
      -  ATMO2020, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, Sonora Bobcat, PM13, BEX
   *  -  W4
      -  wise
      -  22.09
      -  0.03432
      -  Wide-field Infrared Survey Explorer (WISE) W1-band filter
      -  Wright et al., AJ 140, 1868 (2010)
      -  ATMO2020, MIST, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen, Sonora Bobcat, PM13, BEX
   *  -  y
      -  panstarrs
      -  0.9707
      -  0.395
      -  Panstarrs DR1 y-band filter
      -  Magnier et al., ApJS, 251, 6 (2020)
      -  BHAC15, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
   *  -  z
      -  panstarrs
      -  0.8671
      -  0.487
      -  Panstarrs DR1 z-band filter
      -  Magnier et al., ApJS, 251, 6 (2020)
      -  BHAC15, PARSEC, AMES-Cond, AMES-Dusty, BT-Settl, NextGen
