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
   :widths: 7 7 7 60
   :stub-columns: 1

   *  -  Day
      -  Min Temp
      -  Max Temp
      -  Summary
   *  -  Monday
      -  11C
      -  22C
      -  A clear day with lots of sunshine.
         However, the strong breeze will bring
         down the temperatures.
   *  -  Tuesday
      -  9C
      -  10C
      -  Cloudy with rain, across many northern regions. Clear spells
         across most of Scotland and Northern Ireland,
         but rain reaching the far northwest.
