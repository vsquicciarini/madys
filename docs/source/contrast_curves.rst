Contrast curves
=====

Creation of a CurveObject instance
----------------

In addition to the characterization of stars and substellar objects, MADYS provides another particularly useful feature for the high-contrast imaging (HCI) community: namely, the capability to convert detection limit curves of a HCI observation into mass limit curves.

The computation is mediated by the ``CurveObject`` class. Let us analyze the syntax needed to initialize an instance:


.. code-block:: python

   file = "examples/SPHERE_contrast_curve.fits"
   
   params = {'parallax': 5.8219, 'parallax_error':0.31,
             'ebv': 0.04, 'ebv_error': 0.01,
             'app_mag': 6.182, 'app_mag_error': 0.233,
             'band': 'SPH_K1', 'age': 16.,
             'age_error': 7.
            }

   curve = CurveObject(file, 'contrast_map', params)
   
The basic inputs needed to create an instance are therefore:

* file: string, required. Full path to the .fits file containing the input curve;
* file_type: string, required. It can be either:

   * 'contrast_separation', if a 1D curve(separation) with shape (n_points, 2) is provided; the first column is assumed to
    correspond to contrasts, the second one to separations;
   * 'contrast_map', if a 2D curve(x, y) is provided. A 3D map (lambda, x, y) is accepted too.

* stellar_parameters: dict, required. A dictionary containing information for the star under consideration. The following keywords must be present:

   * 'parallax': float. Stellar parallax [mas];
   * 'parallax_error': float. Uncertainty on stellar parallax [mas];
   * 'ebv': float. E(B-V) reddening for the star [mag];
   * 'ebv_error': float. Uncertainty on E(B-V) [mag];
   * 'app_mag': float. Apparent magnitude of the star in the 'band' band [mag];
   * 'app_mag_error': float. Uncertainty on 'app_mag' [mag];
   * 'band': string. Filter which the map refers to. It should be a valid filter name for MADYS;
   * 'age': float. Stellar age [Myr];
   * 'age_error': float. Uncertainty on stellar age [Myr].

* data_unit: string, optional. Choose 'magnitude' if the map is expressed in magnitudes, 'flux' if in flux contrast. Default: 'flux'.
* rescale_flux = float, optional. Renormalization constant the flux is to be multiplied by. Default: 1.

Once created, the instance possesses the following attributes:

* file: string. Corresponding to input 'file'.
* file_type: string. Corresponding to input 'file_type'.
* stellar_parameters: dict. Corresponding to input 'stellar_parameters'.
* data_unit: string. Corresponding to input 'data_unit'.
* rescale_flux: float. Corresponding to input 'rescale_flux'.
* contrasts: numpy array. Renormalized contrast curve (flux ratio).
* contrasts_mag: numpy array. Renormalized contrast curve (magnitude contrast).
* abs_phot: numpy array. Absolute magnitudes in the required filters.
* abs_phot_err: numpy array. Uncertainties on absolute magnitudes in the required filters.
* abs_phot: numpy array. Apparent magnitudes in the required filters.
* abs_phot_err: numpy array. Uncertainties on apparent magnitudes in the required filters.
* mag_limits: numpy array. Limit absolute magnitudes corresponding to input curve.
* mag_limits_err: numpy array. Uncertainties on limit absolute magnitudes.
* mag_limits_app: numpy array. Limit apparent magnitudes corresponding to input curve.
* mag_limits_app_err: numpy array. Uncertainties on limit apparent magnitudes.
* separations: numpy array. Input separations, if 'file_type'='contrast_separation'; zero-filled array otherwise.
* band: string. Input stellar_parameters['band'].
* platescale: float. Platescale of the instrument FOV [mas/px].


Creation of mass curves
----------------

Starting from the object create above, it's easy to compute the corresponding mass curve through the function :py:func:`CurveObject.compute_mass_limits`:

.. code-block:: python

   results = curve.compute_mass_limits('atmo2020-ceq')


If ``file_type``='contrast_map', the program will additionally collapse the map along the azimuthal direction, yielding a (N-1)-dimensional output in addition to the N-dimensional mass curve.
