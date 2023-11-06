Available classes
=====

In its current form, MADYS comprises five classes which serve different purposes:

* ``SampleObject``: it allows one to create a catalog of objects will full astrometric, kinematic and photometric information. See `here <https://madys.readthedocs.io/en/latest/instance_creation.html>`_;
* ``ModelHandler``: handles data and metadata of local isochrone files, allowing one to inspect the locally available models and to download new ones. See `here <https://madys.readthedocs.io/en/latest/available_models.html>`_;
* ``IsochroneGrid``: builds and handles the isochronal grid used for parameter determination. Usually not be called directly;
* ``FitParams``: stores the result of an isochronal analysis performed upon a SampleObject instance. See `here <https://madys.readthedocs.io/en/latest/computation_run.html#the-fitparams-class>`_;
* ``CurveObject``: allows the conversion of a contrast curve (i.e., detection limits) from a direct imaging survey into a mass limit curve. See HERE.
