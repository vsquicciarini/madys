Working with DPMs
=====

Detection probability maps generated via :py:func:`DetectionMap.compute_completeness_map` can be plotted using that same function, as in the examples shown here. The plotting is based on a dedicated method, :py:func:`DetectionMap.plot_completeness_map`, that can be directly called by the user. In fact, we strongly advice to use the function in its stand-alone mode, as it allows for a much larger degree of customization of the plotting options.
The function takes as input the output of :py:func:`DetectionMap.compute_completeness_map`. Its possible parameters are:

* dtype: string, optional. Type of map: 'mass' or 'Teff'. Default: 'mass'.
* band: string, optional. Filter of the DPM. Mandatory iff map_dict contains more than one key. Default: None.
* to_file: bool or string, optional. Use a bool to select whether to export the result to a file or not. Alternatively, one can provide a string with the full path to the output file. Default: True.
* output_path: string, optional. Only used if to_file=True. Full output path for the output file. If not set, it uses the same path where self.file is located. Default: None (=not set).
* assume_resolved: bool, optional. Only used if output_path is set. Whether to consider 'output_path' as a full path (True), or the last directory of a larger path. If False, the output path will be os.file.basename(self.file)+output_path. Default: False.
* fig_xlim: NoneType or list, optional. A two-element list with the range for the x axis. Default: None (=determined automatically).
* fig_ylim: NoneType orlist, optional. A two-element list with the range for the y axis. Default: None (=determined automatically).
* show_model_limits: bool, optional. If True, it shows on the plot the areas where the model is not defined.
* contourf_options: NoneType or dict, optional. Optional parameters to be passed to :py:func:`plt.contourf` (see documentation of that function for details). In addition to this, two keywords named ``colorbar`` and ``serif_font`` can be used to specify if the colorbar is to be shown (default: True) and if the figure fonts and fontsizes should be optimized (default: True).
* contour_options: NoneType or dict, optional. Optional parameters to be passed to :py:func:`plt.contour` (see documentation of that function for details).
* ax: NoneType or AxesSubplot, optional. Axis object where the map will be drawn upon. Default: None (=a new figure is created).

and its outputs are:

* ax: Axis object where the map was plotted. Only returned if ax was not None.

It is possible, for instance, to draw multiple DPMs with different colorbars in the same figure, or to include a DPM in a pre-existing plot.
