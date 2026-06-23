from __future__ import annotations

import numpy as np
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from iminuit import Minuit

def plot_NLL_scans(parameter_label: str,
                   list_scan_points: list[list[float]],
                   list_nll_values: list[list[float]],
                   list_labels: list[str],
                   list_linestyles: list[str],
                   list_colors: list[str],
                   ax: plt.Axes | None = None):
    """
    Plot one or more NLL profile scan curves on a single axes.

    Draws each scan as a line on a shared axes, adds horizontal
    reference lines at :math:`\\Delta NLL = 1, 4, 9` corresponding to
    the :math:`1\\sigma`, :math:`2\\sigma`, and :math:`3\\sigma`
    confidence intervals, and annotates them accordingly.

    Parameters
    ----------
    parameter_label : str
        LaTeX-formatted label for the x-axis, e.g. ``r"$\\mu$"``.
        If an empty string is passed the raw ``parameter_name``
        variable is used as a fallback (note: ``parameter_name`` must
        be defined in the calling scope in that case).

    list_scan_points : list of list of float
        Scan point coordinates for each curve. Each inner list must
        have the same length as the corresponding entry in
        ``list_nll_values``. Typically the first return value of
        :meth:`inference.perform_profile_scan`.

    list_nll_values : list of list of float
        :math:`\\Delta NLL` values for each curve, evaluated at the
        corresponding scan points. Values should already be
        minimum-subtracted (i.e. the minimum of each curve sits at 0).

    list_labels : list of str
        Legend labels for each curve, e.g.
        ``["Stat + Syst", "Stat Only"]``.

    list_linestyles : list of str
        Matplotlib linestyle strings for each curve,
        e.g. ``["solid", "dashed"]``.

    list_colors : list of str
        Matplotlib colour strings for each curve,
        e.g. ``["black", "red"]``.

    ax : matplotlib.axes.Axes or None, optional
        Axes object to draw on. If ``None`` (default), a new figure
        and axes are created internally. Pass an existing axes to
        embed the plot in a larger figure layout.

    Notes
    -----
    * All lists (``list_scan_points``, ``list_nll_values``,
      ``list_labels``, ``list_linestyles``, ``list_colors``) must have
      the same length; no length validation is performed.
    * The y-axis lower limit is fixed at ``0.0``; there is no upper
      limit set, so matplotlib will auto-scale to the data.
    * Reference lines at :math:`\\Delta NLL = 1, 4, 9` assume that
      the profile likelihood ratio test statistic
      :math:`t_\\mu = -2\\Delta\\ln L` is used, so confidence
      intervals are valid under Wilks' theorem.
    * If ``ax`` is ``None`` the created figure is not returned; call
      ``plt.savefig`` or ``plt.show`` after this function if needed.

    Examples
    --------
    .. code-block:: python

        scan_pts, nll_vals = fitter.perform_profile_scan("mu", (0.0, 3.0))
        plot_NLL_scans(
            parameter_label=r"$\\mu$",
            list_scan_points=[scan_pts],
            list_nll_values=[nll_vals],
            list_labels=["Stat + Syst"],
            list_linestyles=["solid"],
            list_colors=["black"]
        )
        plt.show()

    See Also
    --------
    inference.perform_profile_scan : Produces the scan arrays consumed
        by this function.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for count in range(len(list_labels)):

        ax.plot(
            list_scan_points[count],
            list_nll_values[count],
            linestyle=list_linestyles[count],
            label=list_labels[count],
            color=list_colors[count])
        ax.legend()
        
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(parameter_label or parameter_name)
    ax.set_ylabel(r"$t_\mu$")
    ax.axhline(y=1.0, color='gray', linestyle='dotted', alpha=0.5)
    ax.text(1.0, 1.02, r"$1\sigma$          ", transform=ax.get_yaxis_transform(), ha='right', va='bottom', color='gray', fontsize=9)

    ax.axhline(y=4.0, color='gray', linestyle='dotted', alpha=0.5)
    ax.text(1.0, 4.02, r"$2\sigma$          ", transform=ax.get_yaxis_transform(), ha='right', va='bottom', color='gray', fontsize=9)

    ax.axhline(y=9.0, color='gray', linestyle='dotted', alpha=0.5)
    ax.text(1.0, 9.02, r"$3\sigma$          ", transform=ax.get_yaxis_transform(), ha='right', va='bottom', color='gray', fontsize=9)


    
class inference:
    def __init__(self,
             model_nll,
             initial_values: list[float],
             list_parameters: list[str],
             num_unconstrained_params: int,
             model_grad=None):
        """
        Initialise the inference engine around a callable NLL function.

        Parameters
        ----------
        model_nll : callable
            A scalar-valued function representing the negative
            log-likelihood to minimise. The signature must be compatible
            with iminuit's expectation: either a single array argument
            (used here via ``Minuit(f, values, name=names)``) or explicit
            keyword arguments. The function must return a scalar NLL value.
            JAX-compiled functions are supported and recommended for
            performance.

        initial_values : list of float or jnp.ndarray, shape (n_params,)
            Starting values for all parameters passed to MIGRAD. The order
            must match ``list_parameters`` exactly. Typically obtained from
            :meth:`sbi_parametric_model.get_model_parameters`.

        list_parameters : list of str
            Names of all parameters in the model, in the same order as
            ``initial_values``. The parameter of interest (POI) is expected
            at index ``0``, followed by unconstrained norm factors, then
            constrained nuisance parameters.

        num_unconstrained_params : int
            Number of leading parameters (starting from index ``0``) that
            are treated as unconstrained (i.e. parameters of interest and
            free norm factors with no Gaussian penalty). Parameters from
            index ``num_unconstrained_params`` onwards are treated as
            constrained nuisance parameters and will be fixed when
            constructing a stat-only NLL curve in
            :meth:`perform_profile_scan`.

        model_grad : callable or None, optional
            A function that returns the gradient of the NLL with respect to
            all parameters.  Signature: ``model_grad(param_array) -> ndarray``.
            If provided, iminuit uses analytical gradients instead of
            finite-difference approximations, reducing the number of NLL
            evaluations by a factor of ~(n_params + 1).

        Notes
        -----
        * ``pulls_global_fit`` is initialised to ``None`` and populated
          only after :meth:`perform_fit` is called. Methods that depend on
          global-fit values (e.g. ``doStatOnly=True`` in
          :meth:`perform_profile_scan`) will raise a ``RuntimeError`` if
          called before :meth:`perform_fit`.

        See Also
        --------
        perform_fit : Run the global MIGRAD minimisation.
        perform_profile_scan : Compute a profiled NLL scan over one parameter.
        """

        self.model_nll                      = model_nll
        self.initial_values                 = initial_values
        self.list_parameters                = list_parameters
        self.num_unconstrained_params       = num_unconstrained_params
        self.model_grad                     = model_grad
        self.pulls_global_fit               = None

    def perform_fit(self, 
                    fit_strategy=2, 
                    freeze_params=[]):
        """
        Run MIGRAD and store best-fit parameter values.

        Parameters
        ----------
        fit_strategy : int
            Minuit strategy (0 = fast, 1 = default, 2 = robust).
        freeze_params : list[str] | None
            List of parameter names to fix during the global fit.

        Notes
        -----
        After a successful fit, ``self.pulls_global_fit`` is set to a
        :class:`numpy.ndarray` of best-fit values. This is required
        before calling :meth:`perform_profile_scan` with
        ``doStatOnly=True``.
        """

        # Instantiate the iminuit object
        m = Minuit(self.model_nll,
                    self.initial_values,
                    grad=self.model_grad,
                    name=tuple(self.list_parameters))
        
        m.errordef = Minuit.LEAST_SQUARES
        strategy = fit_strategy

        # Freeze parameters in freeze_params list to initial values
        if len(freeze_params)>=1:
            for param in freeze_params:
                m.fixed[param] = True

        m.strategy = strategy

        # Run the fit with MIGRAD
        mg = m.migrad()

        # Store best-fit values in parameter order
        self.pulls_global_fit = np.array(m.values)

        # Displays results of the global fit
        print(f'fit: \n{mg}')
    
    def perform_profile_scan(self, 
                      parameter_name: str = '', 
                      bound_range: tuple[float] = (0.0, 3.0), 
                      fit_strategy: int = 2, 
                      freeze_params: list[str] =[], 
                      doStatOnly: bool = False,
                      isConstrainedNP: bool = False,
                      size: int = 100) -> tuple[list[float]]:
        """
        Profile the NLL along `parameter_name` and plot the scan.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to scan (must be in `list_parameters`).
        bound_range : (float, float)
            Scan bounds for profile fit.
        fit_strategy : int
            Minuit strategy for the profile scans.
        freeze_params : list[str] | None
            Parameters to fix for both scans (Stat+Syst and StatOnly).
        doStatOnly : bool
            If True, also produce a "Stat Only" curve by fixing nuisance params
            (those after `num_unconstrained_params`) at their global-fit values.
        isConstrainedNP : bool
            If True, change the y-axis label to t_alpha; else use t_mu.
        size : int
            Number of scan points.

        Returns
        -------
        scan_points : array-like
            Parameter values at which the NLL was evaluated.
        NLL_value : array-like
            :math:`\\Delta NLL` values (minimum-subtracted).
        scan_points_StatOnly : array-like, optional
            Returned only when ``doStatOnly=True``. Scan points for the
            stat-only curve.
        NLL_value_StatOnly : array-like, optional
            Returned only when ``doStatOnly=True``. :math:`\\Delta NLL`
            for the stat-only curve.

        Raises
        ------
        RuntimeError
            If ``doStatOnly=True`` but :meth:`perform_fit` has not been
            called yet.
        """

        m = Minuit(self.model_nll,
                   self.initial_values,
                   grad=self.model_grad,
                   name=tuple(self.list_parameters))
            
        m.errordef = Minuit.LEAST_SQUARES
        m.strategy = fit_strategy

        for param in freeze_params:
            m.fixed[param] = True

        # Profile fit: subtract_min=True returns \Delta NLL
        scan_points, NLL_value, _ = m.mnprofile(parameter_name, 
                                                bound=bound_range, 
                                                subtract_min=True,
                                                size = size)

        # Optionally plot a stat-only NLL curve 
        if doStatOnly: 
            if self.pulls_global_fit is None:
                raise RuntimeError(
                    "perform_fit() must be called before doStatOnly=True, "
                    "so nuisance parameters can be fixed at their global-fit values."
                )

            # Re-initialize with global fit pulls so that fixed params are at the best-fit point
            m_StatOnly = Minuit(self.model_nll,
                                self.pulls_global_fit,
                                grad=self.model_grad,
                                name=tuple(self.list_parameters))
        
            m_StatOnly.errordef = Minuit.LEAST_SQUARES
            m_StatOnly.strategy = fit_strategy

            # Fix the same globally-frozen params
            for param in freeze_params:
                m_StatOnly.fixed[param] = True

            # Additionally fix nuisance parameters (constrained NPs)
            for param in self.list_parameters[self.num_unconstrained_params:]:
                m_StatOnly.fixed[param] = True
            
            scan_points_StatOnly, NLL_value_StatOnly, _ = m_StatOnly.mnprofile(parameter_name, 
                                                                               bound=bound_range, 
                                                                               subtract_min=True,
                                                                                size = size)
            
            return scan_points, NLL_value, scan_points_StatOnly, NLL_value_StatOnly

        else:
            return scan_points, NLL_value
    
