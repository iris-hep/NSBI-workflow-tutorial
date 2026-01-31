import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt

def calculate_preselection_observable(pred_NN_incl, samples_list, signal_processes, background_processes, pre_factor_dict = {'htautau': 1.0, 'ttbar': 1.0, 'ztautau': 1.0}):

    signal_sum = np.sum(
        [pre_factor_dict[signal] * pred_NN_incl[:, samples_list[signal]] for signal in signal_processes], axis=0
    )

    background_sum = np.sum(
        [pre_factor_dict[background] * pred_NN_incl[:, samples_list[background]] for background in background_processes], axis=0
    )
    # the preselection score as defined above - log(P_S/P_B)
    presel_score = np.log(signal_sum/background_sum)

    return presel_score


def preselection_using_score(dataset, channel_selections):

    mask_channel = {}
    dataset_channel = {}

    for channel, selection_dict in channel_selections.items():

        selection = selection_dict['preselections']
        
        dataset_channel[channel] = dataset.query(selection)

    return dataset_channel


def plot_kinematic_features(
    columns,
    nbins,
    variations_to_plot,
    dataset_dict,
    xlabel_dict,
    samples_list
):
    """
    For each feature in `columns`, builds weighted histograms for each (label, variation)
    and then plots them on a grid of subplots.

    Parameters
    ----------
    columns : list of str
        List of dataframe‐column names to histogram.
    nbins : int
        Number of bins to use (based on the 'nominal' variation of each feature).
    variations_to_plot : list of str
        Keys in `dataset_dict` (e.g. ['nominal','TES_up','TES_dn']).
    dataset_dict : dict[str -> pandas.DataFrame]
        Mapping each variation name to a DataFrame.  Each DataFrame must have columns:
            - feature columns in `columns`
            - "detailed_labels" (to mask by label)
            - "weights"
    xlabel_dict : dict[str->str]
        Mapping each feature name → label for the x‐axis.
    samples_list : list
        The list of sample labels to loop over.

    Returns
    -------
    fig, axes
        The Figure and array of Axes where the histograms were drawn.
    """
    # Compute all histograms and bins
    hist = {feat: {lbl: {} for lbl in samples_list} for feat in columns}
    bins_dict = {}

    for feature in columns:
        for label in samples_list:
            # Compute bins from the 'nominal' dataset
            arr_nom = dataset_dict['Nominal'][label][feature].to_numpy()
            bins = np.histogram_bin_edges(arr_nom, bins=nbins)
            bins_dict[feature] = bins
            break

        # For each variation & label, fill the histogram
        for variation in variations_to_plot:
            for label in samples_list:
                df_var = dataset_dict[variation][label]
                vals = df_var[feature].to_numpy()
                wts = df_var["weights"].to_numpy()
                hist_vals, _ = np.histogram(vals, weights=wts, bins=bins)
                hist[feature][label][variation] = hist_vals

    # Set up a grid of subplots (up to 2 columns, adjusting rows if needed)
    n_plots = len(columns)
    ncols = 2
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    # Prepare color/linestyle maps
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_label_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(samples_list)}
    linestyle_map = {
        variations_to_plot[0]: '-',
        variations_to_plot[1]: '--',
        variations_to_plot[2]: '--'
    }

    # Plot each feature in its own Axes
    for ax, feature in zip(axes, columns):
        for label in samples_list:
            for variation in variations_to_plot:
                hep.histplot(
                    hist[feature][label][variation],
                    bins=bins_dict[feature],
                    label=(label if variation == 'Nominal' else None),
                    ax=ax,
                    linewidth=1.5,
                    color=color_label_map[label],
                    linestyle=linestyle_map.get(variation, '-')
                )

        ax.set_yscale('log')
        ax.set_xlabel(xlabel_dict.get(feature, feature), size=14)
        ax.set_ylabel('Events', size=14)
        ax.legend()

    # If there are unused subplots (e.g. 3 features → 4 subplots), turn off extras
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig, axes





