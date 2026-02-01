import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import mplhep as hep
import math
from scipy import stats

hep.set_style("ATLAS")

# method for plotting lines
def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color='gray')

# General method for filling weighted histograms 
def fill_histograms_wError(data, weights, edges, histrange, normalize=True):
        
    h, _ = np.histogram(data, edges, histrange, weights=weights)
    
    if normalize:
        i = np.sum(h)

        h = h/i
    
    h_err, _ = np.histogram(data, edges, histrange, weights=weights**2)
    
    if normalize:
        
        h_err = h_err/(i**2)
    
    return h, h_err

# Diagnostics for training loss and accuracy 
def plot_loss(loss_history, path_to_figures=""):

    plt.plot(loss_history.train_loss)
    plt.plot(loss_history.val_loss)
    plt.title("model loss", size=12)
    plt.ylabel("loss", size=12)
    plt.xlabel("epoch", size=12)
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f"{path_to_figures}/loss_plot.png", bbox_inches="tight")
    plt.clf()


def plot_calibration_curve(data_den, weight_den, data_num, weight_num, 
                           data_den_holdout, weight_den_holdout, data_num_holdout, weight_num_holdout, 
                           path_to_figures="", nbins=100, epsilon=1.0e-20, 
                           label="Calibration Curve", score_range="standard"):

    data = np.concatenate([data_num, data_den, data_den_holdout, data_num_holdout]).flatten()
    xmin = np.amin(data)
    xmax = np.amax(data)
    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin,xmax)

    # Fill histograms
    hist_den, hist_den_err = fill_histograms_wError(data_den, weight_den, edges, histrange)
    hist_num, hist_num_err = fill_histograms_wError(data_num, weight_num, edges, histrange)

    hist_ratio = hist_num / (hist_den + hist_num)
    hist_ratio_err = (
        hist_ratio**2 * np.abs(hist_den / hist_num) *
        np.sqrt((hist_num_err / (hist_num)**2) + (hist_den_err / (hist_den)**2))
    )

    # Holdout histograms
    hist_den_holdout, hist_den_holdout_err = fill_histograms_wError(
        data_den_holdout, weight_den_holdout, edges, histrange)
    hist_num_holdout, hist_num_holdout_err = fill_histograms_wError(
        data_num_holdout, weight_num_holdout, edges, histrange)

    hist_ratio_holdout = hist_num_holdout / (hist_den_holdout + hist_num_holdout)
    hist_ratio_err_holdout = (
        hist_ratio_holdout**2 * np.abs(hist_den_holdout / hist_num_holdout) *
        np.sqrt((hist_num_holdout_err / (hist_num_holdout)**2) + (hist_den_holdout_err / (hist_den_holdout)**2))
    )

    slopeOne = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='col',
                             gridspec_kw={'height_ratios': [3, 1]})
    plt.rc('xtick', labelsize=16)

    # ---------- TRAINING ----------
    plt.sca(axes[0, 0])
    hep.histplot(hist_ratio, bins=edges, yerr=hist_ratio_err, label='')
    abline(1.0, 0.0)
    plt.title(label + " (Training)", fontsize=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-0.1, ymax=1.1)
    plt.ylabel("Probability ratio", size=16)
    # plt.legend(loc='lower right', fontsize=16)

    residue = (hist_ratio - slopeOne) / hist_ratio_err
    plt.sca(axes[1, 0])
    plt.errorbar(slopeOne, residue, yerr=1.0, drawstyle='steps-mid')
    abline(0.0, 0.0)
    plt.xlabel("Predicted Score", size=16)
    plt.ylabel("Residue", size=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0, ymax=4.0)

    # ---------- HOLDOUT ----------
    plt.sca(axes[0, 1])
    hep.histplot(hist_ratio_holdout, bins=edges, yerr=hist_ratio_err_holdout, label='', color="blue")
    abline(1.0, 0.0)
    plt.title(label + " (Holdout)", fontsize=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-0.1, ymax=1.1)
    # plt.legend(loc='lower right', fontsize=16)

    residue_holdout = (hist_ratio_holdout - slopeOne) / hist_ratio_err_holdout
    plt.sca(axes[1, 1])
    plt.errorbar(slopeOne, residue_holdout, yerr=1.0, drawstyle='steps-mid', color="blue")
    abline(0.0, 0.0)
    plt.xlabel("Predicted Score", size=16)
    plt.ylabel("Residue", size=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0, ymax=4.0)

    # ---------- Finalize ----------
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path_to_figures}/calib_plot_{score_range}.png", bbox_inches='tight')
    plt.clf()

def plot_calibration_curve_ratio(
    data_den, weight_den, data_num, weight_num, 
    data_den_holdout, weight_den_holdout, data_num_holdout, weight_num_holdout, 
    path_to_figures="", nbins=100, epsilon=1.0e-20, 
    label="Calibration Curve", score_range="standard"):

    # --- transform scores to logit (LLR proxy) ---
    data_den        = np.log(data_den / (1.0 - data_den))
    data_den_holdout= np.log(data_den_holdout / (1.0 - data_den_holdout))
    data_num        = np.log(data_num / (1.0 - data_num))
    data_num_holdout= np.log(data_num_holdout / (1.0 - data_num_holdout))

    data = np.concatenate([data_num, data_den, data_den_holdout, data_num_holdout]).flatten()
    xmin = np.amin(data); xmax = np.amax(data)
    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin, xmax)

    # --- TRAINING histograms (weighted counts + their standard errors) ---
    hist_den, hist_den_err = fill_histograms_wError(data_den, weight_den, edges, histrange)
    hist_num, hist_num_err = fill_histograms_wError(data_num, weight_num, edges, histrange)

    # Safe counts to avoid divisions by zero in the error propagation
    N  = np.maximum(hist_num, epsilon)
    D  = np.maximum(hist_den, epsilon)

    # LLR per bin and its error via delta method: Var[log(N/D)] = (σN/N)^2 + (σD/D)^2
    hist_ratio     = np.log(N / D)
    hist_ratio_err = np.sqrt((np.sqrt(hist_num_err) / N)**2 + (np.sqrt(hist_den_err) / D)**2)

    # --- HOLDOUT histograms ---
    hist_den_h, hist_den_err_h = fill_histograms_wError(data_den_holdout, weight_den_holdout, edges, histrange)
    hist_num_h, hist_num_err_h = fill_histograms_wError(data_num_holdout, weight_num_holdout, edges, histrange)

    Nh = np.maximum(hist_num_h, epsilon)
    Dh = np.maximum(hist_den_h,  epsilon)

    hist_ratio_h     = np.log(Nh / Dh)
    hist_ratio_err_h = np.sqrt((np.sqrt(hist_num_err_h) / Nh)**2 + (np.sqrt(hist_den_err_h) / Dh)**2)

    # Bin centers (x) for residuals
    slopeOne = (edges[:-1] + edges[1:]) / 2

    # Mask bins with undefined/zero uncertainty to avoid inf residuals
    err_ok          = hist_ratio_err    > 0
    err_ok_holdout  = hist_ratio_err_h  > 0

    # --- Layout: training (left), holdout (right) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='col',
                             gridspec_kw={'height_ratios': [3, 1]})
    plt.rc('xtick', labelsize=16)

    # ---------- TRAINING ----------
    plt.sca(axes[0, 0])
    hep.histplot(hist_ratio, bins=edges, yerr=hist_ratio_err, label='')
    abline(1.0, 0.0)
    plt.title(label + " (Training)", fontsize=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-10, ymax=10)
    plt.ylabel("Probability ratio", size=16)

    plt.sca(axes[1, 0])
    residue = np.empty_like(hist_ratio); residue[:] = np.nan
    residue[err_ok] = (hist_ratio[err_ok] - slopeOne[err_ok]) / hist_ratio_err[err_ok]
    plt.errorbar(slopeOne[err_ok], residue[err_ok], yerr=1.0, drawstyle='steps-mid')
    abline(0.0, 0.0)
    plt.xlabel("Predicted Score", size=16)
    plt.ylabel("Residue", size=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0, ymax=4.0)

    # ---------- HOLDOUT ----------
    plt.sca(axes[0, 1])
    hep.histplot(hist_ratio_h, bins=edges, yerr=hist_ratio_err_h, label='', color="blue")
    abline(1.0, 0.0)
    plt.title(label + " (Holdout)", fontsize=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-10, ymax=10)

    plt.sca(axes[1, 1])
    residue_h = np.empty_like(hist_ratio_h); residue_h[:] = np.nan
    residue_h[err_ok_holdout] = (hist_ratio_h[err_ok_holdout] - slopeOne[err_ok_holdout]) / hist_ratio_err_h[err_ok_holdout]
    plt.errorbar(slopeOne[err_ok_holdout], residue_h[err_ok_holdout], yerr=1.0, drawstyle='steps-mid', color="blue")
    abline(0.0, 0.0)
    plt.xlabel("Predicted Score", size=16)
    plt.ylabel("Residue", size=16)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0, ymax=4.0)

    # --- finalize ---
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path_to_figures}/calib_plot_llr_{score_range}.png", bbox_inches='tight')
    plt.clf()


def plot_reweighted(
    dataset_train, score_den_train, weight_den_train, score_num_train, weight_num_train,
    dataset_holdout, score_den_holdout, weight_den_holdout, score_num_holdout, weight_num_holdout,
    path_to_figures="", num=15, variables=['NN_MELA_incl_disc'],
    sample_name=['Bkg','Ref'], scale="linear",
    label_left='Training Data Diagnostic', label_right='Holdout Data Diagnostic'
):
    """
    Draw training (left) and holdout (right) reweighting diagnostic plots side-by-side..
    """

    # Helper that reproduces original per-panel computations
    def _panel_data(dataset, score_den, weight_den, score_num, weight_num, variable):
        data_den = dataset[dataset.train_labels==0].copy()
        data_den['score'] = score_den
        data_num = dataset[dataset.train_labels==1].copy()
        data_num['score'] = score_num

        weight_den_arr = np.ravel(data_den.weights)
        weight_num_arr = np.ravel(data_num.weights)

        score_den_arr = np.ravel(data_den.score)
        ratio_den = score_den_arr / (1.0 - score_den_arr)  # pA/pB from score
        den_to_num_rwt = weight_den_arr * ratio_den

        var_den = np.ravel(data_den[variable])
        var_num = np.ravel(data_num[variable])

        concat = np.concatenate([var_den, var_num]).flatten()
        xmin = np.amin(concat); xmax = np.amax(concat)

        edges = np.linspace(xmin, xmax, num=num+1)
        histrange = (xmin, xmax)

        # reweighted ref -> target, and target (as in original)
        hist_den,  hist_den_err  = fill_histograms_wError(var_den, den_to_num_rwt, edges, histrange)
        hist_num,  hist_num_err  = fill_histograms_wError(var_num, weight_num_arr, edges, histrange)

        # original (unreweighted) reference and target (for comparison)
        hist_deno, hist_deno_err = fill_histograms_wError(var_den, weight_den_arr, edges, histrange)
        hist_numo, hist_numo_err = fill_histograms_wError(var_num, weight_num_arr, edges, histrange)

        return {
            'edges': edges, 'xmin': xmin, 'xmax': xmax,
            'hist_den': hist_den, 'hist_den_err': hist_den_err,
            'hist_num': hist_num, 'hist_num_err': hist_num_err,
            'hist_deno': hist_deno, 'hist_deno_err': hist_deno_err,
            'hist_numo': hist_numo, 'hist_numo_err': hist_numo_err
        }

    for variable in variables:
        # Compute data for each panel independently (same as original behavior)
        L = _panel_data(dataset_train,   score_den_train,   weight_den_train,   score_num_train,   weight_num_train,   variable)
        R = _panel_data(dataset_holdout, score_den_holdout, weight_den_holdout, score_num_holdout, weight_num_holdout, variable)

        # Figure: 2 rows (main, ratio) × 2 cols (training, holdout)
        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='col',
                                 gridspec_kw={'height_ratios': [3, 1]})
        plt.rc('xtick', labelsize=16)

        # ---------- LEFT (Training) MAIN ----------
        plt.sca(axes[0,0])
        hep.histplot(L['hist_den'],  L['edges'], yerr=np.sqrt(L['hist_den_err']),
                     label=str(sample_name[1])+'->'+str(sample_name[0])+' rwt', linewidth=2.0)
        hep.histplot(L['hist_num'],  L['edges'], yerr=np.sqrt(L['hist_num_err']),
                     label=sample_name[0], linewidth=2.0)
        hep.histplot(L['hist_deno'], L['edges'], yerr=np.sqrt(L['hist_deno_err']),
                     label=sample_name[1], linewidth=2.0)
        plt.title(label_left, fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.ylabel("Normalized events", size=16)
        plt.axis(xmin=L['xmin'], xmax=L['xmax'])
        if scale=='log': plt.yscale('log')

        # ---------- LEFT (Training) RATIO ----------
        plt.sca(axes[1,0])
        rat_L = L['hist_den']/L['hist_num']
        yerr_L = np.abs(rat_L*np.sqrt((np.sqrt(L['hist_den_err'])/L['hist_den'])**2 + (np.sqrt(L['hist_num_err'])/L['hist_num'])**2))
        hep.histplot(rat_L, L['edges'], yerr=yerr_L, linewidth=2.0)
        plt.axis(xmin=L['xmin'], xmax=L['xmax'], ymin=0.5, ymax=1.5)
        plt.xlabel(variable, size=16)
        plt.ylabel("Ratio", loc='center', size=16)
        abline(0.0,1.0)

        # ---------- RIGHT (Holdout) MAIN ----------
        plt.sca(axes[0,1])
        hep.histplot(R['hist_den'],  R['edges'], yerr=np.sqrt(R['hist_den_err']),
                     label=str(sample_name[1])+'->'+str(sample_name[0])+' rwt', linewidth=2.0)
        hep.histplot(R['hist_num'],  R['edges'], yerr=np.sqrt(R['hist_num_err']),
                     label=sample_name[0], linewidth=2.0)
        hep.histplot(R['hist_deno'], R['edges'], yerr=np.sqrt(R['hist_deno_err']),
                     label=sample_name[1], linewidth=2.0)
        plt.title(label_right, fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.axis(xmin=R['xmin'], xmax=R['xmax'])
        if scale=='log': plt.yscale('log')

        # ---------- RIGHT (Holdout) RATIO ----------
        plt.sca(axes[1,1])
        rat_R = R['hist_den']/R['hist_num']
        yerr_R = np.abs(rat_R*np.sqrt((np.sqrt(R['hist_den_err'])/R['hist_den'])**2 + (np.sqrt(R['hist_num_err'])/R['hist_num'])**2))
        hep.histplot(rat_R, R['edges'], yerr=yerr_R, linewidth=2.0)
        plt.axis(xmin=R['xmin'], xmax=R['xmax'], ymin=0.5, ymax=1.5)
        plt.xlabel(variable, size=16)
        plt.ylabel("Ratio", loc='center', size=16)
        abline(0.0,1.0)

        plt.tight_layout()
        plt.show()
        plt.savefig(f'{path_to_figures}/reweighted_{str(variable)}.png', bbox_inches='tight')
        plt.clf()

def plot_overfit(score_1, score_2, w_train, w_test, nbins=50, 
                 plotRange=[0.0,1.0], holdout_index=1, 
                 label='X', path_to_figures=""):

    bins = np.linspace(plotRange[0],plotRange[1],num=nbins)

    w_train = w_train/w_train.sum()
    w_test = w_test/w_test.sum()

    score_1_h, bins = np.histogram(score_1, bins=bins, weights=w_train)
    score_2_h, bins = np.histogram(score_2, bins=bins, weights=w_test)

    score_1_err, bins = np.histogram(score_1, bins=bins, weights=w_train**2)
    score_2_err, bins = np.histogram(score_2, bins=bins, weights=w_test**2)

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.1,.3,.6,.6),xticklabels=([]))
    hep.histplot(score_1_h, bins, yerr=np.sqrt(score_1_err), label='Holdout')
    hep.histplot(score_2_h, bins, yerr=np.sqrt(score_2_err), label='Train')

    plt.title("Overfit Plot for "+str(label), fontsize=18)
    plt.ylabel("Normalized", fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.axis(xmin=0.0, xmax=1.0)

    #Residual plot
    difference = score_1_h - score_2_h
    frame2=fig1.add_axes((.1,.1,.6,.2))
    hep.histplot(difference, bins, yerr = np.sqrt(score_2_err+score_1_err))
    plt.axis(xmin=0.0, xmax=1.0, ymin=-0.0015, ymax=0.0015)

    plt.xlabel("NN Prediction", fontsize=18)
    plt.ylabel("Residue", loc="center", size=18)
    abline(0.0,0.0)
    plt.savefig(f'{path_to_figures}/overfit_'+str(holdout_index)+'_'+str(label)+'.png', bbox_inches='tight')
    plt.show()
    plt.clf()

def plot_overfit_side_by_side(
    score_den_train, score_den_holdout, w_den_train, w_den_holdout,
    score_num_train, score_num_holdout, w_num_train, w_num_holdout,
    nbins=50, plotRange=[0.0,1.0], holdout_index=1,
    labels=('Bkg','Ref'),  # left=den label, right=num label
    path_to_figures=""
):
    """
    Draw overfit diagnostics for DEN (left) and NUM (right) side-by-side.
    Each column shows Train & Holdout superimposed (same aesthetics as original).
    """

    # Common binning (same as original)
    bins = np.linspace(plotRange[0], plotRange[1], num=nbins)

    def _compute(score_1, score_2, w_train, w_test):
        # NOTE: keep original normalization and error definitions exactly
        w_train = w_train / w_train.sum()
        w_test  = w_test  / w_test.sum()

        h1, _ = np.histogram(score_1, bins=bins, weights=w_train)
        h2, _ = np.histogram(score_2, bins=bins, weights=w_test)

        e1, _ = np.histogram(score_1, bins=bins, weights=w_train**2)
        e2, _ = np.histogram(score_2, bins=bins, weights=w_test**2)

        return h1, h2, e1, e2

    # Left (den) panel data
    den_h1, den_h2, den_e1, den_e2 = _compute(
        score_den_train, score_den_holdout, w_den_train, w_den_holdout
    )
    # Right (num) panel data
    num_h1, num_h2, num_e1, num_e2 = _compute(
        score_num_train, score_num_holdout, w_num_train, w_num_holdout
    )

    # Figure: 2 rows (main + residual) × 2 cols (den + num)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='col',
                             gridspec_kw={'height_ratios': [3, 1]})
    plt.rc('xtick', labelsize=18)

    # ---------- LEFT (DEN) MAIN ----------
    plt.sca(axes[0,0])
    hep.histplot(den_h1, bins, yerr=np.sqrt(den_e1), label='Holdout')
    hep.histplot(den_h2, bins, yerr=np.sqrt(den_e2), label='Train')
    plt.title("Overfit Plot for " + str(labels[0]), fontsize=18)
    plt.ylabel("Normalized", fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.axis(xmin=plotRange[0], xmax=plotRange[1])

    # ---------- LEFT (DEN) RESIDUAL ----------
    plt.sca(axes[1,0])
    difference_den = den_h1 - den_h2
    hep.histplot(difference_den, bins, yerr=np.sqrt(den_e2 + den_e1))
    plt.axis(xmin=plotRange[0], xmax=plotRange[1], ymin=-0.0015, ymax=0.0015)
    plt.xlabel("NN Prediction", fontsize=18)
    plt.ylabel("Residue", loc="center", size=18)
    abline(0.0, 0.0)

    # ---------- RIGHT (NUM) MAIN ----------
    plt.sca(axes[0,1])
    hep.histplot(num_h1, bins, yerr=np.sqrt(num_e1), label='Holdout')
    hep.histplot(num_h2, bins, yerr=np.sqrt(num_e2), label='Train')
    plt.title("Overfit Plot for " + str(labels[1]), fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.axis(xmin=plotRange[0], xmax=plotRange[1])

    # ---------- RIGHT (NUM) RESIDUAL ----------
    plt.sca(axes[1,1])
    difference_num = num_h1 - num_h2
    hep.histplot(difference_num, bins, yerr=np.sqrt(num_e2 + num_e1))
    plt.axis(xmin=plotRange[0], xmax=plotRange[1], ymin=-0.0015, ymax=0.0015)
    plt.xlabel("NN Prediction", fontsize=18)
    plt.ylabel("Residue", loc="center", size=18)
    abline(0.0, 0.0)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{path_to_figures}/overfit_side_by_side_{holdout_index}.png', bbox_inches='tight')
    plt.clf()


def plot_all_features(dataframe, weight_array, label_array, nbins=20):
    
    # get list of feature names
    features = list(dataframe.columns)
    n_features = len(features)

    # define grid: 2 rows × ceil(n_features/2) columns
    n_rows = 2
    n_cols = math.ceil(n_features / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)

    # flatten axes array for easy indexing
    axes_flat = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes_flat[i]
        kin_feature = dataframe[feature].to_numpy()

        # compute bins
        min_val = np.amin(kin_feature)
        max_val = np.amax(kin_feature)
        bins = np.linspace(min_val, max_val, nbins)

        kin_feature_label_0 = kin_feature[label_array==0]
        kin_feature_label_1 = kin_feature[label_array==1]

        hist_kin_feature_label_0 = np.histogram(kin_feature_label_0, 
                                                        weights=weight_array[label_array==0],
                                                        bins = bins)[0]

        hist_kin_feature_label_0_err = np.sqrt(np.histogram(kin_feature_label_0, 
                                                        weights=weight_array[label_array==0]**2,
                                                        bins = bins)[0])
        
        hist_kin_feature_label_1 = np.histogram(kin_feature_label_1, 
                                                        weights=weight_array[label_array==1],
                                                        bins = bins)[0]

        hist_kin_feature_label_1_err = np.sqrt(np.histogram(kin_feature_label_1, 
                                                        weights=weight_array[label_array==1]**2,
                                                        bins = bins)[0])

        # plot weighted histogram on this axis
        hep.histplot(hist_kin_feature_label_0,
                     yerr = hist_kin_feature_label_0_err,
                     bins=bins,
                     ax=ax,
                     label = 'class 0')
        
        hep.histplot(hist_kin_feature_label_1,
                     yerr = hist_kin_feature_label_1_err,
                     bins=bins,
                     ax=ax,
                     label = 'class 1')

        ax.set_xlabel(feature)
        plt.ylabel('Density')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls=":", lw=0.5)

    # turn off any unused subplots
    for j in range(n_features, n_rows * n_cols):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()
        
