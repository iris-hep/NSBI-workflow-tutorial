# Isotonic, Histogram, and Platt-scaling calibration strategies for NSBI density ratios. HistogramCalibrator bins in linear ratio space; PlattScalingCalibrator fits in log-ratio space internally (the BCE objective requires it) but exposes a ratio-in / ratio-out API. All three public APIs are ratio-in / ratio-out.
# Base part of the code for Histogram-based calibration copied from https://github.com/smsharma/mining-for-substructure-lens
# New weighted quantiles method added, and Platt scaling.

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

LOG_EPS = 1e-30  # floor for ratios before taking log, gives log floor of ~-69 (safe in float64).


class IsotonicCalibrator:
    """Isotonic-regression calibrator that operates in ratio space: fits a monotonic map from the uncalibrated density ratio to the per-event probability."""

    def __init__(self, ratio_predicted, truth_labels, weights):

        self.regressor = IsotonicRegression(out_of_bounds='clip')
        self.regressor.fit(ratio_predicted, truth_labels, sample_weight=weights)

    def cali_pred(self, ratio_uncalibrated):

        calib_score = self.regressor.predict(ratio_uncalibrated)
        calib_score = np.clip(calib_score, 1e-9, 1.0 - 1e-9)
        return calib_score / (1.0 - calib_score)


class HistogramCalibrator:
    """Histogram-based calibrator that bins in linear ratio space.

    Inputs and outputs are density ratios. With method="direct", the per-bin density ratio `hist_num / hist_den` IS the calibrated ratio; with method="indirect" (the legacy "calibrating score directly" branch), the same expression gives a calibrated score and we map back to a ratio via s/(1-s).
    """

    def __init__(self,
                calibration_data_num,
                calibration_data_den,
                w_num, w_den,
                mode="dynamic",
                nbins=100,
                histrange=None,
                method="direct"):

        self.range, self.edges = self._find_binning(
            calibration_data_num, calibration_data_den, mode, nbins, histrange,
            w_num=w_num if mode == "dynamic" else None,
            w_den=w_den if mode == "dynamic" else None,
        )

        self.hist_num, self.num_err = self._fill_histogram(calibration_data_num, w_num)

        self.method = method

        if self.method == "direct":
            self.hist_den, self.den_err = self._fill_histogram(calibration_data_den, w_den)
        else:
            print("calibrating score directly")
            h1, e1 = self._fill_histogram(calibration_data_num, w_num)
            h2, e2 = self._fill_histogram(calibration_data_den, w_den)
            self.hist_den = h1 + h2
            self.den_err  = e1 + e2

    def return_hist(self):
        return self.hist_num, self.hist_den, self.num_err, self.den_err, self.quant_binning

    def cali_pred(self, data):
        indices = self._find_bins(data)
        num = self.hist_num[indices]
        den = self.hist_den[indices]
        cal_pred = num/den

        if self.method == "direct":
            return cal_pred
        else:
            return cal_pred / (1.0 - cal_pred)

    def _find_binning(self, data_num, data_den, mode, nbins, histrange, w_num = None, w_den = None):
        data = np.hstack((data_num, data_den)).flatten()
        if histrange is None:
            hmin = np.min(data)
            hmax = np.max(data)
        else:
            hmin, hmax = histrange

        if mode == "fixed":
            edges = np.linspace(hmin, hmax, nbins + 1)
        elif mode == "dynamic":
            weights = None
            if (w_num is not None) and (w_den is not None):
                weights = np.hstack((w_num, w_den)).astype(float)
            edges = self.weighted_quantile(data, np.linspace(0.0, 1.0, nbins+1), sample_weight=weights)
        elif mode == "dynamic_unweighted":
            percentages = 100.0 * np.linspace(0.0, 1.0, nbins+1)
            edges = np.percentile(data, percentages)

        else:
            raise RuntimeError("Unknown mode {}".format(mode))

        self.quant_binning = edges
        return (hmin, hmax), edges

    def _fill_histogram(self, data, weights, epsilon=1.0e-39):
        histo, _ = np.histogram(data, bins=self.edges, range=self.range, weights=weights)
        i = np.sum(histo)
        histo = histo / i

        err,_ = np.histogram(data, bins=self.edges, range=self.range, weights=weights**2)
        err = err/(i**2)

        return histo, err

    def _find_bins(self, data: np.ndarray):
        idx = np.searchsorted(self.edges, data, side="right") - 1
        idx = np.clip(idx, 0, len(self.edges) - 2)
        return idx

    def weighted_quantile(self, data, quantiles, sample_weight=None):

        values = np.array(data)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]

        return np.interp(quantiles, weighted_quantiles, values)


class PlattScalingCalibrator:
    """Two-parameter Platt / temperature-scaling calibrator in log-ratio space.

    Fits `log r_cal = a · log r_raw + b` by minimising the weighted binary cross-entropy on the training labels. Useful when histogram/isotonic struggle: the affine transform handles tail saturation (`a < 1` deflates over-confidence in the tail) and asymmetric over/under-prediction (`b != 0`). Two parameters means low variance — no overfitting on small calibration sets, no degenerate fits at saturated 0/1 scores.

    Use as a drop-in replacement for IsotonicCalibrator / HistogramCalibrator: same `cali_pred(ratio) -> ratio` contract.
    """

    def __init__(self, ratio_predicted, truth_labels, weights, max_logr=80.0):
        # max_logr clips logr to ±max_logr to keep the optimisation from chasing infinite NN outputs. Default ±80 is comfortably inside float64's exp range (~709).
        self.max_logr = float(max_logr)
        L = np.clip(np.log(np.clip(np.asarray(ratio_predicted, dtype=float), LOG_EPS, None)), -self.max_logr, self.max_logr)
        y = np.asarray(truth_labels, dtype=float)
        w = np.asarray(weights, dtype=float)

        def neg_log_likelihood(params):
            a, b = params
            L_cal = np.clip(a * L + b, -self.max_logr, self.max_logr)
            # Numerically stable BCE in logit space: -log sigmoid(L_cal) = log1p(exp(-L_cal)); -log(1 - sigmoid(L_cal)) = log1p(exp(L_cal)).
            loss_pos = np.log1p(np.exp(-L_cal))
            loss_neg = np.log1p(np.exp( L_cal))
            return float(np.sum(w * (y * loss_pos + (1.0 - y) * loss_neg)))

        result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6})
        if not result.success:
            print(f"PlattScalingCalibrator: optimiser did not converge cleanly ({result.message}); using last iterate (a={result.x[0]:.4f}, b={result.x[1]:.4f}).")
        self.a, self.b = float(result.x[0]), float(result.x[1])

    def cali_pred(self, ratio_uncalibrated):
        L = np.clip(np.log(np.clip(np.asarray(ratio_uncalibrated, dtype=float), LOG_EPS, None)), -self.max_logr, self.max_logr)
        L_cal = np.clip(self.a * L + self.b, -self.max_logr, self.max_logr)
        return np.exp(L_cal)
