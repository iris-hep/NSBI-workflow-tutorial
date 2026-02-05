import os, sys
import argparse
import logging
import warnings
import matplotlib.pyplot as plt
import mplhep as hep
import yaml
import jax
import contextlib

sys.path.append('../src')
import nsbi_common_utils
from nsbi_common_utils import workspace_builder, model, inference

jax.config.update("jax_enable_x64", True)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set Style
hep.style.use(hep.style.ATLAS)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and parameter fitting.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.pipeline.yaml",
        help="Path to the main pipeline configuration file."
    )
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_dir):
    """
    Sets up the root logger to write to both Console and File.
    Returns the logger and the path to the log file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "fit_results.log")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    
    file_fmt = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_fmt)
    
    # Console Handler (Message only)
    console_fmt = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    
    return root_logger, log_file

def build_workspace(cfg_path):
    """Builds the workspace."""
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    ws_builder = nsbi_common_utils.workspace_builder.WorkspaceBuilder(config_path=cfg_path)
    return ws_builder.build()

def save_nll_plot(scan_data, output_dir, parameter_label):
    """Plots the NLL scans and saves the figure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    
    for item in scan_data:
        plt.plot(
            item['points'], 
            item['nll'], 
            label=item['label'], 
            linestyle=item['style'], 
            color=item['color'],
            linewidth=2
        )
    
    plt.xlabel(parameter_label, fontsize=20)
    plt.ylabel(r"$-2\Delta \ln L$", fontsize=20)
    plt.legend(fontsize=14, frameon=False)
    plt.ylim(0, 8)
    plt.xlim(0, 3)
    
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(4, color='gray', linestyle=':', alpha=0.5)
    
    hep.atlas.label(data=False, label="Internal", loc=0)
    
    output_path = os.path.join(output_dir, "nll_scan_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    args = parse_args()
    cfg_main = load_config(args.config)["parameter_fitting"]
    
    
    plots_dir = cfg_main["output"]["plots_dir"]
    logs_dir = cfg_main["output"]["logs_dir"]
    

    logger, log_file = setup_logging(logs_dir)
    
    logger.info("Starting Inference Pipeline...")
    logger.info(f"Configurations: Hist={cfg_main['configs']['hist']}, NSBI={cfg_main['configs']['nsbi']}")

    measurement = cfg_main["measurement"]
    scan_param = cfg_main["scan"]["parameter"]
    scan_range = tuple(cfg_main["scan"]["range"])
    scan_steps = cfg_main["scan"]["steps"]

    try:
        
        logger.info("\n=== Building Workspaces ===")
        ws_hist = build_workspace(cfg_main["configs"]["hist"])
        ws_nsbi = build_workspace(cfg_main["configs"]["nsbi"])
        
        
        logger.info("\n=== Initializing Models ===")
        model_hist = nsbi_common_utils.model.Model(workspace=ws_hist, measurement_to_fit=measurement)
        model_nsbi = nsbi_common_utils.model.Model(workspace=ws_nsbi, measurement_to_fit=measurement)
        
        list_params, init_values = model_hist.get_model_parameters()
        num_unconstrained = model_hist.num_unconstrained_param
        
        inf_hist = nsbi_common_utils.inference.inference(
            model_nll=model_hist.model,
            initial_values=init_values,
            list_parameters=list_params,
            num_unconstrained_params=num_unconstrained
        )
        
        inf_nsbi = nsbi_common_utils.inference.inference(
            model_nll=model_nsbi.model,
            initial_values=init_values,
            list_parameters=list_params,
            num_unconstrained_params=num_unconstrained
        )

        logger.info("\n=== Performing Fits (Tables logged to file) ===")
        
        with open(log_file, "a") as f, contextlib.redirect_stdout(f):
            print("\n" + "="*40)
            print(" NSBI FIT RESULTS ")
            print("="*40 + "\n")
            inf_nsbi.perform_fit()
            
            print("\n" + "="*40)
            print(" HISTOGRAM FIT RESULTS ")
            print("="*40 + "\n")
            inf_hist.perform_fit()

        logger.info(f"\n=== Running Profile Scans for {scan_param} ===")
        freeze_params = []

        logger.info("Scanning Histogram Model...")
        pts_hist, nll_hist, pts_stat_hist, nll_stat_hist = inf_hist.perform_profile_scan(
            parameter_name=scan_param,
            freeze_params=freeze_params,
            bound_range=scan_range,
            fit_strategy=0,
            doStatOnly=True,
            size=scan_steps
        )

        logger.info("Scanning NSBI Model...")
        pts_nsbi, nll_nsbi, pts_stat_nsbi, nll_stat_nsbi = inf_nsbi.perform_profile_scan(
            parameter_name=scan_param,
            freeze_params=freeze_params,
            bound_range=scan_range,
            fit_strategy=0,
            doStatOnly=True,
            size=scan_steps
        )

        
        logger.info("\n=== Generating Plots ===")
        
        plot_data = [
            {
                'points': pts_hist, 'nll': nll_hist, 
                'label': "Histogram Stat+Syst", 'style': "-", 'color': "blue"
            },
            {
                'points': pts_stat_hist, 'nll': nll_stat_hist, 
                'label': "Histogram Stat Only", 'style': "--", 'color': "blue"
            },
            {
                'points': pts_nsbi, 'nll': nll_nsbi, 
                'label': "NSBI Stat+Syst", 'style': "-", 'color': "black"
            },
            {
                'points': pts_stat_nsbi, 'nll': nll_stat_nsbi, 
                'label': "NSBI Stat Only", 'style': "--", 'color': "black"
            }
        ]
        
        parameter_label_latex = r'$\mu_{h\tau\tau}$' 
        plot_path = save_nll_plot(plot_data, plots_dir, parameter_label_latex)
        
        logger.info(f"Plot saved to: {plot_path}")
        logger.info(f"Full log file: {log_file}")
        logger.info("Inference workflow completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()