# import nsbi_common_utils
import yaml, sys, os

config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)["neural_likelihood_ratio_estimation"]

job_config_path = os.path.basename(config_path)

# nsbi_fit_config_path = config["nsbi_fit_config"]
# logger.info(f"Initializing NSBI ConfigManager from: {nsbi_fit_config_path}")
# fit_config_nsbi = nsbi_common_utils.configuration.ConfigManager(file_path_string=nsbi_fit_config_path)
# 
# basis_processes = fit_config_nsbi.get_basis_samples()
# logger.info(f"Basis processes: {basis_processes}")

basis_processes = config["basis_processes_to_train"]
print(basis_processes)

lines = []
for process in basis_processes:
    n_ensemble = config["num_ensemble_members_training"]
    for idx in range(n_ensemble):
        node = f"train_{process}_{idx}"
        lines.append(f"JOB {node} examples/FAIR_universe_Higgs_tautau/htcondor/job_density_ratio_training.sub")
        lines.append(f'VARS {node} PROCESS_TYPE="{process}" ENSEMBLE_INDEX="{idx}" CONFIG="{job_config_path}" CPUS="8" MEM="16GB" GPUS="1" DISK="32GB"')
        lines.append("")

with open("examples/FAIR_universe_Higgs_tautau/htcondor/train_ensemble.dag", "w") as f:
    f.write("\n".join(lines))
