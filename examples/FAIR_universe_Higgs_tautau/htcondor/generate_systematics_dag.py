import yaml, sys, os

config_path = sys.argv[1]
with open(config_path) as f:
    config_all = yaml.safe_load(f)

# nsbi_fit_config is relative to the pipeline config's directory
config_dir = os.path.dirname(config_path)
nsbi_config_path = os.path.join(config_dir, config_all["systematic_uncertainty"]["nsbi_fit_config"])
with open(nsbi_config_path) as f:
    nsbi_config = yaml.safe_load(f)

job_config_path = os.path.basename(config_path)
basis_processes = [s["Name"] for s in nsbi_config["Samples"] if s.get("UseAsBasis")]
n_ensemble = config_all["systematic_uncertainty"].get("num_ensemble_members_training", 1)

lines = []
for dict_syst in nsbi_config["Systematics"]:
    if dict_syst["Type"] != "NormPlusShape":
        continue
    syst = dict_syst["Name"]
    for process in basis_processes:
        if process not in dict_syst["Samples"]:
            continue
        for direction in ["Up", "Dn"]:
            for idx in range(n_ensemble):
                node = f"syst_{process}_{syst}_{direction}_{idx}"
                lines.append(f"JOB {node} examples/FAIR_universe_Higgs_tautau/htcondor/job_systematics_training.sub")
                lines.append(f'VARS {node} PROCESS="{process}" SYSTEMATIC="{syst}" DIRECTION="{direction}" ENSEMBLE_INDEX="{idx}" CONFIG="{job_config_path}" CPUS="8" MEM="16GB" GPUS="1" DISK="32GB"')
                lines.append(f'RETRY {node} 3')
                lines.append("")

print(f"Generated {len(lines)//4} jobs")

with open("examples/FAIR_universe_Higgs_tautau/htcondor/train_systematics.dag", "w") as f:
    f.write("\n".join(lines))
