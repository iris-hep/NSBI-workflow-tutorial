FAIR Universe Dataset
--

The tabular dataset used in this demonstration is hosted on Zenodo (https://zenodo.org/records/15131565), and is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The dataset provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - good challenge to test the sensitivty of NSBI techniques.

**NB** If you need access to pre-trained ensemble NNs, feel free to download the directory [here](https://cernbox.cern.ch/files/spaces/eos/user/j/jsandesa/NSBI_tutorial_data/saved_datasets) and move it inside `NSBI-workflow-tutorial/FAIR_Universe_Higgs_tautau/` in your cloned copy.

Workflow
--

1. [`DataLoader.ipynb`](./DataLoader.ipynb)

This notebook is to download the FAIR Universe data and store it in the form of `.root` ntuples. This step only needs to be done once and is independent of the overall NSBI workflow. 

2. [`DataPreprocessing.ipynb`](./DataPreprocessing.ipynb)

This notebook defines selections and preselects events in the signal and control regions.

3. [`Preselection_withNN.ipynb`](./Preselection_withNN.ipynb) 

This notebook trains a multi-class classification neural network for signal/control region definitions.

4. [`Neural_Likelihood_Ratio_Estimation.ipynb`](./Neural_Likelihood_Ratio_Estimation.ipynb)

This notebook is used to train the desnity ratios to be used for modeling the negative log-likelihood ratios for inference.

5. [`Systematic_Uncertainty_Estimation.ipynb`](./Systematic_Uncertainty_Estimation.ipynb)

This notebook is used to train the density ratios associated with the systematic uncertainty model, then used for modeling the negative log-likelihood ratios for inference.

6. [`Parameter_Fitting_with_Systematics.ipynb`](./Parameter_Fitting_with_Systematics.ipynb)

This notebook is used to create a workspace object and perform parameter fitting with the NSBI model.
