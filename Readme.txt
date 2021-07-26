This repository contains all the scripts and seed data required to run the experiments listed in the dissertation.

The subdirectories can be broken down as follows:
1) Delenox_Experiment_Data: contains the seed autoencoder and populations used for the beginning of a DeLeNoX run.
2) Real-World Datasets: contains two datasets of human-like buildings generated from the aHousev5 dataset.
3) Results: contains the results from my experiments that were included in the dissertation.

The root directory contains all the scripts needed to run the experiments:
1) Main.py is used to run the algorithm, specifying the experiment name and parameters for the transformation phase.
2) Delenox_Config.py and Neat.cfg contain more detailed parameters for the algorithm that were not modified during our testing approach.
3) ExperimentAnalysis.py was used to manually plot the results from the experiment data directory.
4) The rest of the python scripts handle the individual modules of the system