# Label-Ownership-Verification
Official implementation of Label Ownership Verification

---

## Abstract

High-quality labels constitute valuable intellectual property that requires detecting unauthorized use. Existing dataset ownership verification (DOV) methods have proven effective in detecting unauthorized use of dataset, relying on image tampering to inject watermarks; however, label owners cannot modify images in released datasets (which are independently provided by image owners) and can only alter the labels. Consequently, existing DOV techniques cannot be directly leveraged to safeguard label copyright. To fill this gap, this paper proposes Label Ownership Verification via Backdoor Watermarking (LVBW) which verifies whether a specific label owner's labels are utilized during the training of the black-box target model. Specifically, our method consists of two main components: (i) label watermarking, where labels are watermarked via a clean-image backdoor attack, and (ii) label ownership verification, where a hypothesis-test-guided method is designed to verify ownership based on probability vectors. We also provide theoretical analyses to ensure the credibility of the verification results. Extensive experiments demonstrate the effectiveness of our method and its resilience against potential adaptive methods. 

---

## In this repo

### Existing modules:

1. `base_utils`: Utility module, used by the base modules.
1. `train_expert`: training expert models and recording trajectories.
1. `generate_labels`: generating poisoned labels from trajectories.
1. `select_flips`: strategically alter hard labels within some budget.
1. `train_user`: Evaluation module to assess attack success rate.

---

## Installation

```
conda install --file requirements.txt
```

Note that the requirements encapsulate our testing enviornments and may be unnecessarily tight! Any relevant updates to the requirements are welcomed.

## Running An Experiment

### Setting up:

To initialize an experiment, create a subfolder in the `experiments` folder with the name of your experiment:

```
mkdir experiments/[experiment name]
```

In that folder initialize a config file called `config.toml`. An example can be seen here: `experiments/example_attack/config.toml`.

The `.toml` file should contain references to the modules that you would like to run with each relevant field as defined by its documentation in `schemas/[module name]`. This file will serve as the configuration file for the entire experiment. As a convention the output for module **n** is the input for module **n + 1**.

**Note:** the `[INTERNAL]` block of a schema should not be transferred into a config file.

```
[module_name_1]
output=...
field2=...
...
fieldn=...

[module_name_2]
input=...
output=...
...
fieldn=...

...

[module_name_k]
input=...
field2=...
...
fieldn=...
```

### Running a module:

At the moment, all experiments must be manually run using:

```
python run_experiment.py [experiment name]
```

The experiment will automatically pick up on the configuration provided by the file. 

As an example, to run the `example_attack` experiment one could run:

```
python run_experiment.py example_attack
```

More module documentation can be found in the `schemas` folder.
