In order to replicate the results from our experiments, the train and test files need to be executed.

Before running the code make sure your working directory is `Carla-CTS02`
### HyLEAR
* For training run `python train_hylear.py --shared --cuda --port=2000`

`--shared` parameter enables sharing of weights between actor and critic networks of the learner in HyLEAR, `--cuda` switches pyotrch to use cuda, `port=200` defines the carla port

* For testing run `python eval_hylear.py --shared --port=2000 --agent=hylear --test="12"`

`--port=2000` defines carla port, `--test="12"` is an optional parameter and can be used if testing on a single scenario, by default tetsing will be done on all scenarios.

### NavSAC-p
* For training run `python train_sac.py --shared --cuda --port=2000`
* For testing run `python eval_sac.py --shared --port=2000 --test="11"`

`--test="11"` is an optional parameter required only if testing on a single scenario

### A2C-CADRL-p
* For training run `python train_a2c.py -p=2000`
 
where `-p=2000` defines the carla port

* For testing run `python eval_a2c.py --test="01"`

similar to testing for HyLEAR, `--test` is an optional parameter required only if testing on single scenario

### IS-DESPOT-p
* Since no training is required for IS-DESPOT-p, for testing run `python eval_isdespot.py --despot_port=1245 --test="01" --agent="isdespot"`

`--despot_port=1245` defines the port to communicate with the planner, `--test` is the same as HyLEAR and `--agent=isdespot` defines whether to use IS-DESPOT-p or IS-DESPOT-p*. In order to use IS-DESPOT-p*, pass the argument `--agent=isdespot*`

### HyLEAP
* For training run `python train_hyleap.py --despot_port=1250`
* For testing run `python eval_hyleap.py --despot_port=1250 --test="11"`

`--despot_port=1250` defines the port to communicate with the planner, `--test` is an optional parameter same as in HYLEAR

### HyPAL
* For training run `python train_hypal.py --shared --cuda --port=2000`
* For testing run `python eval.py --shared --port=2000 --agent=hypal --test="11"`


## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
