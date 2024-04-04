# Adaptive HAC

## Running the code
Normal run with set parameters via arguments.  
Call: "python main.py {node_method} {sample_method} {subsegmentation} {features}"

## Sweeping the code
Multiple combinations for the processing can be sweeped by using hydra multi-run.

### Sweeping parameters
    - Config file for parameter sweep: "paramsweep.yaml"
    - Run sweep with: "python main.py -m --config-name=paramsweep.yaml"

### Sweeping features
    - Config file for feature sweep: "featuresweep.yaml"
    - Run sweep with: "python main.py -m --config-name=featuresweep.yaml"
