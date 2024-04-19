# Adaptive HAC

## Running the code
Normal run with set parameters via arguments.  
Call: "python main.py {node_method} {sample_method} {subsegmentation} {features}"

### Running on GPU machine (linux)
 - Make sure the matlab license is activated by first running: "/bulk/software/MATLAB/R2023b/bin/matlab"

#### Dowloading data (linux)
scp -rp {username}@sftp.tudelft.nl:/staff-groups/ewi/me/MS3/MS3-Shared/Ro
nny_MonostaticData /home/{username}/Gitlab/AdaptiveHAC/AdaptiveHAC/test/data

## Sweeping the code
Multiple combinations for the processing can be sweeped by using hydra multi-run.

### Sweeping parameters
    - Config file for parameter sweep: "paramsweep.yaml"
    - Run sweep with: "python main.py -m --config-name=paramsweep.yaml"

### Sweeping features
    - Config file for feature sweep: "featuresweep.yaml"
    - Run sweep with: "python main.py -m --config-name=featuresweep.yaml"

