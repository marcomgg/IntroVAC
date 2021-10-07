## Code to reproduce results of IntroVAC

Project files explanation:

program.py: contains the code to train the model and compute the losses, this is where most of the stuff happen

collector.py: is used to collect results and run langevin dynamics to sample from the conditional

modules/models.py: contains the architectures used, Encoder and Decoder are two resnets as reported in the paper

datasets.py: contains a modified version of celeba where we added the FacialHair attribute and 
transformed the attribute no_beard into beard for simplicity

executer.sh: there are the commands to train all the models as in the paper


The trainer saves all logs and models in two folders respectively results/logs and results/save
for every execution a different new folder is created inside them whose name contains the time stamp 
and some of the hyperparameters used to distinguish different executions. Logging is done with Tensorboard.
Once training is done, the collector can be run as follows:

```
python introvac/collector.py \
--folder 'path_to_execution_folder' \
--distance 4.0 \
--ratio 4 \
--best 0 \
--direction 1 0 
```
for the EyeFacial dataset use 
--direction 1 0 0 to add the first attribute
--direction 0 1 0 to add the second attribute
--direction 1 1 0 to add both attributes
the last 0 is always referred to the real/fake class

Note that everything must be run from the top folder "IntroVAC" the one containing the executer

We suggest to create a conda environment as follows:
```
conda create -n introvac python=3
conda activate introvac
pip install -r requirements.txt
```
