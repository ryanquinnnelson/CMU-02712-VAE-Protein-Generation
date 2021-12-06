# CMU-02712-PROJECT

Fall 2021 Biological Modeling and Simulation - Group Project

## Summary
The focus of this project is to take a set of protein sequences, train a generative model on the data,
then use the generative model to produce novel sequences based on the training data.

This codebase contains multiple components.
- `/bin`: contains scripts used with VAE training
- `/configs`: hyperparameter configurations for VAE training
- `/customized`: custom code for VAE training of this dataset
- `/data`: input and output for converting FASTA data into csv/txt format
- `/octopus`: generalized framework code for VAE training
- `/sample_output`: example of the logging and Weights & Biases model training for the VAE.
- `protein_data_from_fasta.py`: Convert FASTA data into a csv/txt format.
- `run_octopus.py`: Wrapper code to run VAE training.


## Requirements

- `wandb` account: Running the VAE model code requires a `wandb` account.


## To Run the Code for VAE Model

1. The code for this model consists of the following components:
    - python module `octopus`
    - python module `customized`
    - bash script `mount_drive`

2. Activate an environment that contains torch, pandas, numpy. I used `pytorch_latest_p37`
   environment as part of the Deep Learning AMI for AWS EC2 instances.

3. Define each configuration file as desired. Configuration file options are listed below. See configuration file
   "config_vae_prop_r31.txt" as example.

4. If your instance has a mountable storage drive (i.e. `/dev/nvme1n1`), execute the bash script to mount the drive and
   change permissions to allow the user to write to directories inside the drive.

```bash
$ mount_drive 
```

5. Execute the code using the following command from the shell to run `octopus` for each configuration file in the
   configuration directory. Files will be processed in alphabetical order.

```bash
$ python run_octopus.py path/to/config_directory
```




## Configuration File Options

Configurations must be parsable by `configparser`.

```text
[DEFAULT]
run_name = PaperVAE-Run-31  # sets the name of the run in wandb, log, checkpoints, and output files 

[debug]
debug_path = /home/ubuntu/  # where log file will be saved



[pip]
packages=--upgrade wandb==0.10.8  # commands to install particular pip packages during startup

[wandb]      
wandb_dir = /home/ubuntu/                             # fully qualified directory for wandb internal files
entity = ryanquinnnelson                              # wandb account name
project = CMU-02712-PROJECT-Propeller-Proteins        # project to save model data under
notes = Protein Generation using VAE                  # any notes to save with this run
tags = CNN,octopus,VAE                                # any tags to save with this run

[stats]
comparison_metric=val_acc                 # the evaluation metric used to compare this model against previous runs 
comparison_best_is_max=True               # whether a maximum value for the evaluation metric indicates the best performance
comparison_patience=40                    # number of epochs current model can underperform previous runs before early stopping
val_metric_name=val_acc                   # the name of the second metric returned from Evaluation.evalute_model() for clarity in stats

[data]
data_type = numerical                                                                               # indicates data is not image based
data_dir = /home/ubuntu/content/data/proteins 						       # fully qualified path to root of data subdirectories
train_data= /home/ubuntu/content/data/proteins/beta_propeller_encoded_train.npy                     # fully qualified path to training data
val_data=/home/ubuntu/content/data/proteins/beta_propeller_encoded_val.npy                          # fully qualified path to validation data

[output]
output_dir = /home/ubuntu/output         # fully qualified directory where test output should be written


[dataloader]
num_workers=8        # number of workers for use in DataLoader when a GPU is available
pin_memory=True      # whether to use pin memory in DataLoader when a GPU is available
batch_size=256       # batch size regardless of a GPU or CPU

[model]
model_type=CnnLSTM   # type of model to initialize.
lstm_input_size=256 # Dimension of features being input into the LSTM portion of the model.
hidden_size=256     # Dimension of each hidden layer in the LSTM model.
num_layers=5        # Number of LSTM layers in LSTM portion of the model.
bidirectional=True  # True if LSTM is bidirectional. False otherwise.
dropout=0.2         # The percent of node dropout in the LSTM model.
lin1_output_size=42 # The number of labels in the feature dimension of the first linear layer if there are multiple linear layers.
lin1_dropout=0.0    # The percent of node dropout in between linear layers in the model if there are multiple linear layers.
output_size=42      # The number of labels in the feature dimension of linear layer output.


model_type=PaperVAE   	# type of model to initialize.
input_size=11000      	# dimension of features used as input.
hidden_sizes=8192,256	# dimension of each of the intermediate hidden layers. At least one hidden layer is required.
latent_dim=128		# dimension of the latent feature space.
batch_normalization=True	# whether or not to perform batch normalization after each intermediate hidden layer in the model.
dropout=0.4		# the percent of dropout in each of the intermediate hidden layers in the model. Use 0.0 to not use dropout.




[checkpoint]
checkpoint_dir = /data/checkpoints   # fully qualified directory where checkpoints will be saved
delete_existing_checkpoints = False  # whether to delete all checkpoints in the checkpoint directory before starting model training (overridden to False if model is being loaded from a previous checkpoint)
load_from_checkpoint=False           # whether model will be loaded from a previous checkpoint
checkpoint_file = None               # fully qualified checkpoint file

[hyperparameters]
num_epochs = 50                                               		# number of epochs to train
criterion_type=CustomLoss1                                    		# the loss function to use
criterion_kwargs=use_burn_in=False,delta_burn_in=0.003,burn_in_start=0      # any arguments to be passed to the criterion function
optimizer_type=Adam                                            		# optimizer to use
optimizer_kwargs=lr=0.005     						# any optimizer arguments
scheduler_type=ReduceLROnPlateau                              		# the scheduler to use
scheduler_kwargs=factor=0.75,patience=3,mode=max,verbose=True  		# any scheduler arguments
scheduler_plateau_metric=val_acc                              		# if using a plateau-based scheduler, the evaluation metric tracked by the scheduler 
```



## On the differences between VAE versions

I compared two versions of Variational Autoencoders:

1) https://avandekleut.github.io/vae/
2) https://github.com/psipred/protein-vae/

### Standard Deviation vs Variance

I found there were differences in how the authors interpreted the output of the second linear layer in the Encoder.

1) Assumes this value is the log of the std dev: log(sigma).
2) Assumes this value is the log of the variance: log(sigma^2)

This can be determined by how each uses this value to obtain z (called sample in 2).

1) z = mu + exp(log(sigma)) * N(0,1)
2) z = mu + exp(log(sigma^2) / 2) * N(0,1)

Note that version 2 sends the output of the second linear layer through a Softplus layer as well. Softplus simply
smooths the output, so it shouldn't be able to account for the difference between log(sigma) to log(sigma^2).

We've decided to follow the assumptions of version 2. We assume the output of the second linear layer in the Encoder to
be log(sigma^2).

### Calculating KL Loss

I found there were differences in how the authors calculated KL divergence.

1) KL = (1.0 * sigma^2 + 1.0 * mu^2 - log(sigma) - 0.5).sum()
2) KL = (0.5 * sigma^2 + 0.5 * mu^2 - log(sigma) - 0.5).sum()

Version 2 adds half as much of each of the first two terms.


