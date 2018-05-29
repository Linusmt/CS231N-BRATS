# CS 231N Project

## Setup
The requirements files are not yet fully updated and many packages must be install using pip alone.

pip install -r requirements.txt


## Running the files

The models can all be run using the command:

python models.py --model="baseline" --epochs=15

Additional options can be found in the models.py file

## Adding models

To add a model, add the models to the MODELS object in the models.py file, and then copy an old model class and update it to with the new model.


## Plotting the models
In order to plot the model histories for a series of models, add them to the MODELS array in plot_models.py, and update the variables for EPOCHS and IMAGE_SIZE. Then run:


python plot_models.py