# CS 231N Project

## Setup
The requirements files are not yet fully updated and many packages must be install using pip alone.

pip install -r requirements.txt


## Running the files

The models can all be run using the command:

python run_model.py --model="baseline" --epochs=15

Additional options can be found in the models.py file

## Adding models

To add a model, add the models to the MODELS object in the models.py file, and then copy an old model class and update it to with the new model.


## Plotting the models
In order to plot the model histories for a series of models, add them to the MODELS array in plot_models.py, and update the variables for EPOCHS and IMAGE_SIZE. Then run:


python plot_models.py


## Testing if the files run

In order to test the models, you can use the --test_model flag. This runs the model with only 5 examples in the train set and 3 examples in the validation set. If you just want to make sure the model runs after making your updates, you can set the epochs to 2. It also stores these files in the "tmp" directory.

python run_model.py --model="baseline" --epochs=2 --test_model=True

## Plotting the models

Running the plot_models.py function allows you to plot model histories for models filtered by image size and number of epochs. To graph all model histories that ran for 30 epochs with an image size of 64, call:

python plot_models.py --image_size=64 --epochs=30

The resulting merged graphs will be added to the merged_graphs directory