# Rethinking Radiology: An Analysis of Different Approaches to BraTS

This is the code for our paper by the same name. Link in the title.

This Project was done for Stanford's CS 231N.

## Licence
MIT

## Architectures & Results



|model|average_time_per_epoch|binary_accuracy|val_binary_accuracy|brats_f1_score|val_brats_f1_score|loss|val_loss|
| --- | --- | --- |--- | --- | --- |--- | -- | 
|u3d_inception_30_64_dropout_0.2|191.2815|0.9982|0.9981|0.6907|0.6112|0.0029|0.0033|
|use_inception_30_64|186.0150|0.9972|0.9972|0.3056|0.3796|0.0065|0.0091|
|use_30_64_dropout_0.2|313.5883|0.9968|0.7164|0.5347|0.2772|0.0958|1.9571|
|use_30_64|296.0747|0.9982|0.9977|0.7176|0.5903|0.0093|0.0638|
|use_res_30_64|331.6290|0.9982|0.9978|0.7209|0.5769|0.0029|0.0045|
|u3d_30_64_dropout_0.2|305.1904|0.9976|0.9977|0.5242|0.4222|0.0225|0.0206|
|baseline_30_64|56.0148|0.9975|0.9977|0.4442|0.4330|0.0054|0.0049|
|ures_30_64_dropout_0.2|350.1806|0.9984|0.9981|0.7790|0.6367|0.0022|0.0031|
|baseline_30_64_dropout_0.2|63.8444|0.9972|0.9975|0.3447|0.3836|0.0063|0.0052|
|use_inception_30_64_dropout_0.2|195.0391|0.9978|0.9973|0.5654|0.5273|0.0042|0.0055|
|use_res_30_64_dropout_0.2|352.4954|0.9983|0.9981|0.7615|0.6348|0.0025|0.0032|
|u3d_30_64|287.2559|0.9983|0.9982|0.7529|0.6389|0.0045|0.0050|
|u3d_inception_30_64|185.4796|0.9981|0.9979|0.6731|0.6204|0.0033|0.0036|


## About
This is part of a project done for Stanford's CS 231N.

## Usage

### Prerequisites
Keras for TensorFlow

### Setup
The requirements files are not yet fully updated and many packages must be install using pip alone.

pip install -r requirements.txt


### Running the files

The models can all be run using the command:

python run_model.py --model="baseline" --epochs=15

Additional options can be found in the models.py file

### Adding models

To add a model, add the models to the MODELS object in the models.py file, and then copy an old model class and update it to with the new model.


### Plotting the models
In order to plot the model histories for a series of models, add them to the MODELS array in plot_models.py, and update the variables for EPOCHS and IMAGE_SIZE. Then run:


python plot_models.py


### Testing if the files run

In order to test the models, you can use the --test_model flag. This runs the model with only 5 examples in the train set and 3 examples in the validation set. If you just want to make sure the model runs after making your updates, you can set the epochs to 2. It also stores these files in the "tmp" directory.

python run_model.py --model="baseline" --epochs=2 --test_model=True

### Plotting the models

Running the plot_models.py function allows you to plot model histories for models filtered by image size and number of epochs. To graph all model histories that ran for 30 epochs with an image size of 64, call:

python plot_models.py --image_size=64 --epochs=30

The resulting merged graphs will be added to the merged_graphs directory

#### Pre-Trained Models

Certain Pretrained models availible upon request.

## [Our Paper]()
COMING SOON :)
