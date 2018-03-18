import utils


data = utils.get_brats_data('./BRATS/BRATS2015_Training/HGG/**/*T1c*.mha', './BRATS/BRATS2015_Training/HGG/**/*OT*.mha', [256,256,256], model_name='im_256', preprocessed=False, save=True)
