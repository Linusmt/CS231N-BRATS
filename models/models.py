from models.baseline import BaselineModel
from models.Unet3DModel import Unet3DModel
from models.Unet3D_Inception import Unet3DModelInception
from models.URes3DModel import URes3DModel
from models.USE3DModel import USE3DModel
from models.USEnet3D_Inception import USEnet3DModelInception
from models.USERes3DModel import USERes3DModel

MODELS = {"baseline":BaselineModel, 
		  "u3d":Unet3DModel, 
		  "u3d_inception": Unet3DModelInception, 
		  "ures": URes3DModel,
		  "use": USE3DModel,
		  "use_inception": USEnet3DModelInception,
		  "use_res": USERes3DModel,
		 }
