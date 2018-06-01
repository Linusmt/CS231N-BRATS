python run_model.py --augment_data=True --use_dropout=0.2 --model=ures --epochs=30 --image_size=64
python run_model.py --augment_data=True --use_dropout=0.2 --model=use_res --epochs=30 --image_size=64
python plot_models.py --image_size=64 --epochs=30

python run_model.py --augment_data=True --use_dropout=0.2 --model=u3d --epochs=30 --image_size=64
python run_model.py --augment_data=True --use_dropout=0.2 --model=use --epochs=30 --image_size=64
python run_model.py --augment_data=True --use_dropout=0.2 --model=baseline --epochs=30 --image_size=64
python plot_models.py --image_size=64 --epochs=30


python run_model.py --augment_data=True --use_dropout=0.2 --model=u3d_inception --epochs=30 --image_size=64
python run_model.py --augment_data=True --use_dropout=0.2 --model=use_inception --epochs=30 --image_size=64
python plot_models.py --image_size=64 --epochs=30

