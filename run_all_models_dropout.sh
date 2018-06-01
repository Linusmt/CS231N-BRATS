python models.py --use_dropout=0.2 --augment_data=True --model=ures --epochs=30 --image_size=64
python models.py --use_dropout=0.2 --augment_data=True --model=use_res --epochs=30 --image_size=64

python models.py --use_dropout=0.2 --augment_data=True --model=u3d --epochs=30 --image_size=64
python models.py --use_dropout=0.2 --augment_data=True --model=use --epochs=30 --image_size=64
python models.py --use_dropout=0.2 --augment_data=True --model=baseline --epochs=30 --image_size=64


python models.py --use_dropout=0.2 --augment_data=True --model=u3d_inception --epochs=30 --image_size=64
python models.py --use_dropout=0.2 --augment_data=True --model=use_inception --epochs=30 --image_size=64
