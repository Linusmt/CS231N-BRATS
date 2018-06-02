

python run_model.py --use_dropout=0.15 --model=uX3d --epochs=40 --image_size=64 --weight_decay=0.90
python run_model.py --use_dropout=0.15 --model=uX3d_inception --epochs=40 --image_size=64 --weight_decay=0.90
python plot_models.py --image_size=64 --epochs=30

python run_model.py --use_dropout=0.2 --model=ures --epochs=50 --image_size=64 --weight_decay=0.92 
python run_model.py --use_dropout=0.2 --model=u3d_inception --epochs=50 --image_size=64 --weight_decay=0.92 
