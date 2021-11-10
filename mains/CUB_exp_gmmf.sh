


python main.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="gmmf_resnet34_pretrained_batchnorm_sgd1e4_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.0001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --dataset_reduction=0.4 --train_mode="gmmf" --scheduler="ManualLRDecayNWReset" > CUB_gmmf_resnet34_pretrained_batchnorm_sgd1e4_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1_verbose.txt
