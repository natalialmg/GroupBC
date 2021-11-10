#### Train initial ERM model (h1) with 40% of total training dataset

python main.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="erm_resnet34_pretrained_batchnorm_sgd1e4_ManualLRDecayPlateau_reg1e4_datared04_CE_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.0001 --loss="CE" --epochs=52 --normlayer="batchnorm" --dataset_reduction=0.4 --train_mode="erm" --scheduler="ManualLRDecayPlateau" > CUB_erm_resnet34_pretrained_batchnorm_sgd1e4_ManualLRDecayPlateau_reg1e4_datared04_CE_seed42_split1_verbose.txt

## Train updated models (h2) adding the remaining data for one of the groups ##

# h2 = erm


# h2 = gmmf



# h2 = grm



# h2 = srm