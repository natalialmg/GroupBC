

#### Train initial GMMF model (h1) with 40% of total training dataset

python main.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --dataset_reduction=0.4 --train_mode="gmmf" --scheduler="ManualLRDecayNWReset" > CUB_gmmf_resnet34_pretrained_batchnorm_sgd1e4_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1_verbose.txt


## Train updated models (h2) adding the remaining data for one of the groups ##


# h2 = erm


python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC0_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=0 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="erm" --scheduler="ManualLRDecayPlateau" > CUB_h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC0_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC1_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=1 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="erm" --scheduler="ManualLRDecayPlateau" > CUB_h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC1_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC2_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=2 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="erm" --scheduler="ManualLRDecayPlateau" > CUB_h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC2_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC3_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=3 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="erm" --scheduler="ManualLRDecayPlateau" > CUB_h2erm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayPlateau_reg1e4_CE_h1gmmf42data04addC3_seed42_split1_verbose.txt






# h2 = gmmf


python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC0_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=0 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="gmmf" --scheduler="ManualLRDecayNWReset" > CUB_h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC0_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC1_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=1 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="gmmf" --scheduler="ManualLRDecayNWReset" > CUB_h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC1_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC2_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=2 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="gmmf" --scheduler="ManualLRDecayNWReset" > CUB_h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC2_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC3_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=3 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="gmmf" --scheduler="ManualLRDecayNWReset" > CUB_h2gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC3_seed42_split1_verbose.txt









# h2 = grm

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC0_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=0 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="grm" --scheduler="ManualLRDecayNWReset" > CUB_h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC0_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC1_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=1 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="grm" --scheduler="ManualLRDecayNWReset" > CUB_h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC1_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC2_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=2 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="grm" --scheduler="ManualLRDecayNWReset" > CUB_h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC2_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC3_seed42_split1" --gpu=1 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=3 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="grm" --scheduler="ManualLRDecayNWReset" > CUB_h2grm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC3_seed42_split1_verbose.txt




# h2 = srm

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC0_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=0 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="srm" --scheduler="ManualLRDecayNWReset" > CUB_h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC0_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC1_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=1 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="srm" --scheduler="ManualLRDecayNWReset" > CUB_h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC1_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC2_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=2 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="srm" --scheduler="ManualLRDecayNWReset" > CUB_h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC2_seed42_split1_verbose.txt

python main_bc.py --basedir="/data/natalia/models/CUB/" --dataset="CUB" --model_name="h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC3_seed42_split1" --gpu=0 --seed=42 --split=1 --augmentation=True --batch=64 --network="resnet34" --pretrained=True --optim_wreg=0.0001 --optim="sgd" --lr=0.001 --loss="CE" --epochs=52 --min_weight=0.0 --max_weight_change=0.5 --cost_delta_improve=0.25 --normlayer="batchnorm" --addclass=3 --previous_model="/data/natalia/models/CUB/gmmf_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_datared04_mw0wc05c025_CE_seed42_split1" --train_mode="srm" --scheduler="ManualLRDecayNWReset" > CUB_h2srm_resnet34_pretrained_batchnorm_sgd1e3_ManualLRDecayNWReset_reg1e4_mw0wc05c025_CE_h1gmmf42data04addC3_seed42_split1_verbose.txt





