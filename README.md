# GroupBC

Code for the methods presented in the paper :


Workshop on Distribution shifts: connecting methods and applications (DistShift Neurips'21)

# Running Experiments

-- Waterbird dataset --

* Original model (h1) is ERM, Updated Model (h2) is ERM,GMMF,GRM,SRM :  /experiments/CUB_erm_experiments.sh  

* Original model (h1) is GMMF (group minimax), Updated Model (h2) is ERM,GMMF,GRM,SRM :  /experiments/CUB_gmmf_experiments.sh  


-- celebA dataset --

* Original model (h1) is ERM, Updated Model (h2) is ERM,GMMF,GRM,SRM :  /experiments/celebA_erm_experiments.sh  

* Original model (h1) is GMMF (group minimax), Updated Model (h2) is ERM,GMMF,GRM,SRM :  /experiments/celebA_gmmf_experiments.sh  

# Analyze Results, Generate Tables 

* notebook/PrintResults.ipynb



