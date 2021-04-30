## protein_transformer
This is a basic implentaion for prediction of the interactions between peptide chains and amino acids.
### enviroment and dependencies
pytorch1.4.0

DGL0.4.3

lie_learn

wandb

the model need to import funtion from se(3)-transformer
### process data
the code need three json file as the input data, the small example code is shown in experiment/protein/

they are atom_feature.json,train_pair.json,test_pair.json
### train and test the model
run `python3 experiment/protein/train.py` to train the model

use the eval function in experiment/protein/train.py to evaluate the model
