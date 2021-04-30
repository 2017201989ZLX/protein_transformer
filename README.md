# protein_transformer
This is a basic implentaion for prediction of the interactions between peptide chains and amino acids.
# enviroment and dependicies
pytorch1.4.0
DGL0.4.3
lie_learn
wandb
# process data
the code need three json file as the input data, the small example code is shown in example_data/
they are atom_feature.json,train_pair.json,test_pair.json
# train and test the model
run python3 experiments/qm9/train.py to train the model
use the eval function in experiments/qm9/train.py to evaluate the model
