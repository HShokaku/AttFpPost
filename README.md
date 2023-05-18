# AttFpPost_trained
A repository for trained AttFpPost to make predictions

## Supported Predictions
-   hERG:   Whether to inhibit hERG
-   BBB:    Whether it can cross the blood-brain barrier
-   CYP2C9: Whether to inhibit CYP2C9
-   CYP3A4: Whether to inhibit CYP3A4
-   Pgp-inhibitor:  Whether to inhibit P-gp
## Future Plans
-   Pgp-substrate:  Whether it is a P-gp substrate

## Setup
```
conda create -n postnet python==3.9
conda install -c conda-forge rdkit
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge tqdm
conda install -c conda-forge dgllife
conda install -c dglteam dgl-cuda11.1
pip install typed-argument-parser
conda install -c conda-forge mordred
pip install pyro-ppl
```

## Usage
open `single_molecule_prediction.ipynb` 
```
smiles = 'CCCSC1=CC2=C(NC(NC(=O)OC)=N2)C=C1' # input your molecule
task_name = 'Pgp-inhibitor' # choose your prediction task

single_molecule_prediction(smiles, task_name)
```
