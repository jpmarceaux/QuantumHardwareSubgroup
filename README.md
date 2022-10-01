## Installation guide
- (Highly recommended) Create a `conda` environment  of name `qcircuit`: 
```
conda create -n qcircuit python==3.9
conda activate qcircuit
```
- Run `pip install -r requirements.txt`
- In `lib` folder, run `pip install -e .`
- Attach the conda environment to jupyter via `python -m ipykernel install --user --name=qcircuit`