# Virgin Media Petfinder:

Some of the experiments are shown in experimenting.ipynb.
The dataset suffers from class imbalance and sever feature skewness, I tried performing multiple transformations to the features to reduce the impact of their skewed distribution, due to class imbalance i supply class weights to the models.

## Want to run it on Docker?

```bash
make build
```

run the following command for training (Task 1):
```bash
make train
```

run the following command for inference (Task 2):
```bash
make infer
```

## Want to run it locally?

```bash
python -m venv .venv
python install -r requirements.txt
```

run the following command for training (Task 1):
```bash
python run.py --train
```

run the following command for inference (Task 2):
```bash
python run.py --infer
```

------
Mo
