

## Packages

To successfully run the project, the following Python packages need to be installed:

```
captum==0.6.0
libclang==16.0.6
numpy==1.26.3
pandas==2.2.0
scikit_learn==1.3.2
tokenizers==0.15.1
torch==2.1.0
tqdm==4.66.1
transformers==4.34.0
gensim==4.3.2
```



### Datasets

The dataset (big_vul) is provided by [Fan et al.](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git) We focus on the following 3 columns to conduct our experiments:

1. `func_before` (str): The original function written in C/C++.
2. `target` (int): The function-level label that determines whether a function is vulnerable or not.
3. `flaw_line_index` (str): The labeled index of the vulnerable statement. 

| func_before | target | flaw_line_index |
| ----------- | ------ | --------------- |
| ...         | ...    | ...             |



Datasets can be downloaded from:

```
https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ
https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
```

The entire dataset without splitting can be downloaded from:

```
https://drive.google.com/uc?id=1WqvMoALIbL3V1KNQpGvvTIuc3TL5v5Q8
```

The preprocessed dataset can be downloaded from:

```
https://drive.google.com/file/d/1496S3mWWcqKmPJWvcZDjJP3EigOnE9Wl https://drive.google.com/file/d/1FG_qVz7waxvWmoh09CH1ibLENwtN2igb https://drive.google.com/file/d/1Qpc1E1ISDe7uR9yEG8AGqxefH4atVzYV
```

After downloading, place all data into `./resource/Dataset/target`.

(Note that some of this these dataset links are provided by the author of [LineVul](https://github.com/awsm-research/LineVul.git), which is an important transformer-based vulnerability detection method and also serves as a crucial baseline for our research.)

## RFLD Model

RFLD model can be downloaded from the links below:

```
https://drive.google.com/file/d/1NdriaylgmaNvrIhw6YT6a32C92iJoxPW
https://drive.google.com/file/d/1OUDY-eT-Txy7D2OSZ8ulbUwiaHDKFjrI
```

After downloading, place the file into `./resource/SavedModels`.

## How to run the model

`Entry/main.py` is the entry point for training and evaluation. Please run this file to reproduce the results presented in our paper.