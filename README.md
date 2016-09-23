# Mini Project - 1
## Predicting Participation and Race Time for Montreal Marathon 2016

Assignment for Comp 551 Fall 2016, McGill University

## Instructions

To run the classification and regression, use the `runner.py` script and run like follows :

```
python src/runner.py -c [Classification Data File] -r [Regression Data File] -s [Submission File Name]
```

## Data Cleaning

Data Cleaning is done in two parts. First the given dataset is row transformed, basic features extracted and location added using [data-process.py](preprocessing/data-process.py). Then using SQL Server the processed data is again mined in [FinalScript.sql](preprocessing/FinalScript.sql) for new features and made ready for Regression. In order to run the SQL file, Microsoft SQL Server 2016 should be installed and along with it necessary drivers.

The cleaned data can be found [in this public location](https://github.com/koustuvsinha/data-adventures/tree/master/montreal2016).

## Authors

* Koustuv Sinha, McGill ID : 260721248, _koustuv.sinha@mail.mcgill.ca_
* Ramchalam Kinttinkara Ramakrishnan, McGill ID : _ramchalam.kinattinkararamakrishn@mail.mcgill.ca_
* Xiaoqing Ma, McGill ID : 260668927, _xiaoqing.ma@mail.mcgill.ca_
