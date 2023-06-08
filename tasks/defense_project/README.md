# Defense Project

## Dowloading datasets
Download the datasets from here and copy them the following path
```
tasks/defense_project/datasets/CIFAR10/student/
```

## Assignment
Complete or revise the train.py and predict.py in the following path and then run the train.py file by the following commands to generate the defense_project-model.pth file.
```
$ cd tasks/defense_project/submission/ # predict.py and train.py are under the path.
$ python train.py
```

## Evaluating the submissions
Run the evaluator by following steps
```
$ cd tasks/defense_project/
$ python Evaluator_defense_project.py
```


## Results
View your results at
```
tasks/defense_project/results.json
```

## Submit
Just submit `predict.py` and the `defense_project-model.pth` by GradeScope without changing the file names.


## Evaluation metrics
The score contains two parts: The raw accuracy score cover 40% and four attack methods cover 60% equally. The raw accuracy over 50% starts to earn points and if the rar accuracy is over 77% you can get the whole 40 points. As to each attack method, if you defend one successfully (the attack success rate is 0%), you will get 15 points. Two attack methods (FGSM and PGD) are released and the other two are hidden.

`raw accuracy score = (min(results["raw_acc"] - 50, 27)/27)*40`
`each attack method score = 15*(100-success_rate)/100`

