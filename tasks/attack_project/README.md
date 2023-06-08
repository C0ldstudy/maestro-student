# Attack Project

## Description
Approximately 20 percent of the global forest area around the world is currently under some form of legal protection. The Wilderness Society is testing the robustness of monitoring systems based on computer vision targeting several legal reserves in the Amazon rainforest, located in Central America.

The proposed monitoring systems are able to recognize animals and vehicles using a vision neural network (VGG11), divided into four classes:

    Class 0: air and water vehicles (ships and airplanes)
    Class 1: land vehicles (trucks and automobiles)
    Class 2: small animals (bird, deer, cat)
    Class 3: large animals (dog, horse, deer)

The final goal is the identification of activities or events that can trigger alarms for damaging human intrusion. In particular, adversaries want to make the alarm classifier generate false positives, so they are interested in making it predict normal animals (classes 2 and 3) to the alarming land vehicles (class 1). We have been asked to test the robustness of the proposed system by implementing several adversarial attacks.

To achieve this goal, we are going to attack the vision model in teams. This is very similar to the homework, except:

    This is a different task. Instead of digits, the model will get a color image that it has to classify.
    You will be given images from classes 2 and 3, and you have to have them be predicted as one of class 1.
    The scoring is using a combination of success rate, distance, and queries, all continuous.
    You are free to implement any attack you like, using the predictions and gradients from the model. An FGSM implementation is included, along with the code you have from the attack homework.


## Dowloading datasets
Download the datasets from the teacher version and copy them the following path
```
tasks/attack_project/datasets/CIFAR10/student/
```

## Assignment
Write your attack method in ToDo sections of attack.py in the following path
```
tasks/attack_project/submission/attack.py
```

## Evaluating the submissions
Run the evaluator by following steps
```
$ cd tasks/attack_project/
$ python Evaluator_attack_project.py
```

## Results
View your results at
```
tasks/attack_project/results.json
```

## Submmit
Just submit `attack.py` by GradeScope without changing the file name.


## Evaluation metrics
A final score is generated based on your results(attack success rate, queries, distance). Score is calculated as follows:
```
Final Score = 70* max(success_rate - 40, 0)/60 +  20 * max(1000 - total_queries, 0)/1000 + 10 * max(15 - distance, 0)/15
```
This is tentative and may change based on submissions.

## FGSM attack -- Reference
FGSM implementation is the following location
```
tasks/attack_project/FGSM/attack.py
```
run your code using entering the following commands in the terminal
```
$ cd tasks/attack_project/
$ python Evaluator_attack_project.py --folder_path FGSM
```

# GOOD LUCK !!!
