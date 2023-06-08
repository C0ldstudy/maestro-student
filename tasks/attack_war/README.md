# Attack War_Phase

## Description
Previously, in the attack and defense projects, students have implemented methods to test the vulnerability of the system to differentiate animal activity versus human activity using a vision neural network. Now the goal is to improve further the robustness of the system, we propose a competition where each team can decide to challenge the system by implementing a new attack or a defense.

To start the competition, the best models from attack and defense projects will be chosen. In total, there are going to be 2 rounds, at the beginning of each round, the best models from the previous round will be taken.

## Dowloading datasets
Download the datasets from the teacher version and copy them the following path
```
tasks/attack_war/datasets/CIFAR10/student/
```

## Assignment
Write your attack method in ToDo sections of attack.py in the following path
```
tasks/attack_war/submission/attack.py
```
You could also add other files/models like detector model in defense_project and use it in the code. These folders need to be present in the following folder
```
tasks/attack_war/submission/
```
And you can load these files in you attack.py using attack_path argument passed in attack class instance.

## Evaluating the submissions
Run the evaluator by following steps
```
$ cd tasks/attack_war/
$ python Evaluator_attack_war.py
```

## What to submit
You need to submit attack.py and all the files/models you used for attack.py if any (only as files of submission folder NOT the submission folder)

## Results
View your results at
```
tasks/attack_war/results.json
```

## Evaluation metrics
You will be evaluated on 3 defense models, 2 given to you 1 hidden. For each defense model the score is calculated as below.
A score is generated based on your results(attack success rate, queries, distance). Score is calculated as follows:
```
Score = 70* max(success_rate - 40, 0)/60 +  20 * max(1000 - total_queries, 0)/1000 + 10 * max(15 - distance, 0)/15
```
This is tentative and may change based on submissions.

## Things to be taken note of:
1. get_batch_output function is changed which will output a tuple - (output, detected).
2. You can load/add extra files/models for your attack.py
3. As described [here](../defense_war/README.md), if input image is detected as adversarial image then model returns 4 elements zero array along with a number 1(detection) and attacker won't be able to get output probabilities. And gradients returned by get_batch_input_gradient() are generated directly from the model without any preprocessing and detections functionalities. This is same to as running the defender setup keeping with_preprocess as False and skip_detect as True. So if you want output probabilities and gradients of same input you need to run get_batch_output() with ```with_preprocess=False, skip_detect=True```.
