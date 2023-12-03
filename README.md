# maestro-student
This repository contains the student version of the maestro platform to learn adversarial machine learning algorithm.


![](./logo.png)

## Installation
To install the Maestro Platform, follow these steps:
1. Create a python environment by [conda](https://docs.anaconda.com/free/anaconda/install/index.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html).
```bash
# take conda as an example
conda create -n maestro-student python==3.9.11
conda activate maestro-student
```
2. Clone the repository:
```bash
git clone git@github.com:C0ldstudy/maestro-student.git
```
3. Navigate to the project directory

4. Install the required python packages
```bash
pip install -r requirements.txt
```

## Usage
We list six tasks for students to complete. It also supports adding other assigments follow our basic code structure.
For each task, we provide a brief README file that includes the background information, required datasets, and evaluation requirements. To gain a deeper understanding of each task, we also list related knowledge and papers for further reading.

- Attack Homework: [Gentic Algorithm](https://arxiv.org/abs/1906.03181)
- Attack Project: [PGD](https://arxiv.org/abs/1706.06083), [CW](https://arxiv.org/abs/1608.04644).
- Defense Homework/Project: [Adversarial Training](https://arxiv.org/abs/1412.6572)

## Teacher Version
If you are interested in obtaining the teacher version, please send a request to `maestro.uci@gmail.com`.

## Citation
```
@article{geleta2023maestro,
  title={Maestro: A Gamified Platform for Teaching AI Robustness},
  author={Geleta, Margarita and Xu, Jiacen and Loya, Manikanta and Wang, Junlin and Singh, Sameer and Li, Zhou and Gago-Masague, Sergio},
  journal={arXiv preprint arXiv:2306.08238},
  year={2023}
}
```
