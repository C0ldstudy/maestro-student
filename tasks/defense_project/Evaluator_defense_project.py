import sys
sys.path.append("../../")
from utils import *
import argparse
from EvaluatePair import EvaluatePair, VirtualModel
import json

def evaluate_defense(attack_list, defense_path, dataset, device, target_label=None):
    results = {}
    RAW = 0
    for attack_path in attack_list:
        e = EvaluatePair(attack_path, defense_path, dataset, device)
        if RAW == 0:
            r = e.raw_evaluate()
            results['raw_acc'] = r['raw_acc']
            results["score"] = (min(results["raw_acc"] - 50, 27)/27)*40
            RAW = 1
        r = e.evaluate(target_label)
        results[attack_path.split('/')[-1].split('.')[0]+"_targeted_sr"] = r['targeted_adv_sr']
        results["score"] += 15 * (100 - r['targeted_adv_sr']) / 100
    return results


def run():
    parser = argparse.ArgumentParser(description="Defense Homework Evaluation")
    parser.add_argument('--data_path', type=str, default='datasets/CIFAR10/student/', help='path to the folder containing datasets')
    parser.add_argument('--folder_path', type=str, default='', help='the folder path that need to evaluate. such as tasks/defense_homework')
    parser.add_argument('--defender_path', type=str, default='attacker_list', help='the folder path that need to evaluate.')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    # print(args)
    students_submission_path = args.folder_path + 'submission'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: ", device)
    dataset_configs = {
                "name": "CIFAR10",
                "binary": True,
                "dataset_path": args.data_path,
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }
    dataset = get_dataset(dataset_configs)
    attack_list = [
            args.defender_path + "/target_FGSM",
            ]
    target_label = 0
    results = evaluate_defense(attack_list, students_submission_path, dataset,  device, target_label)
    print(results)
    file_path = 'results.json'
    with open(file_path, 'w') as f:
        json.dump(results,f, indent= 4)
    return results


if __name__ == "__main__":
    run()
