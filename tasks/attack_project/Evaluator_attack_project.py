import sys
sys.path.append("../../")
from utils import *
import argparse
from EvaluatePair import EvaluatePair, VirtualModel
import json

def evaluate_attack(defense_list, attack_path, dataset, device, target_label=None):
    results = {}
    for defense_path in defense_list:
        e = EvaluatePair(attack_path, defense_path, dataset, device)
        r = e.evaluate(target_label)
        results["attacker_success_rate"] = r['targeted_adv_sr']
        results["time"] = r['run_time']
        results["dist"] = r['distance']
        results["predict"] = r['predict_queries']
        results["gradient"] = r['gradient_queries']

        results["total_queries"] = results["predict"] + results["gradient"]
        if results["gradient"] == 0:
            results["attack_type"] = "BlackBox"
        else:
            results["attack_type"] = "WhiteBox"

        results["score"] = (max(results["attacker_success_rate"] - 40, 0)/60) * 70 + (max(1000 - results["total_queries"], 0)/1000) * 20 + (max(15-results["dist"],0)/15) * 10

        # if (100 - r['targeted_adv_acc'] > 96) and (r['distance'] < 7.75) and (r['predict_queries'] < 8500):
        #     results["meets_expectations"] = "True"
        # else:
        #     results["meets_expectations"] = "False"

    return results

def run():
    parser = argparse.ArgumentParser(description="Attack Homework Evaluation")
    parser.add_argument('--data_path', type=str, default='datasets/CIFAR10/student', help='path to the folder containing datasets')
    parser.add_argument('--folder_path', type=str, default='submission', help='the folder path that need to evaluate. If evaluating the attack, use tasks/attack_homework.')
    parser.add_argument('--defender_path', type=str, default='defender', help='the folder path that need to evaluate. If evaluating the attack, use tasks/war_attack.')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    # print(args)
    students_submission_path = args.folder_path
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
    defense_list = [
            args.defender_path,
        ]
    target_label = 1
    results = evaluate_attack(defense_list, students_submission_path, dataset,  device, target_label)
    print(results)
    file_path = 'results.json'
    with open(file_path, 'w') as f:
        json.dump(str(results),f, indent= 4)
    return results


if __name__ == "__main__":
    run()
