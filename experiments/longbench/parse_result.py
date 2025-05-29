import json
import argparse
import os

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--output_path', type=str)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    result_path = args.result_path
    model = args.model
    exp_name = args.exp_name
    output_path = args.output_path
    dataset_names = os.listdir(f"{result_path}/{model}")
    result_dict = dict()
    for dataset in dataset_names:
        try:
            with open(f"{result_path}/{model}/{dataset}/{exp_name}/result.json", "r") as f:
                obj = json.load(f)
                scores = obj["score"]
                for k,v in scores.items():
                    exp_conf = k[len(dataset):]
                    result_dict.setdefault(exp_conf, dict())
                    result_dict[exp_conf][dataset] = v
        except:
            print(f"No result file is found in dataset {dataset}")

    for conf, result in result_dict.items():
        score = 0
        for k,v in result.items():
            score += v
        result_dict[conf]["total"] = score

    with open(output_path, "w") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    
    print(result_dict)