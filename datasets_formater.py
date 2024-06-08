from collections import Counter
import json
import os
import sys


def main():
    src = sys.argv[1]
    dest = sys.argv[2]
    with open(src, "r") as f:
        vnd = json.load(f)
    variable_names_per_file = vnd["variable_names_per_file"]
    identifiers_per_file = vnd["identifiers_per_file"] if "identifiers_per_file" in vnd else None
    os.makedirs(dest, exist_ok=True)
    c = []
    train_len = 0
    valid_len = 0
    test_len = 0
    for label in variable_names_per_file:
        if "train" in label:
            train_len += 1
        if "valid" in label:
            valid_len += 1
        if "test" in label:
            test_len += 1
    for d, len in zip(["train", "valid", "test"], [train_len, valid_len, test_len]):
        with open(os.path.join(dest, f"{d}.txt"), "w") as f:
            for i in range(len):
                f.write(" ".join(variable_names_per_file[f"{d}#{i}"]) + "\n")
                if d == "train":
                    c += list(variable_names_per_file[f"{d}#{i}"])
        if identifiers_per_file is not None:
            with open(os.path.join(dest, f"{d}_identifiers.txt"), "w") as f:
                for i in range(len):
                    f.write(" ".join(identifiers_per_file[f"{d}#{i}"]) + "\n")
    c = Counter(c)
    with open(os.path.join(dest, f"token.txt"), "w") as f:
        f.write("\n".join(map(lambda x: x[0], c.most_common(20000))))
    with open(os.path.join(dest, f"unktoken.txt"), "w") as f:
        f.write("\n".join(set(c.keys()) - set(map(lambda x: x[0], c.most_common(20000)))))


if __name__ == "__main__":
    main()

