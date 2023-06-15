

import glob
import pickle


def main():
    for filename in glob.glob("results/train_test/**/*.pkl"):
        if filename.endswith("train.pkl") or filename.endswith("test.pkl"):
            continue
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        probs, queries, queries_truncated, shots_truncated = data["probs"]
        data["probs"] = probs
        data["queries"] = queries
        data["queries_truncated"] = queries_truncated
        data["shots_truncated"] = shots_truncated

        with open(filename, "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    main()