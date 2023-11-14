import os
import argparse
import numpy as np
from config import DatasetConfig


parser = argparse.ArgumentParser(description="Generate data for Chapter 0")
parser.add_argument("--num-rules", type=int, default=10, help="The number of rules to generate")
parser.add_argument("--num-per-rule", type=int, default=10000, help="The number of data points generated from a rule")
parser.add_argument("--missing-rate", type=float, default=0.1, help="The rate of missing values")

args = parser.parse_args()

cfg = DatasetConfig()

class Rule:
    def __init__(self, id : int) -> None:
        self.id = id
        self.n = np.random.randint(cfg.n_range[0], cfg.n_range[1]+1)
        self.s_interval = np.array([np.random.randint(cfg.s0_range[0],cfg.s0_range[1]+1)] + [np.random.randint(cfg.steplen_range[0], cfg.steplen_range[1]+1) for _ in range(self.n-1)])
        self.s = np.concatenate([np.cumsum(self.s_interval), [np.inf]])
        self.c0 = np.random.randint(cfg.c0_range[0],cfg.c0_range[1]+1)
        self.c = np.array([np.random.uniform(cfg.ci_range[0],cfg.ci_range[1]+1) for _ in range(self.n)])

rules = [Rule(i) for i in range(args.num_rules)]

def calculate_price(dists : float, rule : Rule):
    eps = np.random.normal(0, cfg.ep_sigma, dists.shape[0])
    mask = (dists[:,np.newaxis] >= rule.s)[: ,:-1]
    intervals = np.concatenate([rule.s_interval[1:],[np.inf]])
    price_per_interval = rule.c * np.min(np.stack([dists[:, np.newaxis]-rule.s[:-1], np.repeat(intervals[np.newaxis, :], dists.shape[0],0)],axis=-1), axis=-1)
    masked_price = price_per_interval * mask
    return rule.c0 + np.sum(masked_price,axis=1) + eps

def generate_data_from_rule(rule : Rule, size : int):
    xs = np.random.uniform(0, cfg.x_ubound, size)
    ys = calculate_price(xs, rule)
    rs = np.array([rule.id for _ in range(size)])
    res = np.stack([xs, ys, rs], axis=-1).reshape(-1,3)
    missing_value = (np.random.uniform(0,1,size) <= 0.1)
    which_to_nan = np.random.randint(0,2,size)[missing_value]
    for i in range(len(which_to_nan)):
        res[missing_value][i][which_to_nan[i]] = np.nan
    return res

if __name__ == "__main__":
    data = np.concatenate([generate_data_from_rule(rule, args.num_per_rule) for rule in rules])
    np.random.shuffle(data)
    if not os.path.exists("./data"):
        os.makedirs("data")
    np.save("./data/dataset.npy", data)
    print(f"Generate {args.num_rules * args.num_per_rule} points")
    print("Save to data/dataset.npy")
