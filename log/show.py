import os
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','fashionmnist','emnist','purchase','chmnist'])
    parser.add_argument('--E',  type=int, default=1,
                        help='the index of experiment in AE')
    args = parser.parse_args()
    return args

args = parse_arguments()

E = args.E
data = args.data

if E == 1:
    results_df = pd.DataFrame(columns=["data","mode","epsilon","accuracy"])
    directory = 'log/E1'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
               path = os.path.join(root, file)
               if data in path:
                   df = pd.read_csv(path, header=None)
                   new_header = df.iloc[0]
                   df = df[1:]
                   df.columns = new_header
                   results_df = results_df._append(
                       {"data": df["data"].values[0],"mode": df["mode"].values[0],
                        "epsilon": df["epsilon"].values[0], "accuracy": df["accuracy"].values[0]},
                       ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by = 'mode').reset_index(drop=True))


elif E == 2:
    results_df = pd.DataFrame(columns=["data","mode","model","epsilon","accuracy"])
    directory = 'log/E2'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
               path = os.path.join(root, file)
               if data in path:
                   df = pd.read_csv(path, header=None)
                   new_header = df.iloc[0]
                   df = df[1:]
                   df.columns = new_header
                   results_df = results_df._append(
                       {"data": df["data"].values[0],"mode": df["mode"].values[0],
                        "model": df["model"].values[0], "epsilon": df["epsilon"].values[0], "accuracy": df["accuracy"].values[0]},
                       ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by = 'model').reset_index(drop=True))

elif E == 3:
    results_df = pd.DataFrame(columns=["data", "mode", "ncpc", "accuracy"])
    directory = 'log/E3'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                if data in path:
                    df = pd.read_csv(path, header=None)
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = new_header
                    results_df = results_df._append(
                        {"data": df["data"].values[0], "mode": df["mode"].values[0],
                         "ncpc": df["ncpc"].values[0], "accuracy": df["accuracy"].values[0]},
                        ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by='mode').reset_index(drop=True))

else:
    results_df = pd.DataFrame(columns=["data", "mode", "nc", "accuracy"])
    directory = 'log/E4'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                if data in path:
                    df = pd.read_csv(path, header=None)
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = new_header
                    results_df = results_df._append(
                        {"data": df["data"].values[0], "mode": df["mode"].values[0],
                         "nc": df["num_client"].values[0], "accuracy": df["accuracy"].values[0]},
                        ignore_index=True)
    print(f'==> The results for {data} in E{E} is:')
    print(results_df.sort_values(by='mode').reset_index(drop=True))






    # if os.path.exists(f'log/E{E}/{DATA_NAME}_{NUM_CLIENTS}_{NUM_CLASES_PER_CLIENT}_{MODE}_{MODEL}_{target_epsilon}.csv') == True: