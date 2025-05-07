import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

'''
Usage:
loss_type must be "Training" or "Validation"
epsilon should be epsilon
the csv files should come in the order normal_sgd.csv normal_adamw.csv custom_dp_sgd.csv opacus_dp_sgd.csv

the columsn of the csv should be "epoch,train_loss,val_loss,epsilon"

example usage: python CumulativePlot.py Training 1 Base_Implementation/Transformer_model_loss_data/Base_Model_SGD.csv Base_Implementation/Transformer_model_loss_data/Base_Model_AdamW.csv Custom_Implementation/runs/eps1_trial1.csv Professional_Implementation/data/training_log_eps_1.csv
'''

def main():
    # Expect exactly 4 file paths after the script name
    if len(sys.argv) != 7:
        print("Usage: python plot_losses.py loss_type epsilon normal_sgd.csv normal_adamw.csv custom_dp_sgd.csv opacus_dp_sgd.csv")
        sys.exit(1)

    loss_type = sys.argv[1]
    epsilon = sys.argv[2]

    script_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(os.path.dirname(script_path))

    file_paths = sys.argv[3:]
    file_paths = [os.path.join(file_dir, fp) for fp in file_paths]

    labels = ["Normal SGD", "Normal AdamW", "Custom DP-SGD", "Opacus DP-SGD"]

    data = "train_loss" if loss_type == "Training" else "val_loss"

    # Plot each file
    for fp, label in zip(file_paths, labels):
        df = pd.read_csv(fp)
        # Assumes columns named exactly "epoch" and "train_loss"
        plt.plot(df['epoch'], df[data], label=label)

    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_type} Loss')
    plt.title(f'{loss_type} Loss as a function of Epochs for Epsilon {epsilon}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    current_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    plt.savefig(f"Epsilon_{epsilon}_{loss_type}_Cumulative_Comparison_{current_time}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
