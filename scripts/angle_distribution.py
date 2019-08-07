import matplotlib.pyplot as plt
import seaborn as sns
from boxcar_load_model import visualize_prediction_batch
import numpy as np

def plot_angle_distribution(angle_idx):
    sns.set_context("talk")
    fig = plt.figure(1)
    fig.clf()
    ax1 = fig.add_subplot(111)

    [ground_truth, predicts] = visualize_prediction_batch(getting_angle=True, angle_idx=angle_idx)


    # colors = ["b", "g", "r", "k"]
    # for angle, l, c in zip([alphas, betas, gammas, predicts], ["alpha", "beta", "gamma", "predicts"],
    #                        colors):
    #     sns.distplot(angle, label=l, ax=ax1, kde=False, bins=60, color=c)
    colors = ["r", "g"]
    for angle, l, c in zip([ground_truth, predicts], ["ground_truth", "predicts"],
                        colors):
        sns.distplot(angle, label=l, ax=ax1, kde=False, bins=60, color=c)
    ax1.set_xlabel("angle (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./histogram_resnet50_3angles{}_60bins.png'.format(angle_idx))

def main():
    angle_number = 3
    for i in range(angle_number):
        plot_angle_distribution(i)

if __name__ == '__main__':
    main()