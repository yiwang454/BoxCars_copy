import matplotlib.pyplot as plt
import seaborn as sns
from boxcar_load_model import visualize_prediction_batch, get_length
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
    plt.savefig('./check_resnet50_3angles{}_60bins.png'.format(angle_idx))
    

def plot_length_distribution():
    sns.set_context("talk")
    fig = plt.figure(1)
    fig.clf()

    [length, width, height] = [get_length(dim_idx=i) for i in range(3)]

    colors = ["r", "g", "b"]

    for i in range(2):
        ax1 = fig.add_subplot(2, 2, i+1)

        for dim, l, c in zip([length, width, height], ["length", "width", "height"],
                            colors):
            sns.distplot(dim, label=l, ax=ax1, kde=False, bins=60 // (2**i), color=c)
        ax1.set_xlabel("length (pixel)")
        
        ax2 = fig.add_subplot(2, 2, i+3)
        sns.jointplot(length, width, kind="hex", bins=60 // (2**i), color="#4CB391")
        ax2.set_xlabel("length")
        ax2.set_ylabel("width")

        
    plt.legend()
    plt.tight_layout()

    plt.savefig('./test__3dimemsions_60_or_30bins.png')
    fig.clf()

    ax = fig.add_subplot(111)

    plt.scatter(length, width, marker='.', color='crimson', alpha=0.05)
    ax.set_xlabel("length (pixel)")
    ax.set_ylabel("width (pixel)")

    plt.savefig("./scatter.png")
    fig.clf()


def main():
    # angle_number = 3
    # for i in range(angle_number):
    #     plot_angle_distribution(i)
    plot_length_distribution()

if __name__ == '__main__':
    main()