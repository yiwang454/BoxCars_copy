import matplotlib.pyplot as plt
import seaborn as sns
from boxcar_load_model import visualize_prediction_batch
import numpy as np
sns.set_context("talk")
fig = plt.figure(1)
fig.clf()
ax1 = fig.add_subplot(111)

[alphas, betas, gammas, predicts] = visualize_prediction_batch(getting_angle=True)


# colors = ["b", "g", "r", "k"]
# for angle, l, c in zip([alphas, betas, gammas, predicts], ["alpha", "beta", "gamma", "predicts"],
#                        colors):
#     sns.distplot(angle, label=l, ax=ax1, kde=False, bins=60, color=c)
colors = ["r", "k", 'g']
for angle, l, c in zip([alphas, predicts, -np.array(predicts)], ["alpha", "predicts", "nve predicts"],
                       colors):
    sns.distplot(angle, label=l, ax=ax1, kde=False, bins=60, color=c)
ax1.set_xlabel("angle (rad)")
plt.legend()
plt.tight_layout()
plt.savefig('./histogram.png')