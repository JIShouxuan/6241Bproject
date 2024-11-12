# Add invariant test into the path
import sys, os

# add invariant test module
sys.path.append(os.path.abspath(os.path.join('..', 'invariant-policy-learning-main/simulation_new')))

from tqdm import tqdm
from compute_experiment import compute_experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count


## Conduct multiple experiments with multiprocessing
if __name__ == '__main__':
    repeats = 1000

    # Multiprocess
    pool = Pool(cpu_count() - 2)
    res = np.array(
        list(tqdm(pool.imap_unordered(compute_experiment, range(repeats)), total=repeats)))
    pool.close()

    temp_dict = {k: [val for dic in res for val in dic[k]] for k in res[0].keys()}
    df_regrets = pd.DataFrame(temp_dict)

    sns.set(font_scale=1.2, style='white', palette=sns.set_palette("tab10"))

    g = sns.relplot(
        data=df_regrets, x="Sample Size", y="Acceptance Rate",
        col="n_env", hue="Policy", kind="line", marker='o', markersize=6,
        height=3.25, aspect=1, alpha=.6
    )
    g.set(xscale="log")
    g.set_titles('#training envs: {col_name}')
    g.set_xlabels("Sample Size (Total)")

    leg = g._legend
    leg.set_bbox_to_anchor([1., 0.55])
    for ax in g.axes[0]:
        ax.axhline(0.95, ls='--', color='black', label='95% level', linewidth=0.85, alpha=0.7)
        plt.legend(bbox_to_anchor=(1.5, 1.2))
    plt.tight_layout()

    plt.show()

    #plt.savefig('results/invariant_test.pdf')
