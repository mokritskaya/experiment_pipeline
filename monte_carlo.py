import pandas as pd
import numpy as np
from metric_builder import Metric
from stattests import TTestFromStats, MannWhitneyTest, PropZTest, calculate_statistics, calculate_linearization
from scipy.stats import binom
import warnings
warnings.filterwarnings('ignore')

import config as cfg


def calculate_tpr(df, sim_num):
    names = {
        "tpr": sum(df['pvalue'] < 0.05) / sim_num
    }
    return pd.Series(names)


def monte_carlo(df_orig, metric_items):
    
    ttest = TTestFromStats()
    utest = MannWhitneyTest()
    proptest = PropZTest()

    alpha = 0.05 # уровень значимости
    simulations = 100 # количество симуляций
    lifts = np.arange(1, 1.1, 0.01) # последовательность шагов по эффекту

    sim_res = pd.DataFrame()

    for lift in lifts: 
        df = df_orig.copy()

        for _ in range(0, simulations):

            # Рандомное присвоение групп A/B
            df[cfg.VARIANT_COL] = binom.rvs(1, 0.5, size=len(df.index)) 

            df.loc[df[cfg.VARIANT_COL] == 1, 'num'] = df.loc[df[cfg.VARIANT_COL] == 1, 'num'] * lift

            df_ = calculate_linearization(df)
            stats = calculate_statistics(df_, metric_items.type)

            if metric_items.estimator == 't_test':
                criteria_res = ttest(stats)
            elif metric_items.estimator == 'mann_whitney':
                criteria_res = utest(df_)
            elif metric_items.estimator == 'prop_test':
                criteria_res = proptest(df_)
            else:
                criteria_res = ttest(stats)
            
            sim_res = sim_res.append({"metric_name": metric_items.name, "lift": lift, "pvalue": criteria_res.pvalue}, ignore_index=True)
            
    res = sim_res.groupby(["metric_name","lift"]).apply(calculate_tpr, sim_num=simulations) \
        .pivot_table(index=['metric_name'], columns=["lift"], values='tpr') \
        .add_prefix('tpr_lift_') \
        .reset_index()
    
    res = res.drop(['metric_name'], axis=1)
    
    return res