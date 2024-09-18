import math
import operator
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

TIME_LIMIT = 1800  # Seconds
TIME_THRESHOLD = 120  # Seconds
TIME_DIFF_THRESHOLD = 0.01

MEMORY_LIMIT = 15_728.640  # MB
MEMORY_THRESHOLD = 1_000  # MB
MEMORY_DIFF_THRESHOLD = 0.01

CASE_IDS_COLS = ['model', 'query', 'category', 'subcategory']

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT_DIR = Path(__file__).parent.parent / 'output'

DEMO_CSVS = [
    ('Tapaal', DATA_DIR / 'demo_tapaal.csv'),
    ('Static', DATA_DIR / 'demo_static.csv'),
    ('Dynamic', DATA_DIR / 'demo_dynamic.csv'),
]
BASE_NAME = 'Tapaal'


def answers(df, file_postfix=''):
    rows = {
        'CTL ALL': df.category == 'CTL',
        'CTL Cardinality': (df.category == 'CTL') & (df.subcategory == 'Cardinality'),
        'CTL Fireability': (df.category == 'CTL') & (df.subcategory == 'Fireability'),
    }
    res = pd.DataFrame()
    for experiment in df.experiment.unique():
        res[experiment] = {row: len(df[filt & (df.experiment == experiment) & (df.satisfied != 'unknown')]) for row, filt in rows.items()}
    res['ANY'] = {row: df.assign(answered=df.satisfied != 'unknown')[filt].groupby(CASE_IDS_COLS).answered.any().sum() for row, filt in rows.items()}
    res['ALL'] = {row: df.assign(answered=df.satisfied != 'unknown')[filt].groupby(CASE_IDS_COLS).answered.all().sum() for row, filt in rows.items()}

    file = OUT_DIR / f'answers{file_postfix}.csv'
    res.to_csv(file, sep=';')
    print('▫ Created', file)


def unique_answers(df, file_postfix=''):
    def inner(idf):
        experiments = idf.experiment.unique().tolist()
        res = pd.DataFrame()
        for exp_wrt in experiments:
            exp_wrt_ans = idf[idf.experiment == exp_wrt].set_index(CASE_IDS_COLS).satisfied != 'unknown'
            values = dict()
            for exp in experiments:
                exp_ans = idf[idf.experiment == exp].set_index(CASE_IDS_COLS).satisfied != 'unknown'
                common_ans = exp_ans & exp_wrt_ans
                values[exp] = sum(exp_ans & ~common_ans)
            res[exp_wrt] = values
        return res

    rows = {
        'CTL ALL': df.category == 'CTL',
        'CTL Cardinality': (df.category == 'CTL') & (df.subcategory == 'Cardinality'),
        'CTL Fireability': (df.category == 'CTL') & (df.subcategory == 'Fireability'),
    }
    res = pd.concat([inner(df[filt]).reset_index(names=['experiment\\wrt']).assign(category=row) for row, filt in rows.items()]).reset_index(drop=True)

    file = OUT_DIR / f'answers_unique{file_postfix}.csv'
    res.to_csv(file, sep=';', index=False)
    print('▫ Created', file)


def ratio_box_plot(df1, df2, parameter, none_value, limit, file_postfix=''):
    df1_name = df1["experiment"].iloc[0]
    df2_name = df2["experiment"].iloc[0]
    x_name = 'queries, sorted by ratio'
    y_name = f'{parameter} ratio'

    df1 = df1.set_index(CASE_IDS_COLS)
    df2 = df2.set_index(CASE_IDS_COLS)
    _either = (df1[parameter] != none_value) | (df2[parameter] != none_value)
    df1 = df1[_either]
    df2 = df2[_either]

    # Ratios where unanswered us infinity
    df1_inf = df1.replace({parameter: none_value}, float("inf"))
    df2_inf = df2.replace({parameter: none_value}, float("inf"))

    series_inf = df1_inf[parameter] / (df1_inf[parameter] + df2_inf[parameter])
    series_inf = series_inf.sort_values().reset_index(drop=True)
    series_inf = series_inf.replace(float("nan"), 1.0)
    x_offset = len(series_inf[series_inf <= 0.5])
    series_inf.index -= x_offset
    df_inf = series_inf.reset_index(name=y_name).rename(columns={'index': x_name})
    df_inf['value of unanswered'] = '$\\infty$'

    # Ratios where unanswered us limit
    df1_lim = df1.replace({parameter: none_value}, limit)
    df2_lim = df2.replace({parameter: none_value}, limit)

    series_lim = df1_lim[parameter] / (df1_lim[parameter] + df2_lim[parameter])
    series_lim = series_lim.sort_values().reset_index(drop=True)
    series_lim.index -= x_offset
    df_lim = series_lim.reset_index(name=y_name).rename(columns={'index': x_name})
    df_lim['value of unanswered'] = f'{parameter} limit'

    # Combine
    df = pd.concat([df_inf, df_lim]).reset_index(drop=True)

    # Display
    sns.set_theme(style='whitegrid', palette=sns.color_palette('tab10'))
    plt.tight_layout()
    g = sns.relplot(data=df, kind='line', x=x_name, y=y_name, hue='value of unanswered', style='value of unanswered', dashes={'$\\infty$': '', 'time limit': (5, 2), 'memory limit': (1, 1)}, height=3.0, aspect=1.0)
    plt.axhline(y=0.5, color='k', linestyle=':')
    plt.axvline(x=0, color='k', linestyle=':')
    plt.text(-x_offset, 0.52, f'{-x_offset}')
    plt.text(len(series_inf)-x_offset, 0.52, f'{len(series_inf)-x_offset}', horizontalalignment='right')
    plt.text(5.0, 0.02, f'{df1_name}%')
    plt.text(-5.0, 0.98, f'{df2_name}%', horizontalalignment='right', verticalalignment='top')
    g.ax.set_yticks(np.arange(0, 1.01, 0.1))
    g.ax.set_xticks(np.arange(-46000, 46000, 10 ** math.floor(math.log10(len(series_inf)))))
    g.set(ylim=(-0.005, 1.005), xlim=(-x_offset, len(series_inf) - x_offset))
    g._legend.remove()
    #plt.legend(loc='lower right', title='Value of unanswered', framealpha=1.0)
    plt.tight_layout()

    # Save
    file = OUT_DIR / f'box_ratio{file_postfix}_{parameter}_{df1_name}-{df2_name}.png'
    plt.savefig(file)
    print('▫ Created', file)


def dashes_dict(df):
    experiments = df.experiment.unique()
    unused_dashes = [(2, 1), (1, 1, 1, 1, 3, 1)]
    res = {'Tapaal': '', 'Dynamic': (4, 1), 'Static': (1, 1), 'min(Static,Dynamic)': (4, 1, 1, 1)}
    if 'min(Static,Dynamic)' not in experiments:
        unused_dashes.append(res['min(Static,Dynamic)'])
        del res['min(Static,Dynamic)']
    if 'Static' not in experiments:
        unused_dashes.append(res['Static'])
        del res['Static']
    if 'Dynamic' not in experiments:
        unused_dashes.append(res['Dynamic'])
        del res['Dynamic']
    for e in df.experiment:
        if e not in res:
            res[e] = unused_dashes.pop()
    return res


def cactus(df, parameter, unit, query_prefix='', file_postfix=''):
    df = df[df['satisfied'] != 'unknown']

    xlabel = f'#{query_prefix}queries answered'
    df = df.groupby('experiment').apply(lambda x: x.sort_values(parameter).reset_index(drop=True))
    df = df.reset_index(level=1, names=[None, xlabel]).reset_index(drop=True)
    df[xlabel] += 1

    # Graph
    sns.set_theme(style='whitegrid', palette=sns.color_palette('tab10'))
    plt.tight_layout()
    ax = sns.relplot(data=df, kind='line', x=xlabel, y=parameter, hue='experiment', style='experiment', dashes=dashes_dict(df), height=3.0, aspect=1.0)
    ax.set(yscale='log')
    plt.tight_layout()
    ax.set(ylim=[max(df[parameter].min(), 0.01), df[parameter].max() * 1.03])
    ax.set(ylabel=f'{parameter} limit ({unit})', xlabel=xlabel)
    sns.move_legend(ax, loc='lower right', bbox_to_anchor=(0.95, 0.23), title=None, frameon=True)
    plt.tight_layout()

    # Save
    file = OUT_DIR / f'cactus{file_postfix}_{parameter}.png'
    plt.savefig(file)
    print('▫ Created', file)


def uniformity(dfs, group, filter=None, file_postfix=''):
    PERCENTAGE = 0.03  # 1/32th of queries in group
    DECIMALS = 1
    FACTORS = [2.0, 10.0, 100.0]
    FACTORS = list(map(lambda x: 1.0 / x, reversed(FACTORS))) + FACTORS

    N = 0
    dfs = dfs.copy()
    for i, (e, df) in enumerate(dfs):
        df = df[filter] if filter is not None else df
        N = df[group].nunique()
        df = df.replace({'time': -1}, float('inf'))
        df.time = np.true_divide(np.ceil(df.time * (10 ** DECIMALS)), 10 ** DECIMALS)
        df = df.set_index(CASE_IDS_COLS + ['family'])
        dfs[i] = (e, df)

    assert dfs[0][0] == BASE_NAME, f'The first dataframe is not {BASE_NAME}'
    df_base = dfs[0][1]
    dfs = dfs[1:]

    num_queries_per_group = df_base.groupby(group).time.count()

    res = pd.DataFrame()
    for (e, df) in dfs:
        values = dict()
        for factor in FACTORS:
            if factor <= 1.0:
                num_qs_slower_per_group = ((df.time * factor) > df_base.time).groupby(group).sum()
                values[factor] = ((num_qs_slower_per_group / num_queries_per_group) >= PERCENTAGE).sum()
            if factor >= 1.0:
                num_qs_faster_per_group = ((df.time * factor) < df_base.time).groupby(group).sum()
                values[factor] = ((num_qs_faster_per_group / num_queries_per_group) >= PERCENTAGE).sum()
            if factor == 100.0 and group == 'model':
                # # Find the number of families that the 2-orders-of-magnitude improvements are spread across
                # num_qs_faster_per_group = ((df.time * factor) < df_base.time).groupby(group).sum()
                # faster_models = num_qs_faster_per_group[num_qs_faster_per_group > 0].index.to_list()
                # families = df_base.reset_index()[df_base.reset_index().model.isin(faster_models)].groupby('family').model.any().sum()
                # print(e, families)
                pass

        res[e] = values

    file = OUT_DIR / f'uniformity{file_postfix}_N={N}.csv'
    res.to_csv(file, sep=';', index=False)
    print('▫ Created', file)


def main(named_csvs):
    dfs = [(e, pd.read_csv(csv, sep=';').assign(experiment=e)) for (e, csv) in named_csvs]
    for name, df in dfs:
        df.loc[df['satisfied'] == 'unknown', 'memory'] = float("nan")
        df['memory'] = df['memory'] / 1_000_000  # Recorded in bytes, convert to MB
    N = len(dfs[0][1])

    def is_diff(fractions, threshold):
        return (fractions >= (1.0 + threshold)) | (fractions <= (1.0 / (1.0 + threshold)))


    falses = pd.Series([False] * N)
    finished_by_some = reduce(operator.or_, [df['satisfied'] != 'unknown' for (_, df) in dfs], falses)
    time_hard_for_some = reduce(operator.or_, [df.replace({'time': -1}, TIME_LIMIT)['time'] >= TIME_THRESHOLD for (_, df) in dfs], falses)
    memory_hard_for_some = reduce(operator.or_, [df.replace({'memory': float("nan")}, MEMORY_LIMIT)['memory'] >= MEMORY_THRESHOLD for (_, df) in dfs], falses)
    time_diff = reduce(operator.or_, [is_diff(df.replace({'time': -1}, float("inf"))['time'] / dfs[0][1].replace({'time': -1}, float("inf"))['time'], TIME_DIFF_THRESHOLD) for (_, df) in dfs], falses)
    memory_diff = reduce(operator.or_, [is_diff(df.replace({'memory': float("nan")}, float("inf"))['memory'] / dfs[0][1].replace({'memory': float("nan")}, float("inf"))['memory'], MEMORY_DIFF_THRESHOLD) for (_, df) in dfs], falses)

    challenging = (time_diff & memory_diff) & (time_hard_for_some | memory_hard_for_some)

    chldf = pd.DataFrame(zip(challenging, dfs[0][1].model, dfs[0][1].model.str.split('-').str[0]), columns=['challenging', 'model', 'family'])
    challenging_model = pd.merge(dfs[0][1], chldf.groupby('model').challenging.any(), on='model').challenging
    ms_per_fam = chldf.groupby(['model', 'family']).first().groupby('family').count().challenging
    chl_per_fam = chldf.groupby(['model', 'family']).challenging.any().groupby('family').sum()
    # A family is challenging if the majority (>=0.5) of instances is challenging
    challenging_family = pd.merge(chldf, (chl_per_fam / ms_per_fam) >= 0.5, on='family').challenging_y

    for i, (e, df) in enumerate(dfs):
        dfs[i] = (e, df.assign(
            challenging=challenging,
            challenging_model=challenging_model,
            family=df.model.str.split('-').str[0],
            challenging_family=challenging_family,
        ))

    def min_df(e, df1, df2, parameter, none_value):
        df1_best_mask = df1.replace({parameter: none_value}, 1_000_000_000)[parameter] <= df2.replace({parameter: none_value}, 1_000_000_000)[parameter]
        _df = pd.concat([df1[df1_best_mask], df2[~df1_best_mask]])
        _df = _df.sort_index()
        return e, _df.assign(experiment=e)

    df = pd.concat([df for e, df in dfs], ignore_index=True)
    if len(dfs) > 2 and dfs[1][0] == 'Static' and dfs[2][0] == 'Dynamic':
        edf_min = min_df('min(Static,Dynamic)', dfs[1][1], dfs[2][1], 'time', -1)
        df_w_min = pd.concat([df for e, df in dfs + [edf_min]], ignore_index=True)
    else:
        edf_min = None
        df_w_min = df

    assert len(df[df.satisfied == 'error']) == 0, 'ERRORS DETECTED'
    print('✅ Dataset contains no errors')

    answers(df)
    answers(df[df.challenging], file_postfix='_challenging')
    unique_answers(df)
    cactus(df_w_min, 'time', 's')
    cactus(df, 'memory', 'MB')
    cactus(df_w_min[df_w_min.challenging], 'time', 's', query_prefix='challenging ', file_postfix='_challenging')
    cactus(df[df.challenging], 'memory', 'MB', query_prefix='challenging ', file_postfix='_challenging')
    uniformity(dfs + [edf_min] if edf_min is not None else dfs, 'model', finished_by_some, file_postfix='_all_models')
    uniformity(dfs + [edf_min] if edf_min is not None else dfs, 'family', finished_by_some, file_postfix='_all_families')
    uniformity(dfs + [edf_min] if edf_min is not None else dfs, 'model', finished_by_some & challenging_model, file_postfix='_challenging_models')
    uniformity(dfs + [edf_min] if edf_min is not None else dfs, 'family', finished_by_some & challenging_family, file_postfix='_challenging_families')
    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            (e1, df1), (e2, df2) = dfs[i], dfs[j]
            ratio_box_plot(df1[df1.challenging], df2[df2.challenging], 'time', -1, TIME_LIMIT, file_postfix='_challenging')
            ratio_box_plot(df1[df1.challenging], df2[df2.challenging], 'memory', float("nan"), MEMORY_LIMIT, file_postfix='_challenging')
    print('Done!')
    exit(0)


if __name__ == '__main__':
    """
    If no arguments are received, we use the demo csvs. Otherwise, we expect a series of arguments on the form
    'name=file' where 'name' is the name of the data located in 'file', specifically `data/'file'`. Additionally,
    there must be a least two arguments and the first name must be 'Tapaal'. Example:
    `./python graphs_and_tables.py Tapaal=ae_tapaal.csv Dynamic=ae_dynamic.csv`.
    """
    if len(sys.argv) <= 1:
        main(DEMO_CSVS)

    else:
        named_csvs = [(arg.split('=')[0], DATA_DIR / arg.split('=')[1]) for arg in sys.argv[1:]]
        if named_csvs[0][0] != 'Tapaal':
            print(f'First data file must be named \'Tapaal\' for the graphs to be generated correctly. Got \'{named_csvs[0][0]}\'')
            exit(1)

        main(named_csvs)
