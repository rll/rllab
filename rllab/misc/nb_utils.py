import os.path as osp
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import joblib
from glob import glob
import os


def plot_experiments(name_or_patterns, legend=False, post_processing=None, key='AverageReturn'):
    if not isinstance(name_or_patterns, (list, tuple)):
        name_or_patterns = [name_or_patterns]
    data_folder = osp.abspath(osp.join(osp.dirname(__file__), '../../data'))
    files = []
    for name_or_pattern in name_or_patterns:
        matched_files = glob(osp.join(data_folder, name_or_pattern))
        files += matched_files
    files = sorted(files)
    print('plotting the following experiments:')
    for f in files:
        print(f)
    plots = []
    legends = []
    for f in files:
        exp_name = osp.basename(f)
        returns = []
        with open(osp.join(f, 'progress.csv'), 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[key]:
                    returns.append(float(row[key]))
        returns = np.array(returns)
        if post_processing:
            returns = post_processing(returns)
        plots.append(plt.plot(returns)[0])
        legends.append(exp_name)
    if legend:
        plt.legend(plots, legends)


class Experiment(object):
    def __init__(self, progress, params, pkl_data=None):
        self.progress = progress
        self.params = params
        self.pkl_data = pkl_data
        self.flat_params = self._flatten_params(params)
        self.name = params["exp_name"]

    def _flatten_params(self, params, depth=2):
        flat_params = dict()
        for k, v in params.items():
            if isinstance(v, dict) and depth != 0:
                for subk, subv in self._flatten_params(v, depth=depth - 1).items():
                    if subk == "_name":
                        flat_params[k] = subv
                    else:
                        flat_params[k + "_" + subk] = subv
            else:
                flat_params[k] = v
        return flat_params


def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class ExperimentDatabase(object):
    def __init__(self, data_folder, names_or_patterns='*'):
        self._load_experiments(data_folder, names_or_patterns)

    def _read_data(self, progress_file):
        entries = dict()
        with open(progress_file, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for k, v in row.items():
                    if k not in entries:
                        entries[k] = []
                    entries[k].append(float(v))
        entries = dict([(k, np.array(v)) for k, v in entries.items()])
        return entries

    def _read_params(self, params_file):
        with open(params_file, "r") as f:
            return json.loads(f.read())

    def _load_experiments(self, data_folder, name_or_patterns):
        if not isinstance(name_or_patterns, (list, tuple)):
            name_or_patterns = [name_or_patterns]
        files = []
        for name_or_pattern in name_or_patterns:
            matched_files = glob(
                osp.join(data_folder, name_or_pattern))  # golb gives a list of all files satisfying pattern
            files += matched_files  # this will include twice the same file if it satisfies 2 patterns
        experiments = []
        progress_f = None
        params_f = None
        pkl_data = None
        for f in files:
            if os.path.isdir(f):
                try:
                    progress = self._read_data(osp.join(f, "progress.csv"))
                    params = self._read_params(osp.join(f, "params.json"))
                    params["exp_name"] = osp.basename(f)
                    if os.path.isfile(osp.join(f, "params.pkl")):
                        pkl_data = joblib.load(osp.join(f, "params.pkl"))
                        experiments.append(Experiment(progress, params, pkl_data))
                    else:
                        experiments.append(Experiment(progress, params))
                except Exception as e:
                    print(e)
            elif 'progress.csv' in f:  # in case you're giving as datafolder the dir that contains the files!
                progress_f = self._read_data(f)
            elif 'params.json' in f:
                params_f = self._read_params(f)
            elif 'params.pkl' in f:
                print('about to load', f)
                pkl_data = joblib.load(f)
        if params_f and progress_f:
            if pkl_data:
                experiments.append(Experiment(progress_f, params_f, pkl_data))
            else:
                experiments.append(Experiment(progress_f, params_f))

        self._experiments = experiments

    def plot_experiments(self, key=None, legend=None, color_key=None, filter_exp=None, **kwargs):
        experiments = list(self.filter_experiments(**kwargs))
        if filter_exp:
            experiments = list(filter(filter_exp, experiments))
        plots = []
        legends = []
        color_pool = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        color_map = dict()
        if color_key is not None:
            exp_color_keys = uniq([exp.flat_params.get(
                color_key, None) for exp in experiments])
            if len(exp_color_keys) > len(color_pool):
                raise NotImplementedError
            for exp_color_key, color in zip(exp_color_keys, color_pool):
                print("%s: %s" % (str(exp_color_key), color))
            color_map = dict(list(zip(exp_color_keys, color_pool)))
        used_legends = []
        legend_list = []

        for exp in experiments:
            exp_color_key = None
            if color_key is not None:
                exp_color_key = exp.flat_params.get(color_key, None)
                exp_color = color_map.get(exp_color_key, None)
            else:
                exp_color = None
            plots.append(plt.plot(exp.progress.get(
                key, [0]), color=exp_color)[0])
            if legend is not None:
                legends.append(exp.flat_params[legend])
            elif exp_color_key is not None and exp_color_key not in used_legends:
                used_legends.append(exp_color_key)
                legend_list.append(plots[-1])

        if len(legends) > 0:
            plt.legend(plots, legends)
        elif len(legend_list) > 0:
            plt.legend(legend_list, used_legends)

    def filter_experiments(self, **kwargs):
        for exp in self._experiments:
            exp_params = exp.flat_params
            match = True
            for key, val in kwargs.items():
                if exp_params.get(key, None) != val:
                    match = False
                    break
            if match:
                yield exp

    def unique(self, param_key):
        return uniq([exp.flat_params[param_key] for exp in self._experiments if param_key in exp.flat_params])
