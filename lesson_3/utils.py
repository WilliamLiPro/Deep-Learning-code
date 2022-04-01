import os
import pandas as pd
import time
import numpy as np
import torch


def cat(*xs):
    return torch.cat(xs)


def to_numpy(x):
    return x.detach().cpu().numpy()


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}


class StatsLogger():
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)


class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered))


def parameter_total(model):
    f = 0
    for p in model.parameters():
        f += p.numel()
    return f


def save_csv(summary, filename: str):
    # whether path exist
    print('Save summary to: ' + filename + ' ...')
    (filepath, temp_filename) = os.path.split(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # save the lists to certain text
    if '.csv' not in filename:
        filename += '.csv'
    pd.DataFrame(summary).to_csv(filename)
    print('Save finish')
