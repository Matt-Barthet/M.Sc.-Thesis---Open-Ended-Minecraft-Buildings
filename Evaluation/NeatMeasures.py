from Evaluation.DataLoading import load_metric
from Evaluation.EvalutationConfig import *


def neat_metric(experiment, metric):
    metric = np.asarray(load_metric(experiment, metric[0]))
    if metric.shape == (10, phases_to_evaluate * 100):
        metric = np.stack(metric, axis=1)
    generations = range(len(metric))
    mean = np.mean(metric[generations], axis=1)
    ci = np.std(metric[generations], axis=1) / np.sqrt(10) * 1.96
    ticks = list(range(0, len(mean), 100)) + [len(mean) - 1]
    return ticks, mean[ticks], ci[ticks]
