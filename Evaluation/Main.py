import matplotlib

from Evaluation.DataLoading import *
from Evaluation.DiversityMeasures import diversity_from_target, diversity_correlation
from Evaluation.EvalutationConfig import *
from Evaluation.NeatMeasures import neat_metric
from Evaluation.QualitativeMeasures import surface_ratio, AVG_Properties, AVG_Plot, novelty_spectrum
from Evaluation.ReconstructionMeasures import reconstruction_accuracy


def matrix_set(experiments):
    results = {label: {} for label in experiments}
    for experiment in experiments:
        encoder, decoder = load_autoencoder(experiment, 9)
        experiment_result = {label: [] for label in experiments}
        for target in experiments:
            pops = load_populations(target)[-1]
            reconstruction_error = []
            for pop in pops:
                # vectors = [encoder.predict(lattice[None])[0] for lattice in pop]
                # results = [pool.apply_async(novelty, (vector, vectors)) for vector in vectors]
                for lattice in pop:
                    compressed = encoder.predict(lattice[None])[0]
                    reconstructed = decoder.predict(compressed[None])[0]
                    reconstruction_error.append(calculate_error(lattice, reconstructed))
            experiment_result[target] += reconstruction_error
        results.update({experiment: experiment_result})
    np.save("./Results/Reconstruction_Matrix.npy", results)


def scatter_plots(pca_pops):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    fig.suptitle("PCA Scatter Plots - 10$^{th}$ Iteration")
    pca_pop = pca_pops[0].transform(pca_pops[1][0][-1])
    axis = axes[pca_locs[0][0]][pca_locs[0][1]]
    axis.set_title("Seed", fontsize=12)
    axis.scatter([item[0] for item in pca_pop], [item[1] for item in pca_pop], s=7, alpha=0.5)

    for i in range(8):
        axis = axes[pca_locs[i + 1][0]][pca_locs[i + 1][1]]
        axis.set_title("{}".format(labels[i]), fontsize=12)
        pca_pop = pca_pops[0].transform(pca_pops[1][i + 1][-1])
        axis.scatter([item[0] for item in pca_pop], [item[1] for item in pca_pop], s=7, alpha=0.5)
        plt.setp(axes[-1, :], xlabel='PC1')
        plt.setp(axes[:, 0], ylabel='PC2')
    fig.tight_layout()
    fig.savefig("../PCA.png")
    fig.show()


def confusion_matrix(experiments):
    results = {experiment: [] for experiment in ["Seed"] + experiments}
    seed_pops = [np.load("./Results/Seed/Neat_Population_{}.npy".format(pop), allow_pickle=True) for pop
                 in range(10)]
    results.update(AVG_Properties("Seed", seed_pops))

    for experiment in experiments:
        pops = load_populations(experiment)[-1]
        results.update(AVG_Properties(experiment, pops))
    np.save("./Results/AVG_Results.npy", results)


def compare_plot(experiments, function, title, filename="", args=None):

    matplotlib.rcParams.update({'font.size': 14})

    if args is None:
        figure_name = "./Figures/{}.png".format(function.__name__)
    elif type(args) == str:
        figure_name = "./Figures/{}-{}.png".format(function.__name__, args)
    else:
        figure_name = "./Figures/{}.png".format(filename)

    save = True
    results_dict = {}

    try:
        results_dict = np.load("./Results/{}.npy".format(filename), allow_pickle=True).item()
    except FileNotFoundError:
        print("No file found for {}".format(filename))
        save = True

    # del(results_dict['Static AE'])

    plt.figure()
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel(title, fontsize=16)
    plt.grid()

    counter = 0

    for experiment in experiments:
        try:
            [means, ci] = results_dict[experiment]
            print("{} - mean: {} - CI - {}".format(experiment, means, ci))
            ticks = range(len(means))
        except KeyError:
            print("No data found for {}, calculating new points".format(experiment))
            ticks, means, ci = function(experiment, pool, (args, experiment))
            results_dict.update({experiment: [means, ci]})

        plt.errorbar(x=ticks, y=means, label=experiment, color=colors[counter], marker=markers[counter], alpha=0.7,
                     linestyle=linestyles[counter])
        plt.fill_between(x=ticks, y1=means + ci, y2=means - ci, color=colors[counter], alpha=0.15)
        counter += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=[0.5, 1.235])
    plt.savefig(figure_name)

    if save:
        np.save("./Results/{}".format(filename), results_dict)


if __name__ == '__main__':

    pool = Pool(12)
    # matrix_set(labels)

    # for key in neat_keys:
        # compare_plot(labels, neat_metric, key, args=key, save=True)

    """AVG_Properties(labels)
    for key in AVG_keys:
        compare_plot(labels, AVG_Plot, key, args=key, filename="AVG_Properties", save=True)"""

    # compare_plot(labels, reconstruction_accuracy, "Reconstruction Error", args=None, filename="Reco_Training_Sets", save=True)

    # diversity_correlation(labels, pool, None)

    # compare_plot(labels, diversity_from_target, "Voxel KL-Diversity", args=None, filename="KL_Populations")
    # compare_plot(labels, diversity_from_target, "Voxel KL-Diversity", args=load_seed_pops(), filename="KL_Seed")
    # compare_plot(labels, diversity_from_target, "Voxel KL-Diversity", args=medieval_population(True), filename="KL_Medieval")
    compare_plot(labels, diversity_from_target, "Voxel KL-Diversity", args=block_buildings(), filename="KL_Block")

    # compare_plot(labels, reconstruction_accuracy, "Reconstruction Error", args=None, filename="Reco_Pops")
    compare_plot(labels, reconstruction_accuracy, "Reconstruction Error", args=block_buildings(), filename="Reco_Medieval",)
    # compare_plot(labels, reconstruction_accuracy, "Reconstruction Error", args=medieval_population(True), filename="Reco_Medieval", save=True)
    # novelty_spectrum(labels, pool)

    """matrix_set(labels)
    dict = np.load("./Results/Novelty_Matrix.npy", allow_pickle=True).item()
    for label in labels:
        model_acc = []
        pop_acc = []
        for target in labels:
            model_acc.append(dict[label][target][0])
            pop_acc.append(dict[target][label][0])
        print("{} - Mean Population Error: {}% ± {}%, Mean Model Error: {}% ± {}%".format(label, np.round(np.mean(pop_acc), 2), np.round(confidence_interval(pop_acc, 1.96), 2), np.round(np.mean(model_acc), 2), np.round(confidence_interval(model_acc, 1.96)), 2))"""

    pool.close()
    pool.join()
