import itertools
import os
import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt
import tensorflow as tf
from conda import iteritems
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA
from tensorflow.python.keras.utils.np_utils import to_categorical
from Autoencoder import load_model, convert_to_integer, calculate_error, add_noise
from Constraints import *
from Delenox_Config import config
from NeatGenerator import novelty_search, generate_lattice, NeatGenerator
from Visualization import get_color_map, expressive_graph


def load_training_set(label):
    return [list(
        np.load("Delenox_Experiment_Data/{}/Phase{}/Training_Set.npz".format(label, i),
                allow_pickle=True)['arr_0'])[-1000:] for i in range(10)]


def load_seed_set():
    return list(np.load("Delenox_Experiment_Data/Seed/Initial_Training_Set.npy",
                        allow_pickle=True))


def load_populations(label):
    return [[list(
        np.load("Delenox_Experiment_Data/{}/Phase{}/Population_{}.npz".format(label, j, i),
                allow_pickle=True)['arr_0'].item().values()) for i in range(10)] for j in range(10)]


def load_metric(label, metric):
    try:
        return np.load("Delenox_Experiment_Data/{}/Phase{}/Metrics.npy".format(label, 9),
                       allow_pickle=True).item()[metric]
    except FileNotFoundError:
        return np.load("Delenox_Experiment_Data/{}/Phase{}/Metrics.npz".format(label, 9),
                       allow_pickle=True)['arr_0'].item()[metric]


def load_metrics(labels, metric):
    return [np.load("Delenox_Experiment_Data/{}/Phase{}/Metrics.npy".format(directory, 9),
                    allow_pickle=True)[metric] for directory in labels]


def load_autoencoder(label, phase):
    try:
        encoder = load_model("Delenox_Experiment_Data/{}/Phase{}/encoder".format(label, phase))
        decoder = load_model("Delenox_Experiment_Data/{}/Phase{}/decoder".format(label, phase))
    except FileNotFoundError:
        print("File not found")
        encoder = load_model(
            "Delenox_Experiment_Data/seed/encoder".format(label, 0))
        decoder = load_model(
            "Delenox_Experiment_Data/seed/decoder".format(label, 0))
    return encoder, decoder


def vector_novelty(vector1, population):
    entropies = []
    for neighbour in population:
        entropies.append(np.linalg.norm(vector1 - neighbour))
    return np.mean(np.sort(entropies))


def vector_entropy(vector1, population):
    entropies = []
    for neighbour in population:
        entropies.append(entropy(vector1, neighbour))
    return np.mean(np.sort(entropies))


def novelty(vector, compressed_population):
    distances = []
    for neighbour in compressed_population:
        distance = 0
        for element in range(len(neighbour)):
            distance += np.square(vector[element] - neighbour[element])
        distances.append(np.sqrt(distance))
    distances = np.sort(distances)[1:]
    return np.round(np.average(distances), 2)


def lattice_diversity(experiment, args=None):
    experiment_population = load_populations(experiment)
    experiment_diversity = []
    for phase in range(len(experiment_population)):
        phase_diversities = []
        model = load_autoencoder(experiment, 9)[0]

        for population in range(len(experiment_population[phase])):
            print("Starting Experiment {} - Phase {} - Population {}".format(experiment, phase, population))
            # lattices = [softmax(np.asarray(lattice, dtype='float')).ravel() for lattice in experiment_population[phase][population]]
            lattices = [model.predict(lattice[None])[0] for lattice in experiment_population[phase][population]]
            results = [pool.apply_async(novelty, (vector, lattices)) for vector in lattices]
            diversity = np.mean([result.get() for result in results])
            phase_diversities.append(diversity)
        experiment_diversity.append(phase_diversities)
    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
    diversity_dict.update({experiment: [means, ci]})
    return means, ci


def matrix_set(experiments):

    results = {label: {} for label in experiments}

    for experiment in experiments:
        encoder, decoder = load_autoencoder(experiment, 9)
        experiment_result = {label: [] for label in experiments}

        for target in experiments:
            pops = load_populations(target)[-1]
            reconstruction_error = []

            for pop in pops:
                re = []

                vectors = [encoder.predict(lattice[None])[0] for lattice in pop]
                results = [pool.apply_async(novelty, (vector, vectors)) for vector in vectors]
                """for lattice in pop:
                    compressed = encoder.predict(lattice[None])[0]
                    reconstructed = decoder.predict(compressed[None])[0]
                    re.append(calculate_error(lattice, reconstructed))"""
                reconstruction_error.append(np.mean(re))

            experiment_result[target] += [np.mean(reconstruction_error), confidence_interval(reconstruction_error, 1.96)]
            print(experiment_result)
        results.update({experiment: experiment_result})
    print(results)
    np.save("./Results/Reconstruction_Matrix.npy", results)






def generate_seed():
    for pop in range(10):
        with open("Delenox_Experiment_Data/Seed/Neat_Population_{:d}.pkl".format(pop), "rb") as file:
            generator = pickle.load(file)

        jobs = []
        lattices = []

        for genome_id, genome in list(iteritems(generator.population.population)):
            jobs.append(pool.apply_async(generate_lattice, (genome, config, False, None)))

        for job in jobs:
            result = job.get()
            if result[2]:
                lattices.append(result[0])

        np.save("Delenox_Experiment_Data/Seed/Neat_Population_{:d}.npy".format(pop), lattices)


def diversity_from_seed(experiment, args=None):
    experiment_population = load_populations(experiment)
    experiment_diversity = []
    for phase in range(len(experiment_population)):
        phase_diversities = []

        target = [np.load("./Delenox_Experiment_Data/Seed/Neat_Population_{}.npy".format(pop), allow_pickle=True) for pop in range(10)]
        for population in range(len(experiment_population[phase])):
            targets = [softmax(np.asarray(lattice, dtype='float')).ravel() for lattice in target[population]]
            print("Starting Experiment {} - Phase {} - Population {}".format(experiment, phase, population))
            lattices = [softmax(np.asarray(lattice, dtype='float')).ravel() for lattice in
                        experiment_population[phase][population]]
            results = [pool.apply_async(vector_entropy, (lattice, targets)) for lattice in lattices]

            phase_diversities.append(np.mean([result.get() for result in results]))
        experiment_diversity.append(phase_diversities)
    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
    diversity_dict.update({experiment: [means, ci]})
    return means, ci


def test_population(experiments):
    """

    :param experiments:
    :return:
    """
    test_pop = []
    test_pop += list(np.load("Real-World Datasets/Ahousev5_Buildings_Fixed.npy", allow_pickle=True))
    test_pop += list(np.load("Real-World Datasets/Ahousev5_Buildings_Varied.npy", allow_pickle=True))

    encoded_pop = []
    for individual in test_pop:
        encoded_pop.append(to_categorical(individual, num_classes=5))

    """for experiment in experiments:
        populations = load_training_set(experiment)
        for population in populations:
            for building in random.sample(population, 30):
                test_pop.append(building)"""
    return encoded_pop


def reconstruction_accuracy(experiment, args):
    means = []
    cis = []

    for phase in range(10):
        print("Loading Autoencoder from Phase {}".format(phase))
        encoder, decoder = load_autoencoder(experiment, phase)
        errors = []
        for lattice in args[0]:
            if experiment[-3:] == 'DAE':
                compressed = encoder.predict(add_noise(lattice)[None])[0]
            else:
                compressed = encoder.predict(lattice[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]
            errors.append(calculate_error(lattice, reconstructed))

        if experiment == "Static AE":
            means = np.asarray([np.mean(errors) for _ in range(10)])
            cis = np.asarray([np.std(errors) / np.sqrt(len(errors)) * 1.96 for _ in range(10)])
            diversity_dict.update({experiment: [means, cis]})
            return means, cis
        else:
            means.append(np.mean(errors))
            cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)

    diversity_dict.update({experiment: [np.asarray(means), np.asarray(cis)]})
    return np.asarray(means), np.asarray(cis)


def pca_population(experiments):
    pca = PCA(n_components=2)
    try:
        pca_pop = np.load("Results/PCA.npy", allow_pickle=True)
    except:
        print("Loading experiment training sets and flattening them into 1D arrays...")
        pca_pop = [[[convert_to_integer(lattice).ravel() for lattice in load_seed_set()]]]
        pca_pop += [
            [[convert_to_integer(lattice).ravel() for lattice in phase] for phase in load_training_set(experiment)] for
            experiment in experiments]
        np.save("Results/PCA.npy", np.asarray(pca_pop))
    pca.fit(list(flatten(list(flatten(pca_pop)))))
    return pca, pca_pop


def scatter_plots(pca_pops):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharex=True, sharey=True)
    fig.suptitle("PCA Scatter Plots - 10$^{th}$ Iteration")
    pca_labels = labels

    print(np.asarray(pca_pops[1][0][-1]).shape)

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


def pca_graph(experiment, args=None, shareAxes=True):
    print("PCA - {}".format(experiment))
    experiment_population = args[0][1][args[1] + 1]
    diversity = []
    for training_set in range(len(experiment_population)):
        principal_components = args[0][0].transform(experiment_population[training_set])
        diversities = []
        for pc in principal_components:
            distances = []
            for other in principal_components:
                distances.append(np.linalg.norm(pc - other))
            distances = np.sort(distances)[1:]
            diversities.append(np.mean(distances[:15]))
        diversity.append(diversities)

    return np.asarray([np.mean(diversity1) for diversity1 in diversity]), np.asarray(
        [np.std(diversity1) / np.sqrt(len(diversity1)) * 1.96 for diversity1 in diversity])


def neat_metric(experiment, metric):
    metric = np.asarray(load_metric(experiment, metric[0]))
    if metric.shape == (10, 1000):
        metric = np.stack(metric, axis=1)
    generations = range(len(metric))
    mean = np.mean(metric[generations], axis=1)
    ci = np.std(metric[generations], axis=1) / np.sqrt(10) * 1.96
    return mean, ci


def grid_plot(experiments, function, title, dict, args=None, shareAxes=True, normalize=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6.5), sharex=shareAxes, sharey=shareAxes)

    try:
        [baseline_means, baseline_ci] = dict['Static AE']
        print("Static AE data found!")
    except:
        print("No data found, calculating new points")
        baseline_means, baseline_ci = function(experiments[0], (args, 0))

    counter = 0

    if normalize:
        baseline_means /= np.max(baseline_means, axis=1)
        baseline_ci /= np.max(baseline_ci, axis=1)

    for experiment in [(1,), (2, 3), (4, 5), (6, 7)]:
        axis = axes[locations[counter][0]][locations[counter][1]]
        axis.errorbar(x=range(len(baseline_means)), y=baseline_means, label=labels[0], color=colors[0])
        axis.fill_between(x=range(len(baseline_means)), y1=baseline_means + baseline_ci,
                          y2=baseline_means - baseline_ci, color=colors[0], alpha=0.1)
        axis.set_title(labels[experiment[0]][:-3] + " Autoencoders", fontsize=12)
        for sub in range(len(experiment)):

            if experiment[sub] != 1:
                axis.set_yticks(range(20, 43, 4))
                axis.set_ylim([20, 42])

            try:
                [means, ci] = dict[labels[experiment[sub]]]
                print("{} data found!".format(labels[experiment[sub]]))
            except:
                print("No data found for {}, calculating new points".format(labels[experiment[sub]]))
                means, ci = function(experiments[experiment[sub]], (args, experiment[sub]))

            if normalize:
                means /= np.max(means)
                ci /= np.max(ci)

            axis.errorbar(x=range(len(means)), y=means, label=ae_label[sub], color=colors[sub + 1])
            axis.fill_between(x=range(len(means)), y1=means + ci, y2=means - ci, color=colors[sub + 1], alpha=0.1)
        axis.grid()
        counter += 1
        handles, legendlabels = axis.get_legend_handles_labels()

    fig.legend(handles=handles, labels=legendlabels, fontsize=12, loc='lower center', ncol=6, )
    plt.setp(axes[-1, :], xlabel='Phase')
    plt.setp(axes[:, 0], ylabel=title)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig("../{}.png".format(title))
    fig.show()


def draw_lines_fig(fig):
    line = plt.Line2D((0.12, 0.12), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((0.91, 0.91), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.275, .275), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.4325, .4325), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.595, .595), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.7575, .7575), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    return fig


def novelty_spectrum(labels):
    xlabels = ['Most\nNovel', 'Upper\nQuartile', 'Median\nNovel', 'Lower\nQuartile', 'Least\nNovel']
    for experiment in labels:
        print("Starting Experiment {}".format(experiment))

        for i in range(10):
            fig = draw_lines_fig(plt.figure(figsize=(12, 12)))
            fig.suptitle("Range of Generated Content - {}".format(experiment), fontsize=18)

            phases = load_populations(experiment)

            for phase in range(1, len(phases), 2):
                print("Starting Phase {}".format(phase))
                encoder, _ = load_autoencoder(experiment, phase)

                pop = phases[phase][i]

                fitness = {}
                original = {}
                counter = 0

                compressed = {}
                for lattice in pop:
                    original.update({counter: convert_to_integer(lattice)})
                    if experiment[-3:] == 'DAE':
                        compressed.update({counter: encoder.predict(add_noise(lattice)[None])[0]})
                    else:
                        compressed.update({counter: encoder.predict(lattice[None])[0]})
                    counter += 1
                jobs = []
                for key in compressed.keys():
                    parameters = (key, compressed, [])
                    jobs.append(pool.apply_async(novelty_search, parameters))
                for job, genome_id in zip(jobs, compressed.keys()):
                    fitness.update({genome_id: job.get()})

                sorted_keys = [k for k, _ in sorted(fitness.items(), key=lambda item: item[1])]

                sorted_lattices = [original[key] for key in sorted_keys]
                np.save("{}-{}-{}.npy".format(experiment, i, phase), sorted_lattices)

                for number, plot in enumerate(np.linspace(len(sorted_keys) - 1, 0, 5, dtype=int)):
                    ax = fig.add_subplot(novelty_spectrum_subplots[int(phase / 2)][number][0],
                                         novelty_spectrum_subplots[int(phase / 2)][number][1],
                                         novelty_spectrum_subplots[int(phase / 2)][number][2], projection='3d')

                    ax.voxels(original[sorted_keys[plot]], edgecolor="k",
                              facecolors=get_color_map(original[sorted_keys[plot]], 'blue'))
                    ax.set_axis_off()
                    if phase == 1:
                        ax.text(-37, 0, -5, s=xlabels[number], fontsize=15)
                    if number == 4:
                        ax.text(5, 3, -40, s='Phase {}'.format(phase + 1), fontsize=15)
            fig.show()


def symmetry(lattice, h_bound, v_bound, d_bound):
    symmetry = 0
    symmetry += height_symmetry(lattice, h_bound, v_bound, d_bound)
    symmetry += width_symmetry(lattice, h_bound, v_bound, d_bound)
    symmetry += depth_symmetry(lattice, h_bound, v_bound, d_bound)
    return symmetry / 12000


def surface_ratio(lattice, h_bound, v_bound, d_bound):
    height = v_bound[1]
    depth = (d_bound[1] - d_bound[0])
    width = (h_bound[1] - h_bound[0])

    roof_count = 0
    walls = 0
    floor_count = 0
    interior_count = 0
    total_count = 0
    for (x, y, z) in value_range:
        if lattice[x][y][z] == 0:
            continue
        total_count += 1
        if lattice[x][y][z] == 1:
            interior_count += 1
        elif lattice[x][y][z] == 2:
            walls += 1
        elif lattice[x][y][z] == 3:
            floor_count += 1
        elif lattice[x][y][z] == 4:
            roof_count += 1

    voxel_count = (walls + roof_count + floor_count)
    bounding_box_area = 2 * (depth*width + depth*height + width*height)
    try:
        return voxel_count / bounding_box_area
    except ZeroDivisionError:
        print(voxel_count)
        print(bounding_box_area)
        return 0


def expressive(phase):
    surface_areas = []
    stabilities = []
    x_symmetry = []
    y_symmetry = []
    z_symmetry = []

    converted = [convert_to_integer(lattice) for lattice in phase]
    for lattice in converted:
        horizontal_bounds, depth_bounds, vertical_bounds = bounding_box(lattice)
        x_symmetry.append(width_symmetry(lattice, horizontal_bounds, depth_bounds, vertical_bounds))
        y_symmetry.append(height_symmetry(lattice, horizontal_bounds, depth_bounds, vertical_bounds))
        z_symmetry.append(depth_symmetry(lattice, horizontal_bounds, depth_bounds, vertical_bounds))
        surface_areas.append(surface_ratio(lattice, horizontal_bounds, vertical_bounds, depth_bounds))
        stabilities.append(stability(lattice)[1])
    return {"Surface Area": surface_areas, "Stability": stabilities, "X-Symmetry": x_symmetry, "Y-Symmetry": y_symmetry, "Z-Symmetry": z_symmetry}


def confidence_interval(values, confidence):
    return np.std(values) / np.sqrt(len(values)) * confidence


def AVG_Properties(label, pops):
    experiment_results = {"Surface Area": [], "Stability": [], "X-Symmetry": [], "Y-Symmetry": [], "Z-Symmetry": []}
    for phase in range(len(pops)):
        properties = expressive(pops[phase])
        for property in properties.keys():
            experiment_results[property] += properties[property]
    for property in experiment_results.keys():
        experiment_results[property] = {"Mean": np.round(np.mean(experiment_results[property]), 2),
                                        "CI": np.round(confidence_interval(experiment_results[property], 1.96), 2)}
    print({label: experiment_results})
    return {label: experiment_results}


def confusion_matrix(experiments):
    results = {experiment: [] for experiment in ["Seed"] + experiments}
    seed_pops = [np.load("./Delenox_Experiment_Data/Seed/Neat_Population_{}.npy".format(pop), allow_pickle=True) for pop in range(10)]
    results.update(AVG_Properties("Seed", seed_pops))

    for experiment in experiments:
        pops = load_populations(experiment)[-1]
        results.update(AVG_Properties(experiment, pops))
    np.save("./Results/AVG_Results.npy", results)


def expressive_analysis(experiments, xlabel, ylabel, dict=None):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharex=True, sharey=True)
    plt.setp(axes[-1, :], xlabel=xlabel)
    plt.setp(axes[:, 0], ylabel=ylabel)

    try:
        metric1 = dict['Seed'][xlabel]
        metric2 = dict['Seed'][ylabel]
        print("Seed data found!")
    except:
        print("No seed data found, calculating new points")
        seed = load_seed_set()
        metric1, metric2 = expressive(seed, xlabel, ylabel)
        diversity_dict["Seed"].update({xlabel: metric1, ylabel: metric2})

    expressive_graph(fig, axes[0, 0], x=metric1, y=metric2, title="Seed", x_label=xlabel, y_label=ylabel)
    counter = 1
    locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    for experiment in experiments:
        try:
            metric1 = dict[experiment][xlabel]
            metric2 = dict[experiment][ylabel]
            print("{} data found!".format(experiment))
        except:
            print("{} data not found! Calculating new points..".format(experiment))
            phase = load_training_set(experiment)[-1]
            metric1, metric2 = expressive(phase, xlabel, ylabel)
            diversity_dict[experiment].update({xlabel: metric1, ylabel: metric2})
        expressive_graph(fig, axes[locs[counter][0]][locs[counter][1]], x=metric1, y=metric2, title=experiment,
                         x_label=xlabel, y_label=ylabel)
        counter += 1

    fig.subplots_adjust(bottom=0.15)
    fig.tight_layout()
    fig.savefig("../Expressive-{}vs{}.png".format(xlabel, ylabel))
    fig.show()


def novelty_critic(experiment, args=None):
    human_population = np.load("Real-World Datasets/Ahousev5_Buildings_Varied.npy", allow_pickle=True)
    targets = [to_categorical(lattice, num_classes=5) for lattice in human_population]
    # experiment_population = load_training_set(experiment)
    experiment_diversity = []

    for phase in range(10):
        print("Evaluating Human Buildings: {} - Phase {}".format(experiment, phase))
        encoder, decoder = load_autoencoder(experiment, phase)
        #  = [encoder.predict(lattice[None])[0] for lattice in experiment_population[phase]]
        human_vectors = [encoder.predict(lattice[None])[0] for lattice in targets]
        results = [pool.apply_async(vector_novelty, (lattice, human_vectors)) for lattice in human_vectors]
        experiment_diversity.append([result.get() for result in results])
        print(experiment_diversity[-1])
    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(len(experiment_diversity[0])) * 1.96
    diversity_dict.update({experiment: [means, ci]})
    return means, ci


def compare_plot(experiments, function, title, dict, args=None):
    plt.figure()
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.grid()
    counter = 0

    for experiment in experiments:
        try:
            [means, ci] = dict[experiment]
            print("{} data found!".format(experiment))
        except:
            print("No data found for {}, calculating new points".format(experiment))
            means, ci = function(experiment, (args, experiment))

        if function == neat_metric:
            ticks = list(range(0, len(means), 100)) + [len(means) - 1]
        else:
            ticks = range(len(means))

        plt.errorbar(x=ticks, y=means[ticks], label=experiment, color=colors[counter], marker=markers[counter], alpha=0.7, linestyle=linestyles[counter])
        plt.fill_between(x=ticks, y1=means[ticks] + ci[ticks], y2=means[ticks] - ci[ticks], color=colors[counter], alpha=0.15)
        counter += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.legend(fontsize=12, loc='upper center', ncol=3, bbox_to_anchor=[0.5, 1.21])
    plt.show()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', labelsize=12)

    flatten = itertools.chain.from_iterable

    locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pca_locs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    novelty_spectrum_subplots = [[(5, 5, index * 5 + offset) for index in range(5)] for offset in range(1, 6)]
    ae_label = ['Vanilla AE', 'Denoising AE']

    # pool = Pool(12)
    labels = ["Static AE", "Random AE", "Full History AE", "Latest Set AE", "Novelty Archive AE"]
    colors = ['black', '#d63333', '#3359d6', '#3398d6', '#662dc2']
    markers = ['s', 'v', 'D', '^', 'o']
    linestyles = ['solid', 'dashed', 'dashed', 'dashed', 'dashed']
    # keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]
    keys = ['Species Count']
    # generate_seed()
    # confusion_matrix(labels)#
    # matrix_set(labels)
    # diversity_dict = {}
    # compare_plot(labels, lattice_diversity, "Novelty", dict=diversity_dict)
    # np.save("./Results/Novelty.npy", diversity_dict)

    """dict = np.load("./Results/Reconstruction_Matrix.npy", allow_pickle=True).item()
    for label in labels:
        model_acc = []
        pop_acc = []

        for target in labels:
            model_acc.append(dict[label][target][0])
            pop_acc.append(dict[target][label][0])

        print("{} - Mean Population Error: {}% ± {}%, Mean Model Error: {}% ± {}%".format(label, np.round(np.mean(pop_acc), 2), np.round(confidence_interval(pop_acc, 1.96), 2), np.round(np.mean(model_acc), 2), np.round(confidence_interval(model_acc, 1.96)), 2))
    """

    for key in keys:
        compare_plot(labels, neat_metric, key, args=key, dict={})

    # diversity_dict = np.load("Expressive.npy", allow_pickle=True).item()
    # expressive_analysis(labels, "Instability", "Symmetry", diversity_dict)
    # expressive_analysis(labels, "Surface Area", "Symmetry", diversity_dict)
    # expressive_analysis(labels, "Surface Area", "Instability", diversity_dict)
    # np.save("Expressive.npy", diversity_dict)

    # novelty_spectrum(labels)

    # diversity_dict = np.load("Results/Critic_Results_Intra.npy", allow_pickle=True).item()
    # grid_plot(labels, novelty_critic, "Assigned Novelty", shareAxes=True, dict=diversity_dict)
    # np.save("Results/Critic_Results_Intra.npy", diversity_dict)

    # diversity_dict = np.load("Reconstruction.npy", allow_pickle=True).item()
    # grid_plot(labels, reconstruction_accuracy, "Reconstruction Error", dict=diversity_dict, args=test_population(labels), shareAxes=False)
    # np.save("Reconstruction.npy", diversity_dict)

    # grid_plot(labels, pca_graph, "Eucl. Diversity", args=pca_population(labels), shareAxes=True, dict={})

    # diversity_dict = np.load("./Results/Reconstruction.npy", allow_pickle=True).item()
    # compare_plot(labels, reconstruction_accuracy, "Reconstruction Error", dict=diversity_dict, args=test_population(labels))

    # diversity_dict = np.load("./Results/Diversities_No_K.npy", allow_pickle=True).item()
    # diversity_dict = np.load("./Results/Human_Diversity_No_K.npy", allow_pickle=True).item()
    # compare_plot(labels, diversity_from_humans, "Voxel KL-Diversity", dict=diversity_dict)
    # grid_plot(labels, lattice_diversity, "Diversity", shareAxes=True, dict=diversity_dict)
    # np.save("Divergence_from_seed.npy", diversity_dict)

    # diversity_dict = np.load("Human_Diversity_No_K.npy", allow_pickle=True).item()
    # grid_plot(labels, diversity_from_humans, "Entropy", shareAxes=True, dict=diversity_dict)
    # np.save("Human_Diversity_No_K.npy", diversity_dict)

    # pool.close()
    # pool.join()
