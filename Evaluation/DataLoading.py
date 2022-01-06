from Evaluation.EvalutationConfig import PCA, load_model, flatten, phases_to_evaluate
from Generator.NeatGenerator import *
from Generator.Delenox_Config import *


def load_training_set(label):
    return [list(np.load("../Generator/Results/{}/Phase{}/Training_Set.npz".format(label, i),
            allow_pickle=True)['arr_0'])[-1000:] for i in range(runs_per_phase)]


def load_seed_pops():
    return [np.load("../Generator/Results/Seed/Neat_Population_{}.npy".format(pop), allow_pickle=True) for pop in
            range(runs_per_phase)]


def load_seed_set():
    return list(np.load("../Generator/Results/Seed/Initial_Training_Set.npy", allow_pickle=True))


def load_populations(label):
    return [[list(np.load("../Generator/Results/{}/Phase{}/Population_{}.npz".format(label, j, i),
            allow_pickle=True)['arr_0'].item().values()) for i in range(runs_per_phase)] for j in range(phases_to_evaluate)]


def load_metric(label, metric):
    try:
        return np.load("../Generator/Results/{}/Phase{}/Metrics.npy".format(label, phases_to_evaluate - 1),
                       allow_pickle=True).item()[metric]
    except FileNotFoundError:
        return np.load("../Generator/Results/{}/Phase{}/Metrics.npz".format(label, phases_to_evaluate - 1),
                       allow_pickle=True)['arr_0'].item()[metric]


def load_autoencoder(label, phase):
    try:
        encoder = load_model("../Generator/Results/{}/Phase{}/encoder".format(label, phase))
        decoder = load_model("../Generator/Results/{}/Phase{}/decoder".format(label, phase))
    except FileNotFoundError:
        print("File not found")
        encoder = load_model(
            "../Generator/Results/seed/encoder".format(label, 0))
        decoder = load_model(
            "../Generator/Results/seed/decoder".format(label, 0))
    return encoder, decoder


def medieval_population(categorical):
    test_pop = list(np.load("Other Datasets/Ahousev5_Buildings_Fixed.npy", allow_pickle=True))
    test_pop += list(np.load("Other Datasets/Ahousev5_Buildings_Varied.npy", allow_pickle=True))
    if categorical:
        encoded_pop = [to_categorical(individual, num_classes=5) for individual in test_pop]
    else:
        encoded_pop = test_pop
    return encoded_pop


def block_buildings():
    return np.load("./Other Datasets/Block_Buildings.npy", allow_pickle=True)


def pca_population(experiments):
    pca = PCA(n_components=2)
    try:
        pca_pop = np.load("Results/PCA.npy", allow_pickle=True)
    except FileNotFoundError:
        print("Loading experiment training sets and flattening them into 1D arrays...")
        pca_pop = [[[convert_to_integer(lattice).ravel() for lattice in load_seed_set()]]]
        pca_pop += [
            [[convert_to_integer(lattice).ravel() for lattice in phase] for phase in load_training_set(experiment)] for
            experiment in experiments]
        np.save("Results/PCA.npy", np.asarray(pca_pop))
    pca.fit(list(flatten(list(flatten(pca_pop)))))
    return pca, pca_pop


def generate_seed(pool):
    for pop in range(10):
        with open("../Generator/Results/Seed/Neat_Population_{:d}.pkl".format(pop), "rb") as file:
            generator = pickle.load(file)
        jobs = []
        lattices = []
        for genome_id, genome in list(iteritems(generator.population.population)):
            jobs.append(pool.apply_async(generate_lattice, (genome, config, False, None)))
        for job in jobs:
            result = job.get()
            if result[2]:
                lattices.append(result[0])
        np.save("../Generator/Results/Seed/Neat_Population_{:d}.npy".format(pop), lattices)

