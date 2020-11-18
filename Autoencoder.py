from multiprocessing import Pool
import numpy as np
from keras.layers import Dense, Flatten, Reshape, Input, Conv2D, Conv2DTranspose, Conv3D, MaxPooling2D, UpSampling2D, \
                         MaxPooling3D, Conv3DTranspose, UpSampling3D
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import model_from_json
from Delenox_Config import lattice_dimensions, batch_size, no_epochs, thread_count, value_range
from Visualization import auto_encoder_plot, visualize_training


def update_auto_encoder(ae, population):
    population_noisy = add_noise_parallel(population)
    training_noisy, test_noisy, training, test = train_test_split(population_noisy, population, test_size=0.2,
                                                                  random_state=29)
    history = ae.fit(x=training_noisy, y=training, epochs=no_epochs,
                     batch_size=batch_size, validation_data=(test_noisy, test), shuffle=True)
    visualize_training(history)
    return ae


def create_auto_encoder(compressed_length, model_type, population=None):
    """
    Function to create and train a de-noising auto-encoder to compress 3D lattices
    into a 1D latent vector representation.

    :param compressed_length: length of the compressed representation
    :param model_type: type of auto-encoder to create (2D vs 3D)
    :param population: population of lattices to train the model on (given when performing Delenox)
    :return: the generated encoder and decoder models
    """

    # input_buildings, lattices_noisy = create_population_lattices(config)

    ae, encoder_model, decoder_model = model_type(compressed_length)

    encoder_model.summary()
    decoder_model.summary()

    # Compiling the AE and fitting it using the noisy population as input and the original population as the target
    ae.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'binary_accuracy'])

    # If the function is given a population of lattices, create a set of noisy variants and partition into train-test.
    if population is not None:
        population_noisy = add_noise_parallel(population)
        training_noisy, test_noisy, training, test = train_test_split(population_noisy, population, test_size=0.2, random_state=29)
    else:
        test = np.load("Training_Materials.npy")
        training = np.load("Test_Materials.npy")
        training_noisy = np.load("Test_Materials_Noisy.npy")
        test_noisy = np.load("Training_Materials_noisy.npy")

    history = ae.fit(x=training_noisy, y=training, epochs=no_epochs,
                     batch_size=batch_size, validation_data=(test_noisy, test), shuffle=True)
    visualize_training(history)

    save_model(encoder_model, "material_encoder_256")
    save_model(decoder_model, "material_decoder_256")

    return ae, encoder_model, decoder_model


def auto_encoder_3d(compressed_length):
    """
    Function to create the structure of a 3D auto-encoder, using a series of convolution and sampling layers.
    This model is specifically designed to take a lattice with resolution (20x20x20).

    :param compressed_length: desired latent vector size.
    :return: auto-encoder model, as well as the encoder and decoder separately.
    """

    # Constructing the model for the encoder
    encoder_model = Sequential(name="Encoder_"+str(compressed_length))
    encoder_model.add(Conv3D(compressed_length / 4, kernel_size=(3, 3, 3), activation='relu', input_shape=(20, 20, 20, 5)))
    encoder_model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    encoder_model.add(Conv3D(compressed_length / 2, kernel_size=(2, 2, 2), activation='relu'))
    encoder_model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    encoder_model.add(Conv3D(compressed_length, kernel_size=(3, 3, 3), activation='relu'))
    encoder_model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    encoder_model.add(Flatten())

    # Constructing the model for the decoder
    decoder_model = Sequential(name="Decoder_"+str(compressed_length))
    decoder_model.add(Reshape((1, 1, 1, compressed_length)))
    decoder_model.add(UpSampling3D(size=(2, 2, 2)))
    decoder_model.add(Conv3DTranspose(compressed_length / 2, kernel_size=(3, 3, 3), activation='relu'))
    decoder_model.add(UpSampling3D(size=(2, 2, 2)))
    decoder_model.add(Conv3DTranspose(compressed_length / 4, kernel_size=(2, 2, 2), activation='relu'))
    decoder_model.add(UpSampling3D(size=(2, 2, 2)))
    decoder_model.add(Conv3DTranspose(5, kernel_size=(3, 3, 3), activation='sigmoid'))

    # Combining the two models into the auto-encoder model
    ae_input = Input((20, 20, 20, 5))
    ae_encoder_output = encoder_model(ae_input)
    ae_decoder_output = decoder_model(ae_encoder_output)
    ae = Model(ae_input, ae_decoder_output)

    return ae, encoder_model, decoder_model


def auto_encoder_2d(compressed_length):
    """
    Function to create the structure of a 2D auto-encoder, using a series of convolution and sampling layers.
    This model is specifically designed to take a lattice with resolution (20x20x20).

    :param compressed_length: desired latent vector size.
    :return: auto-encoder model, as well as the encoder and decoder separately.
    """

    # Constructing the model for the encoder
    encoder_model = Sequential(name="Encoder_"+str(compressed_length))
    encoder_model.add(Conv2D(compressed_length / 4, kernel_size=(3, 3), activation='relu',
                             kernel_initializer='he_uniform', input_shape=lattice_dimensions))
    encoder_model.add(MaxPooling2D((2, 2)))
    encoder_model.add(Conv2D(compressed_length / 2, kernel_size=(3, 3), activation='relu'))
    encoder_model.add(MaxPooling2D((2, 2)))
    encoder_model.add(Conv2D(compressed_length, kernel_size=(3, 3), activation='relu'))
    encoder_model.add(Flatten())

    # Constructing the model for the decoder
    decoder_model = Sequential(name="Decoder_"+str(compressed_length))
    decoder_model.add(Reshape((1, 1, compressed_length)))
    decoder_model.add(Conv2DTranspose(compressed_length / 2, kernel_size=(3, 3), activation='relu'))
    decoder_model.add(UpSampling2D((2, 2)))
    decoder_model.add(Conv2DTranspose(compressed_length / 4, kernel_size=(3, 3), activation='relu'))
    decoder_model.add(UpSampling2D((2, 2)))
    decoder_model.add(Conv2DTranspose(lattice_dimensions[2], kernel_size=(3, 3), activation='relu'))
    decoder_model.add(Conv2DTranspose(lattice_dimensions[2], kernel_size=(3, 3), activation='sigmoid'))
    decoder_model.add(Reshape(lattice_dimensions))

    # Combining the two models into the auto-encoder model
    ae_input = Input(lattice_dimensions)
    ae_encoder_output = encoder_model(ae_input)
    ae_decoder_output = decoder_model(ae_encoder_output)
    ae = Model(ae_input, ae_decoder_output)

    return ae, encoder_model, decoder_model


def auto_encoder_2d_scalable(compressed_length):
    """
    Function to create the structure of a 2D auto-encoder, using a series of convolution and sampling layers.
    This model uses a dense layer at the inner most ends of the encoder and decoder to ensure it can scale
    across multiple lattice resolutions.

    :param compressed_length: desired latent vector size.
    :return: auto-encoder model, as well as the encoder and decoder separately.
    """

    # Constructing the model for the encoder
    encoder_model = Sequential(name="Encoder_"+str(compressed_length))
    encoder_model.add(Conv2D(compressed_length / 4, kernel_size=(3, 3), activation='relu',
                             kernel_initializer='he_uniform', input_shape=lattice_dimensions))
    encoder_model.add(MaxPooling2D((2, 2)))
    encoder_model.add(Conv2D(compressed_length / 2, kernel_size=(3, 3), activation='relu'))
    encoder_model.add(MaxPooling2D((2, 2)))
    encoder_model.add(Conv2D(compressed_length, kernel_size=(3, 3), activation='relu'))
    encoder_model.add(Flatten())
    encoder_model.add(Dense(compressed_length))

    final_conv_shape = int(((((lattice_dimensions[0] - 2) / 2) - 2) / 2) - 2)

    # Constructing the model for the decoder
    decoder_model = Sequential(name="Decoder_"+str(compressed_length))
    decoder_model.add(Dense(compressed_length * final_conv_shape ** 2))
    decoder_model.add(Reshape((final_conv_shape, final_conv_shape, compressed_length)))
    decoder_model.add(Conv2DTranspose(compressed_length / 2, kernel_size=(3, 3), activation='relu'))
    decoder_model.add(UpSampling2D((2, 2)))
    decoder_model.add(Conv2DTranspose(compressed_length / 4, kernel_size=(3, 3), activation='relu'))
    decoder_model.add(UpSampling2D((2, 2)))
    decoder_model.add(Conv2DTranspose(lattice_dimensions[2], kernel_size=(3, 3), activation='relu'))
    decoder_model.add(Conv2DTranspose(lattice_dimensions[2], kernel_size=(3, 3), activation='sigmoid'))
    decoder_model.add(Reshape(lattice_dimensions))

    # Combining the two models into the auto-encoder model
    ae_input = Input(lattice_dimensions)
    ae_encoder_output = encoder_model(ae_input)
    ae_decoder_output = decoder_model(ae_encoder_output)
    ae = Model(ae_input, ae_decoder_output)

    return ae, encoder_model, decoder_model


def save_model(model, name):
    """
    Save any given model structure and its weights to disk.
    :param model: model to be saved.
    :param name: desired file name.
    """
    model_json = model.to_json()
    with open("./Autoencoder_Models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Autoencoder_Models/" + name  + ".h5")
    print("Saved " + name + " model to disk.")


def load_model(name):
    """
    Load a model (structure and weights) from the model directory into memory for use.
    :param name: file name of the desired model
    :return: loaded model.
    """
    json_file = open("Autoencoder_Models/" + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Autoencoder_Models/" + name + ".h5")
    print("Loaded model " + name + " from disk.")
    return loaded_model


def compress_lattices(lattices, encoder):
    """
    Compress a population of lattices with the given encoder model.

    :param lattices: population of lattices to be compressed.
    :param encoder: model to compress the lattices.
    :return: population of compressed lattices (1D latent vector).
    """
    compressed = []
    for examples in range(len(lattices)):
        example = np.round(np.asarray(lattices[examples]))
        compressed.append(encoder.predict(example[None])[0])
    return np.asarray(compressed)


def add_noise_parallel(lattices, name=None):
    """
    Multi-process approach to adding noise to a population of lattices, outputting the
    results to a file.

    :param lattices: population of lattices to be noised.
    :param name: name of the output file containing the noisy population.
    :return: the noisy population of lattices.
    """
    pool = Pool(thread_count)
    jobs = []
    noisy_lattices = []
    for lattice in lattices:
        jobs.append(pool.apply_async(add_noise, (lattice, )))
    for job in jobs:
        noisy_lattices.append(job.get())
    pool.close()
    pool.join()
    if name is not None:
        np.save(name + "_Noisy.npy", np.asarray(noisy_lattices))
    return np.asarray(noisy_lattices)


def add_noise(lattice):
    """
    Function to add noise to any given lattice using a set noise method.
    Negative noise = removed voxels.
    Additive noise = add voxels.
    Bit-flip = flip the presence of a voxel at a coordinate.

    :param lattice: lattice to be noised.
    :return: noisy lattice.
    """
    noisy_lattice = lattice.copy()
    for (x, y, z) in value_range:
        if np.random.random() < 0.025:
            noisy_lattice[x][y][z] = 0
    return noisy_lattice


def test_accuracy(encoder, decoder, test):
    """
    Function to test the accuracy of an auto-encoder model through a given test population.

    :param encoder: encoder that will compress the lattices.
    :param decoder: decoder that will reconstruct the lattices from the compressed representation.
    :param test: population of test lattices to calculate its accuracy.
    :return: mean accuracy observed and its standard deviation.
    """
    error = []
    for one_hot_lattice in test:
        lattice = convert_to_integer(one_hot_lattice)
        compressed = encoder.predict(one_hot_lattice[None])[0]
        reconstructed = decoder.predict(compressed[None])[0]
        integer_reconstruct = convert_to_integer(reconstructed)
        error.append(calculate_error(lattice, integer_reconstruct))
        auto_encoder_plot(lattice, compressed, integer_reconstruct, error[-1])
    print("MEAN:", np.mean(error), " - ST-DEV:", np.std(error))


def calculate_error(original, reconstruction):
    """
    Function to calculate the error rate between an original lattice and its reconstructed counterpart.
    Error is calculated categorically, i.e: the value observed in the original must be identical in the
    reconstructed version.

    :param original: original lattice generated through the NEAT module.
    :param reconstruction: reconstructed version generated through the auto-encoder model.
    :return: categorical error.
    """
    error = 0
    for (x, y, z) in value_range:
        if original[x][y][z] != np.round(reconstruction[x][y][z]):
            error += 1
    return round(error / (lattice_dimensions[0] ** 3) * 100, 2)


def convert_to_integer(lattice):
    """
    Convert material lattice from one-hot representation to integer encoding representation.

    :param lattice: lattice of one-hot material vectors.
    :return: lattice of integer material codes.
    """
    integer_reconstruct = np.zeros(lattice_dimensions)
    for channel in range(20):
        for row in range(20):
            integer_reconstruct[channel][row] = np.argmax(lattice[channel][row], axis=1)
    return integer_reconstruct
