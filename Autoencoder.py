# Keras imports for Auto-Encoder implementation
from multiprocessing import Pool
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Input, Conv2D, Conv2DTranspose, Conv3D, MaxPooling2D, UpSampling2D, \
                         MaxPooling3D, Conv3DTranspose, UpSampling3D
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils.data_utils import Sequence
from Delenox_Config import lattice_dimensions, batch_size, no_epochs, thread_count
import numpy as np
import matplotlib.pyplot as plt
from Visualization import auto_encoder_plot, voxel_plot, vizualize
from Utility import add_noise, calculate_error


class DataGenerator(Sequence):
    """

    """

    def __init__(self, batch_size, name, directory):
        """

        :param batch_size:
        :param name:
        """
        self.batch_size = batch_size
        self.directory = directory
        self.name = name

    def __len__(self):
        """

        :return:
        """
        return self.batch_size

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        # print("\n[" + self.name + "]: Getting Batch #" + str(idx))

        offset = 0
        if self.name == "Validation_Generator":
            offset = 3000

        with open(self.directory + "Normal/Lattice_Batch_" + str(idx + offset) + ".npy", 'rb') as file:
            lattice_batch = np.load(file)

        with open(self.directory + "Noisy/Lattice_Batch_" + str(idx + offset) + ".npy", 'rb') as file:
            noisy_batch = np.load(file)

        # voxel_plot(lattice_batch[0], "Original")
        # voxel_plot(noisy_batch[0], "Noisy")

        return np.stack(noisy_batch), np.stack(lattice_batch)


def create_auto_encoder(compressed_length, function):
    """

    :param compressed_length:
    :param function:
    :return:
    """

    # input_buildings, lattices_noisy = create_population_lattices(config)

    ae, encoder_model, decoder_model = function(compressed_length)

    encoder_model.summary()
    decoder_model.summary()

    # training, test, training_noisy, test_noisy = train_test_split(input_buildings, lattices_noisy, test_size=0.2)

    # Compiling the AE and fitting it using the noisy population as input and the original population as the target
    ae.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    # my_training_batch_generator = DataGenerator(2000, "Training_Generator", "Lattice_Dataset/Training/")
    # my_validation_batch_generator = DataGenerator(400, "Validation_Generator", "Lattice_Dataset/Validation/")

    # batch_count = int(np.floor(best_fit_count / batch_size))

    test = np.load("Test_Carved.npy")
    training = np.load("Training_Carved.npy")
    training_noisy = np.load("Training_Carved_Noisy.npy")
    test_noisy = np.load("Test_Carved_Noisy.npy")

    history = ae.fit(x=training_noisy, y=training, epochs=no_epochs,
                     batch_size=batch_size, validation_data=(test_noisy, test), shuffle=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    """for passes in range(100):

        for batch_number in range(batch_count):
            batch = input_buildings[batch_number * batch_size: (batch_number + 1) * batch_size]
            noisy_batch = lattices_noisy[batch_number * batch_size: (batch_number + 1) * batch_size]

            with open("Lattice_Dataset/Normal/Lattice_Batch_" + str(batch_count * passes + batch_number) + '.npy', 'wb') as file:
                np.save(file, np.asarray(batch))
            with open("Lattice_Dataset/Noisy/Lattice_Batch_" + str(batch_count * passes + batch_number) + '.npy', 'wb') as file:
                np.save(file, np.asarray(noisy_batch))"""

    #

    """ae.fit(x=my_training_batch_generator,
           validation_data=my_validation_batch_generator,
           verbose=2,
           epochs=no_epochs,
           shuffle=True,
           )"""

    save_model(encoder_model, "encoder", compressed_length)
    save_model(decoder_model, "decoder", compressed_length)

    return ae, encoder_model, decoder_model


def auto_encoder_3d(compressed_length):
    """

    :param compressed_length:
    :return:
    """

    # Constructing the model for the encoder
    encoder_model = Sequential(name="Encoder_"+str(compressed_length))
    encoder_model.add(Conv3D(compressed_length / 4, kernel_size=(3, 3, 3), activation='relu', input_shape=(20, 20, 20, 1)))
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
    decoder_model.add(Conv3DTranspose(1, kernel_size=(3, 3, 3), activation='sigmoid'))

    # Combining the two models into the auto-encoder model
    ae_input = Input((20, 20, 20, 1))
    ae_encoder_output = encoder_model(ae_input)
    ae_decoder_output = decoder_model(ae_encoder_output)
    ae = Model(ae_input, ae_decoder_output)

    return ae, encoder_model, decoder_model


def auto_encoder_2d(compressed_length):
    """

    :param compressed_length:
    :return:
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

    :param compressed_length:
    :return:
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


def save_model(model, name, compressed_length):
    model_json = model.to_json()
    with open("./Autoencoder_Models/" + name + "_" + str(compressed_length) + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Autoencoder_Models/" + name + "_" + str(compressed_length) + ".h5")
    print("Saved " + name + "_" + str(compressed_length) + " model to disk.")


def load_model(name):
    json_file = open("Autoencoder_Models/" + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Autoencoder_Models/" + name + ".h5")
    print("Loaded model " + name + " from disk.")
    return loaded_model


def add_noise_parallel(lattices, name):
    pool = Pool(thread_count)
    jobs = []
    noisy_lattices = []
    for lattice in lattices:
        jobs.append(pool.apply_async(add_noise, (lattice, )))
    for job in jobs:
        noisy_lattices.append(job.get())
    pool.close()
    pool.join()

    np.save(name + "_Noisy.npy", np.asarray(noisy_lattices))
    return np.asarray(noisy_lattices)


def compress_lattices(lattices, encoder):
    compressed = []
    for examples in range(len(lattices)):
        example = np.round(np.asarray(lattices[examples]))
        compressed.append(encoder.predict(example[None])[0])
    return np.asarray(compressed)

def test_accuracy(encoder, decoder, test):
    error = []
    for lattice in test:
        # reshaped = np.expand_dims(lattice, axis=3)
        compressed = encoder.predict(lattice[None])[0]
        reconstructed = np.round(decoder.predict(compressed[None])[0])
        error.append(calculate_error(lattice, reconstructed.astype(bool)))
        # auto_encoder_plot(apply_constraints(lattice)[1], compressed, apply_constraints(reconstructed)[1])
    print("MEAN:", np.mean(error), " - ST-DEV:", np.std(error))