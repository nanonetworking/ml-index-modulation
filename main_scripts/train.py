import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow import keras
from os import listdir
from os.path import isfile, join
import os 
import argparse
from dataloader import DataLoader

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_path", type=str, default="new_data\\training_data\\", help="Path of simulations outputs which will be used in trainig")
    parser.add_argument("--training_mol_num", type=int, default=100000, help="Molecule number of training data")
    parser.add_argument("--simulation_time", type=float, default=5, help="Total time of simulation")
    parser.add_argument("--downsample_rate", type=float, default=0.1, help="Downsample rate")
    parser.add_argument('--window_numbers', nargs='+', default=[5, 6, 7],  type=int, help='Different number of windows that will be used in training')
    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset that will be created for each window number")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    opt = parser.parse_args()

    return opt

opt = get_args()


time = opt.simulation_time
downsample = opt.downsample_rate
x_range = int(time / downsample)
folder_path = opt.training_path
size = time / downsample
mol_num = opt.training_mol_num





def model_train(data, classes, numberr, opt):
    keras.backend.clear_session()
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=opt.val_ratio)
    inputs = Input(shape=(X_train.shape[1],X_train.shape[2],1))
#    x = Conv2D(1024, kernel_size=(8,5),strides=(1,5), activation="relu")(inputs)
    x = Conv2D(512, kernel_size = 2)(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size = 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size = 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size = 2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(inputs)
    x = Dense(512, activation="relu")(x)
    classes_dict_train = {}
    classes_dict_test = {}
    output_layer = {}
    loss_dic = {}
    for i in range(1,numberr+1):
        classes_dict_train["y_train_" + str(i)] = to_categorical(y_train[:,i-1] - 1)
        classes_dict_test["y_test_" + str(i)] = to_categorical(y_test[:,i-1] - 1)
        output_layer["output_" + str(i)] = Dense(8, activation="softmax")(x)
        loss_dic["dense_" + str(i+1)] = 'categorical_crossentropy'
    model = Model(inputs=inputs, outputs=list(output_layer.values()))
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
    epochs=200
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt, loss=loss_dic, metrics=['accuracy'])
    history = model.fit(X_train, list(classes_dict_train.values()),
                        validation_data=(X_test, list(classes_dict_test.values())), epochs=epochs, batch_size=128)
    return history, model    



dataloader = DataLoader(opt)
data = dataloader.read_folder()
dataloader.training_data_creater(data)
  



for numberr in opt.window_numbers:
    window_size = time / numberr 
    upper_limit = max(4,numberr)      
    data_folder = "window_" + str(window_size) + "_upper_" + str(upper_limit) + "\\"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    data_dict = {}
    data_dict["x"] = np.load(data_folder + "data_reshaped.npy")
    data_dict["y"] = np.load(data_folder + "classes_reshaped.npy")  

    data_reshaped = np.zeros((data_dict["x"].shape[0], 8, x_range, 1))
    classes = np.zeros((data_dict["x"].shape[0], numberr))

    data_reshaped[:,:,:,:] = data_dict["x"]
    classes[:, :] = data_dict["y"]

    hist, model = model_train(data_reshaped, classes, numberr, opt)
    model.save(data_folder + "multi_output_model.h5")
    
    



#PLOT EXAMPLE PICTURE
#x = np.arange(0.1,5.1,0.1)
#y = np.squeeze(data_reshaped)
##y = y * mol_num
#for i in range(1,9):   
#    if i == 1:
#        plt.title("Multivariate Time Series Input (1-7-4-7-1)")
#    ax = plt.subplot(8, 1, i)
#    plt.plot(x, y[3,i-1,:])
#    plt.yticks([0,0.005])
#    if not i==8:
#        plt.xticks([])      
#    if i==5:
#        ax.yaxis.set_label_coords(-0.12,1.2)
#        plt.ylabel("Emitted Molecule Rate", fontsize=16)
##plt.xlabel("Each Subplot Shows the Percentage of Emitted Molecules From Each Receiver")
#plt.xlabel("Time", fontsize=16)
#
#plt.show()