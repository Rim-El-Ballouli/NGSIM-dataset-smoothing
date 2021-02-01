# -*- coding: utf-8 -*-

# //$URL:: https://github.com/Rim-El-Ballouli/NGSIM-dataset-smoothing/tree/master/smothing-code
# //$Author:: Rim El Ballouli
# //$Date:: 31/12/2019

"""
This code
    1. smoothes out the noise in the local x and local y values in the NGSIM Dataset
    2. recomputes the velocites and accelerations
    3. saves smoothed dataset to three separate csv files
"""

from scipy import signal
import pandas as pd
import numpy as np
import numexpr


def get_file_name(index, file_name):
    if index == 0:
        return file_name[0].split('.', 1)[0]
    elif index == 1:
        return file_name[1].split('.', 1)[0]
    else:
        return file_name[2].split('.', 1)[0]


def get_smoothed_x_y_vel_accel(dataset, window):
    """
    this function returns four numpy arrays representing the smoothed
    1) local x, 2) local y, 3) velocity, 4) acceleration for a given numpy dataset.
    It relies on two helper functions  get_smoothed_x_y and get_smoothed_vel_accel
    :param dataset: numpy array representing the dataset to smooth it's local X , Y, velocity, acceleration
                    The numpy array should contains info for a single vehicle ID
                    otherwise result smoothed values are incorrect
    :param window: a smoothing window must be an odd integer value
                    if it set to 11 this means points are smoothed with 1 second interval equivalent to 10 points
                    if it set to 21 this means points are smoothed with 2 second interval equivalent to 20 points
    """
    smoothed_x_values, smoothed_y_values = get_smoothed_x_y(dataset, window)

    initial_vel = dataset[0, 11]
    initial_accel = dataset[0, 12]

    time_values = dataset[:, time_column]
    smoothed_vel, smoothed_accel = get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values,
                                                          time_values, initial_vel, initial_accel)
    return smoothed_x_values, smoothed_y_values, smoothed_vel, smoothed_accel


def get_smoothed_x_y(dataset, window):
    """
    this function computes the smoothed local x and local y using savgol_filter for a given numpy dataset
    and returns two numpy arrays containing the smoothed x and y values.
    :param dataset: numpy array representing the dataset to smooth it's local X , Y, velocity, acceleration
                    The numpy array should contains info for a single vehicle ID
                    otherwise result smoothed values are incorrect
    :param window: a smoothing window must be an odd integer value
                    if it set to 11 this means points are smoothed with 1 second interval equivalent to 10 points
                    if it set to 21 this means points are smoothed with 2 second interval equivalent to 20 points
    """
    smoothed_x_values = signal.savgol_filter(dataset[:, local_x], window, 3)
    smoothed_y_values = signal.savgol_filter(dataset[:, local_y], window, 3)

    return smoothed_x_values, smoothed_y_values


def get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values, time_values, initial_vel, initial_accel):
    """
    This function recomputes the velocity and acceleration for a given array of smoothed x, y values, time value
    To speedup calculation we use matrix functions to compute the values. For example, to compute velocity ,
    the x and y values are stacked to form matrix A. Then matrix B is then formed from Matrix A, but skipping t
    he first row. This implies that the x, y in first row in matrix B, are the next values of x and y in
    first row of matrix A. With two matrixes containing the current x, y and next x, y values we use fast matrix
    expressions to compute the smoothed velocities

    The function returns two numpy arrays representing the smoothed velocity and acceleration;

    :param smoothed_x_values: a numpy array of smoothed x values
    :param smoothed_y_values: a numpy array of smoothed y values
    :param time_values: a numpy array of smoothed time values values for the given x and y
    :param initial_vel: a single number containing the initial velocity
    :param initial_accel: a single number containing the initial acceleration
    """
    #create matrix of A containing current x and y and matrix B containing next x and y values
    x_y_matrix_A = np.column_stack((smoothed_x_values, smoothed_y_values))
    x_y_matrix_B = x_y_matrix_A [1:, :]
    #remove last row as it has no next values
    x_y_matrix_A = x_y_matrix_A[0:-1, :]

    # compute distance travelled between current and next x, y values
    dist_temp = numexpr.evaluate('sum((x_y_matrix_B - x_y_matrix_A)**2, 1)')
    dist = numexpr.evaluate('sqrt(dist_temp)')

    # create matrix A containing current time values, and matrix B containing next time values
    t_matrix_A = time_values
    t_matrix_B = t_matrix_A [1:]
    # remove last row
    t_matrix_A = t_matrix_A[0:-1]

    # evaluate smoothed velocity by dividing distance over delta time
    vel = numexpr.evaluate('dist * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_velocities = np.insert(vel, 0, initial_vel, axis=0)

    # create matrix A containing current velocities and matrix B containing next velocities
    vel_matrix_A = smoothed_velocities
    vel_matrix_B = vel_matrix_A [1:]
    # remove last row
    vel_matrix_A = vel_matrix_A[0:-1]

    # compute smoothed acceleration by dividing the delta velocity over delta time
    acc = numexpr.evaluate('(vel_matrix_B - vel_matrix_A) * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_accelaration = np.insert(acc, 0, initial_accel, axis=0)

    return np.array(smoothed_velocities), np.array(smoothed_accelaration)


def smooth_dataset(window, train, file_names):
    """
    this function loops over a set of train data, and set of unique vehicle ids
    and for each vehicle id in each training dataset, it requests from helper methods the smoothed
    x, y, vel, accel values and replaces the old values with the smoothed values. Finally the new
    smoothed dataset is printed to a file
    :param dataset:  data frame representing the dataset to smooth it's local X and Y
    :param train: a list of 3 numpy arrays containing the original ngsim data
    """
    # find  unique vehicle ids in all the datasets, in the previous version
    vehicle_ids = [train[0]['Vehicle_ID'].unique(), train[1]['Vehicle_ID'].unique(), train[2]['Vehicle_ID'].unique()]

    # convert to numpy arrays to fascilitate matrix operations to compute velocity and acceleration
    numpy_trains = [train[0].to_numpy(), train[1].to_numpy(), train[2].to_numpy()]

    for i in range(3): #in each dataset
        numpy_train = numpy_trains[i]
        print(f"##### smoothing x, y, vel, accl values in train data {str(i)}")

        # for each unique vehicle id smooth x and y, recompute vel and acel
        for vehicle in vehicle_ids[i]:
            # create a filter for given vehicle id and use it to create a numpy array containing info only for that vehicle
            filter = numpy_train[:,0] == vehicle
            numpy_vehicle_dataset = numpy_train[filter,:]

            smoothed_x_values, smoothed_y_values, smoothed_vel,smoothed_accel = \
                 get_smoothed_x_y_vel_accel(numpy_vehicle_dataset, window)

            # replace values of x, y, vel, accel, with new smoothed values
            numpy_train[filter, local_x] = [x for x in smoothed_x_values]
            numpy_train[filter, local_y] = [x for x in smoothed_y_values]
            numpy_train[filter, v_vel] = [x for x in smoothed_vel]
            numpy_train[filter, v_acc] = [x for x in smoothed_accel]

        # print to file
        file_name = get_file_name(i, file_names)
        file_path = path_to_smoothed_dataset + file_name + '_smoothed_' + str(window) + '.csv'
        with open(file_path, 'w') as f:
            np.savetxt(file_path, numpy_trains[i], delimiter=",")


def main():
    # smooth window must be an odd value
    smoothing_window = 21
    print(f"Smoothing window is set to {str(smoothing_window)}")

    # change the file names as needed
    global file_names
    file_names = ['0750_0805_us101.csv', '0805_0820_us101.csv', '0820_0835_us101.csv']

    # define the index of columns containing vehicle id, time, local x, local y, velocity and acceleration
    # these indexes correspond to the original dataset if not modified
    # the indexes help treat the dataset as matrix and perform smoothing using matrix functions
    global vehicle_id, time_column, local_x, local_y, v_vel, v_acc
    vehicle_id, time_column, local_x, local_y, v_vel, v_acc  = 0, 3, 4, 5, 11, 12

    # specify the path to the input NGSIM dataset and the path to the output smoothed dataset
    global path_to_dataset, path_to_smoothed_dataset
    path_to_dataset = 'C:/.../NGSIM/dataset/'
    path_to_smoothed_dataset = 'C:/.../smoothed/'


    # load the NGSIM data from the CSV files
    train1 = pd.read_csv(path_to_dataset + file_names[0], engine='c')
    train2 = pd.read_csv(path_to_dataset + file_names[1], engine='c')
    train3 = pd.read_csv(path_to_dataset + file_names[2], engine='c')

    train = [train1, train2, train3]

    smooth_dataset(smoothing_window, train, file_names)

if __name__ == '__main__':
    main()
