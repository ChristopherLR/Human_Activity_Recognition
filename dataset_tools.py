import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split


class Dataset:
    def __init__(self, dataset_path):
        df = load_dataset(dataset_path)
        df.columns = list(range(24)) + ['type']
        self.sample_rate = 200
        self.data_frame = df
        activities = self.data_frame.groupby('type')
        self.activities = activities
        self.sitting = extract_sensors(activities.get_group(1))
        self.lying = extract_sensors(activities.get_group(2))
        self.standing = extract_sensors(activities.get_group(3))
        self.washing_dishes = extract_sensors(activities.get_group(4))
        self.vacuuming = extract_sensors(activities.get_group(5))
        self.sweeping = extract_sensors(activities.get_group(6))
        self.walking_outside = extract_sensors(activities.get_group(7))
        self.ascending_stairs = extract_sensors(activities.get_group(8))
        self.descending_stairs = extract_sensors(activities.get_group(9))
        self.treadmill_running = extract_sensors(activities.get_group(10))
        self.bicycling_50_watt = extract_sensors(activities.get_group(11))
        self.bicycling_100_watt = extract_sensors(activities.get_group(12))
        self.rope_jumping = extract_sensors(activities.get_group(13))
        self.all = [
            (self.sitting['all'], 'sitting'),
            (self.lying['all'], 'lying'),
            (self.standing['all'], 'standing'),
            (self.washing_dishes['all'], 'washing_dishes'),
            (self.vacuuming['all'], 'vacuuming'),
            (self.sweeping['all'], 'sweeping'),
            (self.walking_outside['all'], 'walking_outside'),
            (self.ascending_stairs['all'], 'ascending_stairs'),
            (self.descending_stairs['all'], 'descending_stairs'),
            (self.treadmill_running['all'], 'treadmill_running'),
            (self.bicycling_50_watt['all'], 'bicycling_50_watt'),
            (self.bicycling_100_watt['all'], 'bicycling_50_watt'),
            (self.rope_jumping['all'], 'rope_jumping')
        ]

    def extract_sensors(self, reduce):
        activities = self.activities
        self.sitting = extract_sensors(activities.get_group(1), reduce)
        self.lying = extract_sensors(activities.get_group(2), reduce)
        self.standing = extract_sensors(activities.get_group(3), reduce)
        self.washing_dishes = extract_sensors(activities.get_group(4), reduce)
        self.vacuuming = extract_sensors(activities.get_group(5), reduce)
        self.sweeping = extract_sensors(activities.get_group(6), reduce)
        self.walking_outside = extract_sensors(activities.get_group(7), reduce)
        self.ascending_stairs = extract_sensors(activities.get_group(8), reduce)
        self.descending_stairs = extract_sensors(activities.get_group(9), reduce)
        self.treadmill_running = extract_sensors(activities.get_group(10), reduce)
        self.bicycling_50_watt = extract_sensors(activities.get_group(11), reduce)
        self.bicycling_100_watt = extract_sensors(activities.get_group(12), reduce)
        self.rope_jumping = extract_sensors(activities.get_group(13), reduce)
        self.all = [
            (self.sitting['all'], 'sitting'),
            (self.lying['all'], 'lying'),
            (self.standing['all'], 'standing'),
            (self.washing_dishes['all'], 'washing_dishes'),
            (self.vacuuming['all'], 'vacuuming'),
            (self.sweeping['all'], 'sweeping'),
            (self.walking_outside['all'], 'walking_outside'),
            (self.ascending_stairs['all'], 'ascending_stairs'),
            (self.descending_stairs['all'], 'descending_stairs'),
            (self.treadmill_running['all'], 'treadmill_running'),
            (self.bicycling_50_watt['all'], 'bicycling_50_watt'),
            (self.bicycling_100_watt['all'], 'bicycling_100_watt'),
            (self.rope_jumping['all'], 'rope_jumping')
        ]

    def extract_all_features(self, features, window_len):

        activity_0, name_0 = self.all[0]
        training, testing = extract_features(activity_0, features, window_len)
        training['activity'] = name_0
        testing['activity'] = name_0

        for activity, name in self.all[1:]:
            train, test = extract_features(activity, features, window_len)
            test['activity'] = name
            train['activity'] = name
            testing = testing.append(test)
            training = training.append(train)

        return training, testing


def extract_features(activity, features, window_len):
    train, test = train_test_split(activity, shuffle=False)
    training = {}
    testing = {}

    train_intervals = math.floor(len(train) / window_len)
    test_intervals = math.floor(len(test) / window_len)

    feature_len = len(features)

    idx = 0
    for column in range(len(train.columns)):

        for n in range(feature_len):
            training[n + idx] = []
            testing[n + idx] = []
        for window in range(train_intervals):
            sample = train[[column]][window * window_len:(window + 1) * window_len].values
            for f_idx, fn in enumerate(features):
                training[idx + f_idx].append(fn(sample))

        for window in range(test_intervals):
            sample = test[[column]][window * window_len:(window + 1) * window_len].values
            for f_idx, fn in enumerate(features):
                testing[idx + f_idx].append(fn(sample))
        idx = idx + feature_len

    training = pd.DataFrame(training)
    testing = pd.DataFrame(testing)

    return training, testing



def load_dataset(filename):
    df = pd.read_csv(filename, sep=',', header=None)
    return df


# Extract each sensor and separate into body location
# - reduce drops initial rows that may contain undesirable data
def extract_sensors(df, reduce=500):
    sensor_dict = dict({'wrist': {}, 'chest': {}, 'hip': {}, 'ankle': {}})
    reduced_dict = df.iloc[reduce:].reset_index(drop=True)
    sensor_dict['all'] = reduced_dict.drop('type', axis=1)

    sensor_dict['wrist']['accel'] = reduced_dict[[0, 1, 2]]
    sensor_dict['wrist']['accel'].columns = ['Ax', 'Ay', 'Az']
    sensor_dict['wrist']['gyro'] = reduced_dict[[3, 4, 5]]
    sensor_dict['wrist']['gyro'].columns = ['Gx', 'Gy', 'Gz']

    sensor_dict['chest']['accel'] = reduced_dict[[6, 7, 8]]
    sensor_dict['chest']['accel'].columns = ['Ax', 'Ay', 'Az']
    sensor_dict['chest']['gyro'] = reduced_dict[[9, 10, 11]]
    sensor_dict['chest']['gyro'].columns = ['Gx', 'Gy', 'Gz']

    sensor_dict['hip']['accel'] = reduced_dict[[12, 13, 14]]
    sensor_dict['hip']['accel'].columns = ['Ax', 'Ay', 'Az']
    sensor_dict['hip']['gyro'] = reduced_dict[[15, 16, 17]]
    sensor_dict['hip']['gyro'].columns = ['Gx', 'Gy', 'Gz']

    sensor_dict['ankle']['accel'] = reduced_dict[[18, 19, 20]]
    sensor_dict['ankle']['accel'].columns = ['Ax', 'Ay', 'Az']
    sensor_dict['ankle']['gyro'] = reduced_dict[[21, 22, 23]]
    sensor_dict['ankle']['gyro'].columns = ['Gx', 'Gy', 'Gz']

    return sensor_dict


def lowpass_filter(data, order, cutoff, sampling_rate=200):
    nyquist_rate = sampling_rate / 2
    Wn = cutoff / nyquist_rate
    b, a = signal.butter(order, Wn, 'lowpass', analog=False)
    num_columns = len(data.columns)-1
    filtered_data = data.values
    for i in range(num_columns):
        data[[i]] = signal.filtfilt(b, a, filtered_data[:,i])


