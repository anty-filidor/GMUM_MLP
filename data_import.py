import numpy as np
import pickle


def unpickle(file):
    """Function to unpack data; argument is a file; returns a dictionary"""
    with open(file, 'rb') as fo: # dict_keys(['batch_label', 'labels', 'data', 'filenames'])
        dict = pickle.load(fo, encoding='latin1') # Bardzo ważne kodowanie, inaczej czyta klucze słownika jak bajty!!!
    return dict


def convert_labels(old_labels):
    '''Function to convert labels from class coding as one number to vector, eg. 3 -> [0,0,0,1,0,0,0,0,0,0];
    argument is an dictionary; returns a dictionary'''
    new_labels = np.zeros(shape=(len(old_labels), 10), dtype=int)
    for i in range(0, len(old_labels)):
        value = old_labels[i]
        new_labels[i][value] = 1
    return np.array(new_labels)


def unpack_core_data(file):
    """Function which merges "unpickle" and "convert_labels"; argument is a file; returns a dictionary"""
    data = unpickle(file)
    data['labels'] = convert_labels(data['labels'])
    return data


def give_mini_batch(file):
    """Function which merges "unpickle" and "convert_labels"; argument is a file; returns a dictionary"""
    data = unpickle(file)
    data = {'labels': convert_labels(data['labels']), 'data': data['data']}
    return data


def give_one_big_batch(filenames_matrix):
    """Function creates one big batch in form of dictionary with labels and images"""
    labels = []
    data = []
    for i in range(0, len(filenames_matrix)):
        temp = give_mini_batch(filenames_matrix[i])
        #  print(type(temp), temp["labels"].shape, temp["data"].shape)
        for j in range(0, len(temp['labels'])):
            labels.append(temp['labels'][j])
            data.append(temp['data'][j])
    labels = np.array(labels)
    data = np.array(data)

    one_big_batch = {'labels': labels, 'data': data}
    #  print(type(one_big_batch),  one_big_batch["labels"].shape,  one_big_batch["data"].shape)
    return one_big_batch
