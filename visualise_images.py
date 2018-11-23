import numpy as np
import matplotlib.pyplot as plt
import data_import as di


'''Import testing data'''
testing_batch = di.give_mini_batch('cifar-10-batches-py/test_batch')
X = testing_batch['data']
Y = testing_batch['labels']
X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
print(len(X))
Y = np.array(Y)
classes_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


'''Import predictions and isolate 5 representatives for each class'''
predictions = (np.genfromtxt('np.csv', delimiter=',')).astype(int) # index of image, predicted as class number, belonged for real to class
representatives = [[],[],[],[],[],[],[],[],[],[]]  # container for representatives of each class
for i in range(0, 10):
    j = 0
    while len(representatives[i]) < 5:
        if predictions[j][2] == i:
            representatives[i].append(predictions[j])
        j += 1
representatives = np.array(representatives)


'''Print images'''
for i in range(0, 10):  # iterate over classes
    group_of_images = representatives[i]
    fig, axes1 = plt.subplots(1, 5, figsize=(7, 2.2))
    for j in range(0, 5): # iterate over representatives in current class
        axes1[j].set_axis_off()
        axes1[j].imshow(X[group_of_images[j][0]:group_of_images[j][0] + 1][0])
        axes1[j].set_title(classes_names[group_of_images[j][1]])

    fig.suptitle("Class " + classes_names[i] + " recognised as:", fontsize=16)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    plt.show()

