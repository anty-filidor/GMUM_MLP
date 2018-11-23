import tensorflow as tf
import data_import as di
import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


'''Import data to learn and test'''
batches_names = ["cifar-10-batches-py/data_batch_1",
                 "cifar-10-batches-py/data_batch_2",
                 "cifar-10-batches-py/data_batch_3",
                 "cifar-10-batches-py/data_batch_4",
                 "cifar-10-batches-py/data_batch_5"]
training_batch = di.give_one_big_batch(batches_names)
length_of_training_batch, _ = training_batch['labels'].shape
testing_batch = di.give_mini_batch('cifar-10-batches-py/test_batch')


'''Network parameters'''
n_hidden_1 = 1024  # 1st layer number of neurons
n_input = 32*32*3  # CIFAR-10 data input (img shape: 32*32*3)
n_classes = 10  # CIFAR-10 total classes (0-9 digits)
hidden_layers = 1
neurons_sequence = [n_input, n_hidden_1, n_classes]

X = tf.placeholder("float", [None, n_input])  # tf Graph input
Y = tf.placeholder("float", [None, n_classes])  # tf Graph input

weights = {  # initialise weights
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {  # initialise biases
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(x):
    """Function witch creates NN. Argument is a placeholder for input data, returns model of NN"""
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    return out_layer


def l1_loss(params):
    return tf.reduce_sum(tf.abs(params))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


'''Construct model'''
logits = multilayer_perceptron(X)
regulariser = tf.nn.l2_loss(weights['h1'])
#  regulariser = l1_loss(weights['h1'])
#  regulariser = l1_loss(weights['h1']) + tf.nn.l2_loss(weights['h1'])
#  regulariser = 0


'''Learning parameters'''
#  learning_rate = 0.005
training_epochs = 20
batch_size = 100


global_step = tf.Variable(0, trainable=False)
'''
starter_learning_rate = 0.3
end_learning_rate = 0.0001
decay_steps = 10000
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps, end_learning_rate, power=0.5)
'''

display_step = 1
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
loss_op = tf.reduce_mean(loss_op + 0.01 * regulariser)  # define loss function
optimizer = tf.train.AdamOptimizer()  # initialise optimiser
#  optimizer = tf.train.AdadeltaOptimizer()
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#  optimizer = tf.train.AdagradOptimizer(0.001)
train_op = optimizer.minimize(loss_op, global_step=global_step)


'''Initializing the variables'''
init = tf.global_variables_initializer()


'''Dictionary to save plotting data'''
training_logs = {'epoch': [], 'accuracy': [], 'cost': []}


with tf.Session() as sess:
    sess.run(init)

    '''Test model at random weights and save it as the best model'''
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1))
    best_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    best_accuracy_float = best_accuracy.eval({X: testing_batch['data'][0: batch_size], Y: testing_batch['labels'][0: batch_size]})
    best_epoch = -1
    best_cost = -1
    best_MLP = logits

    print("Epoch:", '%04d' % 0, "cost unknown (random weights) Accuracy at mini-test:", best_accuracy_float)

    '''Overall training cycle'''
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(length_of_training_batch / batch_size)

        '''Training cycle in one epoch'''
        for i in range(total_batch):
            # start_of_minibatch_in_batch = random.randrange(0, length_of_training_batch - batch_size)
            # batch_y = training_batch['labels'][start_of_minibatch_in_batch: start_of_minibatch_in_batch + batch_size]
            # batch_x = training_batch['data'][start_of_minibatch_in_batch: start_of_minibatch_in_batch + batch_size]
            batch_x = training_batch['data'][batch_size * i: batch_size * (i + 1)]
            batch_y = training_batch['labels'][batch_size * i: batch_size * (i + 1)]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        '''Display logs and save state per epoch step'''
        if epoch % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_float = accuracy.eval({X: testing_batch['data'][0: batch_size], Y: testing_batch['labels'][0: batch_size]})

            training_logs['epoch'].append(epoch)
            training_logs['accuracy'].append(accuracy_float)
            training_logs['cost'].append(avg_cost)

            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), "Accuracy at mini-test:", accuracy_float)
            #  If epoch improved NN save it as the best
            if best_accuracy_float <= accuracy_float:
                best_accuracy_float = accuracy_float
                best_MLP = logits
                best_epoch = epoch
                best_cost = avg_cost

    print("Optimization Finished! Best accuracy was in epoch:", '%04d' % (best_epoch + 1),
          "cost={:.9f}".format(best_cost))

    '''Test model'''
    pred = tf.nn.softmax(best_MLP)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    predictions = tf.argmax(pred, 1)
    predictions = predictions.eval(feed_dict={X: testing_batch['data']}, session=sess)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy = accuracy.eval({X: testing_batch['data'], Y: testing_batch['labels']})

    training_logs['epoch'].append(training_epochs)
    training_logs['accuracy'].append(accuracy)
    training_logs['cost'].append(0)

    print("Accuracy:", accuracy)

    '''Plot data'''
    print("\n\nMLP parameters:\n\tlayers " + str(hidden_layers) + "(" + str(neurons_sequence) + ")" +
          "\n\tactivation function: ReLu \nTraining parameters:\n\tepochs " +
          str(training_epochs) + "\n\tbatch size " + str(batch_size) + "\n\tcost function: softmax cross entropy" +
          "\n\toptimiser: AdamOptimiser" + "\n\tregularisation: l2")

    '''Plot figure acc & cost ~ epoch''' '''
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='red')
    ax1.plot(training_logs['epoch'], training_logs['accuracy'], 'o--', color='red')
    ax1.plot([training_epochs], [accuracy], 'o', color='green', label="Accuracy at full test")
    plt.ylim((0, 1))
    ax1.tick_params(axis='y', labelcolor='red')
    plt.legend()

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Cost', color='blue')  # we already handled the x-label with ax1
    ax2.plot(training_logs['epoch'], training_logs['cost'], 'o--', color='blue', label="Cost")
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()
    plt.show()'''

    '''Plot confusion matrix'''
    labels = di.unpickle('cifar-10-batches-py/test_batch')['labels']
    classes_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # y_pred = pd.Series(prediction, name='Predicted')

    cnf_matrix = confusion_matrix(labels, predictions)
    np.set_printoptions(precision=2)

    '''# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    '''

    '''Save bad predictions to further processing'''
    container = []  # 'index of image', 'predicted as class number', 'belonged for real to class'
    for i in range(0, len(predictions)):   # iteracja po predykcjach
        #  print("i:", i, "predictions[i]:", predictions[i])
        for j in range(0, 10):  # iteracja po klasach
            #  print("\tj:", j)
            if predictions[i] == j:
                if predictions[i] != labels[i]:
                    container.append([i, predictions[i], labels[i]])
                break
    #  print(container)
    #  container.sort(key=lambda r: r[2])
    container = (np.array(container)).astype(int)
    np.savetxt('np.csv', container, fmt='%.0f', delimiter=',',
               header='index of image, predicted as class number, belonged for real to class')

