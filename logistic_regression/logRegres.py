import numpy as np
import matplotlib.pyplot as plt


def load_data():
    x_mat = []
    label_mat = []
    with open('testSet.txt') as file_obj:
        for line in file_obj.readlines():
            line_array = line.strip().split()
            x_mat.append([1.0, float(line_array[0]), float(line_array[1])])
            label_mat.append(int(line_array[2]))
    return x_mat, label_mat


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return .5 * (1 + np.tanh(.5 * x))


def batch_gra_ascent(x_mat, label_mat):
    x_matrix = np.mat(x_mat)
    label_matrix = np.mat(label_mat).transpose()

    num_samples, num_features = np.shape(x_matrix)
    weights = np.ones((num_features, 1))

    max_cycles = 500
    alpha = 0.001

    for i in range(max_cycles):
        h = sigmoid(x_matrix * weights)
        weights = weights + alpha * x_matrix.transpose() * (label_matrix - h)

    return weights


def sto_gra_ascent0(x_mat, label_mat):
    x_array = np.array(x_mat)
    num_samples, num_features = np.shape(x_array)
    alpha = 0.01
    weights = np.ones(num_features)
    for j in range(200):
        for i in range(num_samples):
            h = sigmoid(sum(x_array[i] * weights))
            weights = weights + alpha * (label_mat[i] - h) * x_array[i]

    return weights


def sto_gra_ascent1(x_mat, label_mat, num_iter=150):
    x_array = np.array(x_mat)
    num_samples, num_features = np.shape(x_array)
    weights = np.ones(num_features)
    for i in range(num_iter):
        indexes = list(range(num_samples))
        for j in range(num_samples):
            alpha = 4.0 / (1.0 + i + j) + 0.01
            rand_index = int(np.random.uniform(0, len(indexes)))
            h = sigmoid(sum(weights * x_array[rand_index]))
            weights = weights + alpha * \
                (label_mat[rand_index] - h) * x_array[rand_index]
            del(indexes[rand_index])

    return weights


def plot_best_fit(wei):
    weights = wei
    # weights = wei.getA()
    x_mat, label_mat = load_data()
    data_array = np.array(x_mat)
    num_samples = np.shape(data_array)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(num_samples):
        if int(label_mat[i]) == 1:
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# x_mat, label_mat = load_data()
# weights = sto_gra_ascent1(x_mat, label_mat)
# plot_best_fit(weights)


def classify_vector(input_x, weights):
    prob = sigmoid(sum(input_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    x_mat = []
    label_mat = []
    with open('horseColicTraining.txt') as train_data:
        data_lines = train_data.readlines()
        for current_line in data_lines:
            line_list = current_line.strip().split('\t')
            x_row = []
            for i in range(21):
                x_row.append(float(line_list[i]))
            label_mat.append(float(line_list[21]))
            x_mat.append(x_row)
    weights = sto_gra_ascent1(x_mat, label_mat, num_iter=500)

    error_items = 0
    num_of_items = 0
    with open('horseColicTest.txt') as test_data:
        data_lines = test_data.readlines()
        for current_line in data_lines:
            line_list = current_line.strip().split('\t')
            input_x = []
            for i in range(21):
                input_x.append(float(line_list[i]))
            label = float(line_list[21])
            pred_result = classify_vector(input_x, weights)
            if pred_result != label:
                error_items += 1
            num_of_items += 1
    error_rate = error_items / num_of_items
    print("the error rate of this test is: %f" % error_rate)

    return error_rate


def multi_test():
    num_of_tests = 10
    error_sum = 0.0
    for i in range(num_of_tests):
        error_sum += colic_test()
    print("after %d iterations, the average error rate is %f" %
          (num_of_tests, error_sum / float(num_of_tests)))


multi_test()
