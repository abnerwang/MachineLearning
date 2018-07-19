import numpy as np
import re


def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak',
                        'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_voca_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


# set-of-words model
def set_of_words2vec(vocab_list, document):
    return_vec = [0] * len(vocab_list)
    for word in document:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
    return return_vec


# bag-of-words model
def bag_of_words2vec(vocab_list, document):
    return_vec = [0] * len(vocab_list)
    for word in document:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_naive_bayes0(train_matrix, train_categroy):
    num_of_doc = len(train_matrix)
    num_of_words = len(train_matrix[0])
    num_of_abusive = sum(train_categroy)
    prob_abusive = float(num_of_abusive) / num_of_doc    # flag: 1
    prob_non_abusive = 1 - prob_abusive    # flag: 0

    # Laplace smoothing
    doc0_features = np.ones(num_of_words)
    doc1_features = np.ones(num_of_words)
    num0_of_w = 2.0
    num1_of_w = 2.0

    # sparse discrete values
    for i in range(num_of_doc):
        if train_categroy[i] == 1:
            doc1_features += train_matrix[i]
            num1_of_w += sum(train_matrix[i])
        else:
            doc0_features += train_matrix[i]
            num0_of_w += sum(train_matrix[i])

    # add log function to prevent data underflow
    prob0_vec = np.log(doc0_features / num0_of_w)
    prob1_vec = np.log(doc1_features / num1_of_w)

    return prob_abusive, prob_non_abusive, prob1_vec, prob0_vec


def classify_via_nb0(doc_vector, prob_abusive, prob_non_abusive, prob1_vec, prob0_vec):
    prob0 = sum(doc_vector * prob0_vec) + np.log(prob_non_abusive)
    prob1 = sum(doc_vector * prob1_vec) + np.log(prob_abusive)

    if prob1 > prob0:
        return 1
    else:
        return 0


def test_nb0():
    list_of_posts, list_of_labels = load_dataset()
    vocab_list = create_voca_list(list_of_posts)
    train_matrix = []
    for post in list_of_posts:
        train_matrix.append(set_of_words2vec(vocab_list, post))
    prob_abusive, prob_non_abusive, prob1_vec, prob0_vec = train_naive_bayes0(
        np.array(train_matrix), np.array(list_of_labels))

    test_entry = ['love', 'my', 'dalmation']
    entry_vector = np.array(set_of_words2vec(vocab_list, test_entry))
    classify_result = classify_via_nb0(
        entry_vector, prob_abusive, prob_non_abusive, prob1_vec, prob0_vec)
    print(','.join(test_entry) + ' classified as: ' + str(classify_result))

    test_entry = ['stupid', 'garbage']
    entry_vector = np.array(set_of_words2vec(vocab_list, test_entry))
    classify_result = classify_via_nb0(
        entry_vector, prob_abusive, prob_non_abusive, prob1_vec, prob0_vec)
    print(','.join(test_entry) + ' classified as: ' + str(classify_result))


def text_parse(big_string):
    list_of_tokens = re.split(r'\W+', big_string)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


def spam_test():
    doclist = []
    class_label = []
    for i in range(1, 26):
        with open('email/ham/%d.txt' % i, encoding='ISO-8859-1') as email_obj:
            doclist.append(text_parse(email_obj.read()))
            class_label.append(0)
        with open('email/spam/%d.txt' % i, encoding='ISO-8859-1') as email_obj:
            doclist.append(text_parse(email_obj.read()))
            class_label.append(1)
    vocab_list = create_voca_list(doclist)
    train_index = list(range(50))

    # Randomly select 10 emails for testing
    test_index = []
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(train_index)))
        test_index.append(rand_index)
        del(train_index[rand_index])

    train_matrix = []
    train_categroy = []
    for index in train_index:
        feature_vector = set_of_words2vec(vocab_list, doclist[index])
        train_matrix.append(feature_vector)
        train_categroy.append(class_label[index])

    prob_abusive, prob_non_abusive, prob1_vec, prob0_vec = train_naive_bayes0(
        np.array(train_matrix), np.array(train_categroy))

    error_count = 0
    for index in test_index:
        feature_vector = set_of_words2vec(vocab_list, doclist[index])
        classify_result = classify_via_nb0(
            feature_vector, prob_abusive, prob_non_abusive, prob1_vec, prob0_vec)
        if classify_result != class_label[index]:
            error_count += 1

    error_rate = error_count / (float)(len(test_index))
    print("the error rate is %f" % error_rate)


spam_test()
