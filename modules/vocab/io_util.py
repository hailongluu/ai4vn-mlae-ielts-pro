import pickle


def load_data(filename):
    data = []
    with open(filename, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        pkl_file.close()
    return data


def dump_data(filename, data):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
        pkl_file.close()
