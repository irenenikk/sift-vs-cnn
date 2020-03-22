def get_indices_and_labels(file, label_i):
    indices = pd.read_csv(file, sep=' ', header=None)
    labels = indices.iloc[:, label_i]
    labels = labels - 1
    return indices, labels
