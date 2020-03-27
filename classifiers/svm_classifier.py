import sys
sys.path.append('./')

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from data_pipeline.utils import get_all_data_from_loader

def classify(training_dataloader, test_dataloader, kernel, cv=False):
    training_features, training_labels = get_all_data_from_loader(training_dataloader)
    test_features, test_labels = get_all_data_from_loader(test_dataloader)
    print('Got features')
    classifier = SVC(kernel=kernel)
    if cv:
        cv_scores = cross_val_score(classifier, training_features, training_labels, cv=3)
        print('CV score  mean', cv_scores.mean())
    classifier.fit(training_features, training_labels)
    test_scores = classifier.score(test_features, test_labels)
    print('Test set scores', test_scores)
    return classifier
