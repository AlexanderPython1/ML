import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression


def main():
    train = pd.read_csv("train_with_target.csv")
    test = pd.read_csv("test_without_target.csv")

    X_train = train[["feature_1", "feature_2"]].values
    y_train = train["target"].values
    X_test = test[["feature_1", "feature_2"]].values

    # Train a simple classifier on the old labeled data to help map clusters to labels
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Fit a 2-component GMM on the new (test) data
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X_test)
    clusters = gmm.predict(X_test)

    # Determine which GMM component corresponds to target==1 using classifier's prediction on component means
    means = gmm.means_
    probs = clf.predict_proba(means)
    # probs[:, index_of_class1]
    class_index = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    mean_probs_class1 = probs[:, class_index]
    # component -> target label mapping
    comp_to_label = {0: 1 if mean_probs_class1[0] >= mean_probs_class1[1] else 2,
                     1: 1 if mean_probs_class1[1] >= mean_probs_class1[0] else 2}

    preds = [comp_to_label[c] for c in clusters]

    out = pd.DataFrame({"target": preds})
    out.to_csv("test_target_only.csv", index=False)


if __name__ == "__main__":
    main()
