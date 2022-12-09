import pandas as pd
from data_preparation import prepare_dataset
from data_set import get_data
from models import eval_model, test_model, train_RF, train_SVC, train_NN
from demo import run_demo
from sklearn.metrics import precision_score, recall_score, accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay


def main():
    # DATA PREPARATION
    dataset = pd.read_csv('dataset.csv', encoding="ISO-8859-1")
    data_kaggle = pd.DataFrame(get_data(), columns=['news', 'type'])
    X_train, X_val, X_test, y_train, y_val, y_test, vect = prepare_dataset(data_kaggle)

    # MODELS TRAINING
    print("\n--------------------------------------------------------")
    print("------------------- MODELS  TRAINING -------------------")
    print("--------------------------------------------------------\n")

    plt.figure(figsize=(10, 10))
    # Random Forest
    modelRF = train_RF(X_train, y_train)
    eval_model(modelRF, X_val, y_val)
    plot_confusion_matrix(modelRF, X_test, y_test, cmap='Blues')
    plt.show()
    # SVM
    modelSVC = train_SVC(X_train, y_train)
    eval_model(modelSVC, X_val, y_val)
    plot_confusion_matrix(modelSVC, X_test, y_test, cmap='Blues')
    plt.show()
    # NN
    modelNN = train_NN(X_train, y_train)
    eval_model(modelNN, X_val, y_val)
    plot_confusion_matrix(modelNN, X_test, y_test, cmap='Blues')
    plt.show()

    '''xclf_list = [
        (modelNN, "Logistic"),
        (modelSVC, "SVC"),
        (modelRF, "Random forest"),
    ]
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_predictions(
            clf,
            y_true=X_test,
            y_prob=y_test,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()'''

    # MODELS TESTING
    print("\n--------------------------------------------------------")
    print("-------------- MODELS  TESTING (accuracy) --------------")
    print("--------------------------------------------------------\n")
    print("RANDOM FOREST:     ", test_model(modelRF, X_test, y_test))
    print("SVC:               ", test_model(modelSVC, X_test, y_test))
    print("NEURAL NETWORK:    ", test_model(modelNN, X_test, y_test))

    # RUN THE DEMO
    run_demo(vect, modelSVC)


if __name__ == '__main__':
    main()
