import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_training_metrics(history, save_path='train_metrics.png'):
    fig, ((loss_ax, acc_ax), (val_loss_ax, val_acc_ax)) = plt.subplots(2, 2, figsize=(20, 10))

    loss_ax.plot(history['train_loss'])
    loss_ax.set_title('Training Loss')

    acc_ax.plot(history['train_acc'])
    acc_ax.set_title('Training Accuracy')

    val_loss_ax.plot(history['val_loss'])
    val_loss_ax.set_title('Validation Loss')

    val_acc_ax.plot(history['val_acc'])
    val_acc_ax.set_title('Validation Accuracy');

    fig.savefig(str(save_path))


def plot_confmat(true_labels, pred_labels, save_path='confmat.png'):
    """
    Plots a confusion matrix from given data
    """
    fig, ax = plt.subplots(1, 1, num=2, figsize=(15, 10))

    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix
    for pair in np.argwhere(np.isnan(cm_norm)):
        cm_norm[pair[0]][pair[1]] = 0

    annot = np.zeros_like(cm, dtype=object)
    for i in range(annot.shape[0]):  # Creates an annotation array for the heatmap
        for j in range(annot.shape[1]):
            annot[i][j] = f'{cm[i][j]}\n{round(cm_norm[i][j] * 100, ndigits=3)}%'

    ax = sns.heatmap(cm_norm, annot=annot, fmt='', cbar=True, cmap=plt.cm.magma, vmin=0, ax=ax) # plot the confusion matrix

    ax.set(xlabel='Predicted Label', ylabel='Actual Label')
    fig.savefig(str(save_path))
    