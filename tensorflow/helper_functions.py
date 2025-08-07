import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_loss_accuracy_curves(history):
    """
    Plots the loss and accuracy curves from a model's training history, side by side.

    Args:
    history: A history object from TensorFlow or a dictionary with 'loss', 'val_loss', 'accuracy', 'val_accuracy' keys for PyTorch.

    Example usage:
        plot_loss_accuracy_curves(history=model_history)
    """

    # Check if 'history' is a TensorFlow History object or a dictionary (PyTorch-style)
    if isinstance(history, dict):
        loss = history['loss']
        val_loss = history['val_loss']
        acc = history['accuracy'] if 'accuracy' in history else None
        val_acc = history['val_accuracy'] if 'val_accuracy' in history else None
    else:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['accuracy'] if 'accuracy' in history.history else None
        val_acc = history.history['val_accuracy'] if 'val_accuracy' in history.history else None

    # Plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Loss subplot
    axes[0].plot(loss, label='Training Loss')
    axes[0].plot(val_loss, label='Validation Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy subplot
    if acc and val_acc:
        axes[1].plot(acc, label='Training Accuracy')
        axes[1].plot(val_acc, label='Validation Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'Accuracy data not available', horizontalalignment='center', verticalalignment='center')
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def get_confusion_matrix(model, generator, N):
    """
    Generates a confusion matrix for predictions on the dataset from the generator.

    Args:
    - generator: Data generator (for test or validation set).
    - N: Number of samples to evaluate.

    Returns:
    - cm: Confusion matrix.
    - targets: True labels.
    - predictions: Predicted labels.
    """
    predictions = []
    targets = []
    i = 0
    for x, y in generator:
        if i % 100 == 0:
            print(f'{i/N * 100:.2f}% complete')
            
        p = model.predict(x, verbose = 0)
        p = np.argmax(p, axis=1)  # Get the predicted class
        y = np.argmax(y, axis=1)  # Get the true class
        
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        
        i += 1
        if i >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm, targets, predictions

def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(12, 12), text_size=8):
    """
    Plots a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).

    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.
    """
    
    # Check if y_true and y_pred have the same shape
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors represent correctness
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(n_classes)

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.tick_params(axis='x', rotation=90)

    # Set the threshold for text colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        plt.text(j, i, f"{cm[i, j]}",# ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)

    plt.show()
