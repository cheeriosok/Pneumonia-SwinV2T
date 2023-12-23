import os
import zipfile
import requests
from pathlib import Path
from typing import List
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def set_random_seeds(seed: int = 42):
    """
    Sets random seeds for reproducibility in torch operations.

    Args:
    seed (int): Seed value, default is 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def download_and_extract_data(source_url: str, target_directory: str, remove_zip: bool = True) -> Path:
    """
    Downloads and extracts a zip file from the given URL into the target directory.

    Args:
    source_url (str): URL of the zip file to be downloaded.
    target_directory (str): Local directory to extract the files into.
    remove_zip (bool): If true, removes the zip file after extraction.

    Returns:
    Path: Path to the extracted directory.
    """
    data_path = Path("data/")
    target_path = data_path / target_directory

    if target_path.is_dir():
        print(f"[INFO] {target_path} directory exists, skipping download.")
        return target_path

    print(f"[INFO] Creating {target_path} directory...")
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Download and save the zip file
    zip_file_name = Path(source_url).name
    with open(data_path / zip_file_name, "wb") as file:
        print(f"[INFO] Downloading {zip_file_name} from {source_url}...")
        file.write(requests.get(source_url).content)

    # Extract the zip file
    with zipfile.ZipFile(data_path / zip_file_name, "r") as zip_ref:
        print(f"[INFO] Extracting {zip_file_name}...")
        zip_ref.extractall(target_path)

    # Optionally remove the zip file
    if remove_zip:
        os.remove(data_path / zip_file_name)

    return target_path


def explore_directory(directory_path: str):
    """
    Explores a directory and prints the number of subdirectories and images in each.

    Args:
    directory_path (str): Path to the directory to explore.
    """
    for dirpath, dirnames, filenames in os.walk(directory_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def calculate_accuracy(true_labels: torch.Tensor, predicted_labels: torch.Tensor) -> float:
    """
    Calculates the accuracy of predictions compared to true labels.

    Args:
    true_labels (torch.Tensor): Ground truth labels.
    predicted_labels (torch.Tensor): Predicted labels by the model.

    Returns:
    float: Accuracy as a percentage.
    """
    correct_predictions = torch.eq(true_labels, predicted_labels).sum().item()
    accuracy = (correct_predictions / len(predicted_labels)) * 100
    return accuracy


def measure_training_time(start_time: float, end_time: float, device: str = None) -> float:
    """
    Measures and prints the training time.

    Args:
    start_time (float): Start time of the training.
    end_time (float): End time of the training.
    device (str, optional): Computing device used.

    Returns:
    float: Total training time in seconds.
    """
    total_time = end_time - start_time
    device_info = f" on {device}" if device else ""
    print(f"\nTraining time{device_info}: {total_time:.3f} seconds")
    return total_time


def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots the decision boundary of a model on a 2D dataset.

    Args:
    model (nn.Module): Trained PyTorch model.
    X (torch.Tensor): Input features.
    y (torch.Tensor): Target labels.
    """
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Place everything in the CPU.
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_training_data(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training and test data, and optionally the model's predictions.

    Args:
    train_data: Training data features.
    train_labels: Training data labels.
    test_data: Test data features.
    test_labels: Test data labels.
    predictions (optional): Model's predictions on the test data.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # If predictions are provided, plot them
    if predictions is not None:
        # Plot the predictions in red (on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    # Add labels and title
    plt.xlabel("Features")
    plt.ylabel("Labels")
    plt.title("Training and Test Data with Predictions")

    # Show the plot
    plt.show()


def plot_loss_and_accuracy_curves(results: dict):
    """
    Plots loss and accuracy curves from a training session.

    Args:
    results (dict): Dictionary containing loss and accuracy metrics.
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))

    # Plotting Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plotting Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def predict_and_plot_image(model: nn.Module, image_path: str, class_names: List[str] = None, transform=None, device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts the class of an image using a trained model and plots it.

    Args:
    model (nn.Module): Trained PyTorch model.
    image_path (str): Path to the image.
    class_names (List[str], optional): Class names for prediction.
    transform (optional): Transformations to be applied to the image.
    device (torch.device, optional): Device to perform computation.

    Example usage:
    predict_and_plot_image(model, "path/to/image.jpg", ["class1", "class2"], transform, device)
    """
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def to_2tuple(t):
    """
    Converts an integer or iterable to a tuple of two elements.

    Args:
    t (int or iterable): An integer or iterable to be converted.

    Returns:
    tuple: A tuple of two elements.
    """
    return (t, t) if isinstance(t, int) else t



#    chest_xray -> Train-> Normal    -> 1342
#                       -> Pneumonia -> 3876
#                  Val  -> Normal    ->  9
#                       -> Pneumonia ->  9
#                 Test  -> Normal    ->  235
#                       -> Pneumonia ->  391
#      test  ->  Normal ->   235
#            ->  Pneumonai -> 391
#
#
#