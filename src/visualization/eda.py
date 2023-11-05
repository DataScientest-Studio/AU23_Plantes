import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

def load_and_preprocess_data(base_path):
    """
    Load and preprocess data from a given path.
    
    Parameters:
    - base_path : str : The base directory path where the classes are located.

    Returns:
    - classes : list : List of class names.
    - image_paths : dict : Dictionary mapping class names to a list of image paths.
    - image_shapes : dict : Dictionary mapping class names to a list of image shapes.
    - image_weights : dict : Dictionary mapping class names to a list of image weights.
    - class_distribution : dict : Dictionary mapping class names to the number of images.
    - total_images : int : The total number of images.
    """
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d not in ['.DS_Store', '.ipynb_checkpoints']]
    image_paths = {}
    image_shapes = {}
    image_weights = {}  # Dictionary to store image weights
    class_distribution = {}
    total_images = 0

    for class_name in classes:
        class_folder_path = os.path.join(base_path, class_name)
        num_images = 0
        total_weight = 0  # Total weight for this class

        image_paths[class_name] = []
        image_shapes[class_name] = []
        image_weights[class_name] = []

        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image_shape = image.shape
            image_size = image_shape[0] * image_shape[1]  # Calculate the image size as weight

            image_paths[class_name].append(image_path)
            image_shapes[class_name].append(image_shape)
            image_weights[class_name].append(round(image_size / 1024, 2)) # To put the result in Ko

            num_images += 1
            total_weight += image_size

        class_distribution[class_name] = num_images
        total_images += num_images

    return classes, image_paths, image_shapes, image_weights, class_distribution, total_images

def create_dataframe(classes, image_paths, image_shapes, image_weights):
    """
    Create a DataFrame to hold various details about images for each class, including image weights.
    
    Parameters:
    - classes : list : List of class names.
    - image_paths : dict : Dictionary mapping class names to a list of image paths.
    - image_shapes : dict : Dictionary mapping class names to a list of image shapes.
    - image_weights : dict : Dictionary mapping class names to a list of image weights.

    Returns:
    - data : pd.DataFrame : DataFrame containing the image details.
    """
    df_data = {
        'Class': [],
        'Filename': [],
        'Path': [],
        'Height': [],
        'Width': [],
        'Shape': [],
        'Weight': []  # New column for image weights
    }

    for class_name in classes:
        for path, shape, weight in zip(image_paths[class_name], image_shapes[class_name], image_weights[class_name]):
            df_data['Class'].append(class_name)
            df_data['Filename'].append(os.path.basename(path))
            df_data['Path'].append(path)
            df_data['Height'].append(shape[0])
            df_data['Width'].append(shape[1])
            df_data['Shape'].append(shape)
            df_data['Weight'].append(weight)  # Include the image weight

    data = pd.DataFrame(df_data)
    return data
def save_dataframe_to_csv(data, filename, folder='Files'):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    - data : pd.DataFrame : DataFrame containing the image details.
    - filename : str : The filename for the saved CSV file.
    - folder : str : (optional) The folder where the CSV will be saved. Default is 'Files'.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    full_path = os.path.join(folder, filename)
    data.to_csv(full_path, index=False)
    print(f"DataFrame saved as {full_path}")

def plot_class_distribution(class_distribution, total_images):
    """
    Plots the distribution of classes in the dataset.
    
    Parameters:
    - class_distribution: Dictionary containing the distribution of classes
    - total_images: Total number of images in the dataset
    """
    n_classes = len(class_distribution.keys())
    sorted_classes = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: item[1], reverse=True)}
    summer_cmap = plt.cm.get_cmap('summer')
    colors = np.linspace(0.3, 1, n_classes)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_classes.keys(), sorted_classes.values(), color=summer_cmap(colors))
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(sorted_classes.keys(), rotation=90)
    ax.set_ylabel("Number of images")

    for bar, value in zip(bars, sorted_classes.values()):
        percentage = f"{(value / total_images) * 100:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.4, bar.get_height() + 5, percentage)

    plt.show()

def plot_sample_images(data, selected_classes=None, grid_shape=(4, 3), random_seed=None):
    """
    Plots sample images for each class.
    
    Parameters:
    - data: DataFrame containing image data
    - selected_classes: List of classes to plot, defaults to all classes
    - grid_shape: Tuple indicating the shape of the grid for plotting
    - random_seed: Optional random seed for reproducibility
    """
    if selected_classes is None:
        selected_classes = data['Class'].unique()

    if random_seed is not None:
        np.random.seed(random_seed)

    fig, axes = plt.subplots(*grid_shape, figsize=(15, 20))
    axes = axes.flatten()

    for ax, class_name in zip(axes, selected_classes):
        df_filtered = data[data['Class'] == class_name]
        random_path = np.random.choice(list(df_filtered['Path']))
        img = cv2.imread(random_path)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(class_name)
        ax.axis('off')

    plt.show()

def plot_specific_images(data, image_class_pairs):
    """
    Plots specific images based on given filename and class pairs.
    
    Parameters:
    - data: DataFrame containing image data
    - image_class_pairs: List of tuples, each containing a filename and corresponding class name
    """
    fig, axes = plt.subplots(1, len(image_class_pairs), figsize=(15, 5))
    if len(image_class_pairs) == 1:
        axes = [axes]
    for ax, (filename, class_name) in zip(axes, image_class_pairs):
        df_filtered = data[(data['Class'] == class_name) & (data['Filename'] == filename)]
        image_path = df_filtered['Path'].iloc[0]
        img = cv2.imread(image_path)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{class_name}\n{filename}")
        ax.axis('off')
    plt.show()
def plot_image_dimensions(data):
    """
    Plot the distribution of image dimensions (heights and widths).
    
    Parameters:
    - data : pd.DataFrame : DataFrame containing the image details.
    """
    heights = data['Height']
    widths = data['Width']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].hist(heights, bins=20, color='seagreen', edgecolor='black')
    axes[0].set_title('Image height distribution')
    axes[0].set_xlabel('Height (pixels)')
    axes[0].set_ylabel('Number of images')

    axes[1].hist(widths, bins=20, color='mediumseagreen', edgecolor='black')
    axes[1].set_title('Image width distribution')
    axes[1].set_xlabel('Width (pixels)')
    axes[1].set_ylabel('Number of images')

    plt.tight_layout()
    plt.show()

def plot_image_ratio(data):
    """
    Plot the distribution of the image aspect ratios.
    
    Parameters:
    - data : pd.DataFrame : DataFrame containing the image details.
    """
    ratio = data['Height'] / data['Width']
    plt.figure(figsize=(6, 6))
    plt.hist(ratio, bins=20, color='green', edgecolor='black')
    plt.title('Distribution of the Images ratio')
    plt.xlabel('Ratio (Height/Width)')
    plt.ylabel('Number of images')
    plt.show()

def display_image_distribution(data_df):
    """
    Display the distribution of images in RGB and RGBA and the class distribution for RGBA format.
    
    Parameters:
    - data_df : pd.DataFrame : DataFrame containing the image details.
    """
    data_df['Channels'] = data_df['Shape'].apply(lambda x: x[2] if len(x) == 3 else 4 if len(x) == 4 else 'Other')

    rgba_count = data_df[data_df["Channels"] == 4].shape[0]
    rgb_count = data_df[data_df["Channels"] == 3].shape[0]
    values = [rgb_count, rgba_count]
    labels = ["RGB", "RGBA"]
    colors = ['yellowgreen', 'lightcoral']

    rgba_data = data_df[data_df["Channels"] == 4]
    class_distribution_rgba = rgba_data['Class'].value_counts()

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    axs[0].pie(values, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140)
    axs[0].axis('equal')  # Circular aspect
    axs[0].set_title("Distribution of images in RGB and RGBA")

    axs[1].bar(class_distribution_rgba.index, class_distribution_rgba.values, color=['seagreen', 'mediumseagreen'])
    axs[1].set_title("Class distribution within RGBA images")
    axs[1].set_ylabel("Number of images")
    axs[1].set_xlabel("Class")

    plt.tight_layout()
    plt.show()

def plot_median_weight(classes, data_df):
    """
    Generate a bar plot to visualize median weights per class.

    Parameters:
    classes (list): A list of class names present in the dataset.
    data_df (pandas.DataFrame): A DataFrame with columns 'Class' and 'Weight'
        containing class names and their corresponding weights.
    """

    median_weights_per_class = data_df.groupby('Class')['Weight'].median()

    n_classes = len(classes)
    summer_cmap = plt.colormaps.get_cmap('summer')
    colors = np.linspace(0.3, 1, n_classes)

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    median_weights_per_class.plot(kind='bar', color=summer_cmap(colors))
    plt.title('Median Weights per Class')
    plt.xlabel('Class')
    plt.ylabel('Median Weight (KB)')
    plt.xticks(rotation=45)

    plt.show()