import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

def load_and_preprocess_data(base_path):
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d != '.DS_Store']
    image_paths = {}
    image_shapes = {}
    class_distribution = {}
    total_images = 0

    for class_name in classes:
        class_folder_path = os.path.join(base_path, class_name)
        num_images = len(os.listdir(class_folder_path))
        total_images += num_images
        class_distribution[class_name] = num_images

        image_paths[class_name] = []
        image_shapes[class_name] = []

        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image_shape = image.shape

            image_paths[class_name].append(image_path)
            image_shapes[class_name].append(image_shape)

    return classes, image_paths, image_shapes, class_distribution, total_images

def create_dataframe(classes, image_paths, image_shapes):
    df_data = {
        'Class': [],
        'Filename': [],
        'Path': [],
        'Height': [],
        'Width': [],
        'Shape': []
    }

    for class_name in classes:
        for path, shape in zip(image_paths[class_name], image_shapes[class_name]):
            df_data['Class'].append(class_name)
            df_data['Filename'].append(os.path.basename(path))
            df_data['Path'].append(path)
            df_data['Height'].append(shape[0])
            df_data['Width'].append(shape[1])
            df_data['Shape'].append(shape)

    data = pd.DataFrame(df_data)
    return data

def save_dataframe_to_csv(data, filename, folder='Files'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    full_path = os.path.join(folder, filename)
    data.to_csv(full_path, index=False)
    print(f"DataFrame saved as {full_path}")

def plot_class_distribution(class_distribution, total_images):
    n_classes = len(class_distribution.keys())
    sorted_classes = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: item[1], reverse=True)}
    summer_cmap = plt.cm.get_cmap('summer')
    colors = np.linspace(0.3, 1, n_classes)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_classes.keys(), sorted_classes.values(), color=summer_cmap(colors))
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(sorted_classes.keys(), rotation=90)
    ax.set_ylabel("Nombre d'images")

    for bar, value in zip(bars, sorted_classes.values()):
        percentage = f"{(value / total_images) * 100:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.4, bar.get_height() + 5, percentage)

    plt.show()

def plot_sample_images(data, selected_classes=None, grid_shape=(4, 3), random_seed=None):
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
    heights = data['Height']
    widths = data['Width']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].hist(heights, bins=20, color='seagreen', edgecolor='black')
    axes[0].set_title('Distribution des Hauteurs des Images')
    axes[0].set_xlabel('Hauteur (pixels)')
    axes[0].set_ylabel('Nombre d\'images')

    axes[1].hist(widths, bins=20, color='mediumseagreen', edgecolor='black')
    axes[1].set_title('Distribution des Largeurs des Images')
    axes[1].set_xlabel('Largeur (pixels)')
    axes[1].set_ylabel('Nombre d\'images')

    plt.tight_layout()
    plt.show()

def plot_image_ratio(data):
    ratio = data['Height'] / data['Width']
    plt.figure(figsize=(6, 6))
    plt.hist(ratio, bins=20, color='green', edgecolor='black')
    plt.title('Distribution du ratio des Images')
    plt.xlabel('Ratio (Hauteur/Largeur)')
    plt.ylabel('Nombre d\'images')
    plt.show()
