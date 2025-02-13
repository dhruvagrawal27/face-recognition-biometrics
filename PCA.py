import face_recognition as fr
import cv2
import numpy as np
import os
from typing import List, Tuple
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def setup_paths(train_dir: str = "./train/", test_image: str = "./test/test.jpg") -> Tuple[str, str]:
    """Set up and validate directory paths."""
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(os.path.dirname(test_image)):
        os.makedirs(os.path.dirname(test_image))
    return train_dir, test_image

def load_known_faces(train_dir: str) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Load and encode known faces from training directory."""
    known_names = []
    known_encodings = []
    face_images = []  # Store face images for PCA
    
    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        return known_names, known_encodings, face_images
    
    images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images in training directory")
    
    for image_file in images:
        image_path = os.path.join(train_dir, image_file)
        try:
            image = fr.load_image_file(image_path)
            face_locations = fr.face_locations(image)
            face_encodings = fr.face_encodings(image)
            
            if face_encodings:
                known_encodings.append(face_encodings[0])
                name = os.path.splitext(image_file)[0].capitalize()
                known_names.append(name)
                
                # Extract face region for PCA
                top, right, bottom, left = face_locations[0]
                face_image = image[top:bottom, left:right]
                face_image = cv2.resize(face_image, (64, 64))  # Standardize size
                face_images.append(face_image.flatten())  # Flatten for PCA
                
                print(f"Successfully processed: {name}")
            else:
                print(f"No faces found in {image_file}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            
    return known_names, known_encodings, face_images

def perform_pca_analysis(face_images: List[np.ndarray]):
    """Perform PCA analysis on face images and visualize results."""
    print("\nPerforming PCA Analysis:")
    
    # Convert list to numpy array
    X = np.array(face_images)
    print("\nStep 1: Data Standardization Details")
    print(f"Original data shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Calculate and print mean and standard deviation
    X_mean = X.mean(axis=0)
    X_std_dev = X.std(axis=0)
    print(f"Mean range: [{X_mean.min():.2f}, {X_mean.max():.2f}]")
    print(f"Std Dev range: [{X_std_dev.min():.2f}, {X_std_dev.max():.2f}]")
    
    # Standardize the data
    X_std = (X - X_mean) / X_std_dev
    print("\nAfter standardization:")
    print(f"Standardized data shape: {X_std.shape}")
    print(f"Standardized mean range: [{X_std.mean(axis=0).min():.2f}, {X_std.mean(axis=0).max():.2f}]")
    print(f"Standardized std range: [{X_std.std(axis=0).min():.2f}, {X_std.std(axis=0).max():.2f}]")
    
    # Calculate number of components
    n_components = min(len(face_images), X_std.shape[1])
    print(f"\nStep 2: PCA Configuration")
    print(f"Number of components selected: {n_components}")
    print(f"Maximum possible components: {min(X_std.shape)}")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    
    print(f"\nStep 3: PCA Transformation Results")
    print(f"Transformed data shape: {X_pca.shape}")
    print(f"Components shape: {pca.components_.shape}")
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("\nDetailed Variance Analysis:")
    for i, (var_ratio, cum_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
        print(f"PC{i+1}:")
        print(f"  Explained variance ratio: {var_ratio:.4f}")
        print(f"  Cumulative variance ratio: {cum_ratio:.4f}")
        print(f"  Eigenvalue: {pca.explained_variance_[i]:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Explained variance ratio
    plt.subplot(121)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Component')
    
    # Plot 2: Cumulative explained variance
    plt.subplot(122)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.tight_layout()
    plt.savefig('pca_analysis.png')
    plt.close()
    
    # Visualize eigenfaces
    if len(face_images) > 0:
        n_cols = min(n_components, 5)  # Show maximum of 5 eigenfaces
        plt.figure(figsize=(3*n_cols, 3))
        
        # Calculate the proper reshape dimensions
        h = w = int(np.sqrt(X.shape[1] // 3))  # Divide by 3 for RGB channels
        
        for i in range(n_cols):
            plt.subplot(1, n_cols, i + 1)
            # Reshape considering RGB channels
            eigenface = pca.components_[i].reshape(h, w, 3)
            # Convert to grayscale for visualization
            eigenface_gray = cv2.cvtColor(((eigenface - eigenface.min()) * 255 / (eigenface.max() - eigenface.min())).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            plt.imshow(eigenface_gray, cmap='gray')
            plt.title(f'Eigenface {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('eigenfaces.png')
        plt.close()
    
    return X_pca, pca

def process_test_image(image_path: str, known_names: List[str], known_encodings: List[np.ndarray]) -> np.ndarray:
    """Process test image and draw face recognition results."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_image)
    face_encodings = fr.face_encodings(rgb_image, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        if True in matches:
            face_distances = fr.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return image

def main():
    """Main function to run the face recognition program with PCA analysis."""
    try:
        # Setup paths
        train_dir, test_image = setup_paths()
        
        # Load known faces
        known_names, known_encodings, face_images = load_known_faces(train_dir)
        if not known_names:
            print("No training data available. Please add images to the training directory.")
            return
        
        # Perform PCA analysis
        if len(face_images) > 0:
            print("\nStarting PCA analysis...")
            X_pca, pca = perform_pca_analysis(face_images)
            print("PCA analysis completed. Check 'pca_analysis.png' and 'eigenfaces.png' for visualizations.")
        
        # Process test image
        result_image = process_test_image(test_image, known_names, known_encodings)
        
        # Save and display results
        output_path = "./output.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"\nFace recognition results saved to {output_path}")
        
        cv2.imshow("Face Recognition Results", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()