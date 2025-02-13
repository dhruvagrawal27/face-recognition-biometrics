import face_recognition as fr
import cv2
import numpy as np
import os
from typing import List, Tuple

def setup_paths(train_dir: str = "./train/", test_image: str = "./test/family.jpg") -> Tuple[str, str]:
    """Set up and validate directory paths."""
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(os.path.dirname(test_image)):
        os.makedirs(os.path.dirname(test_image))
    return train_dir, test_image

def load_known_faces(train_dir: str) -> Tuple[List[str], List[np.ndarray]]:
    """Load and encode known faces from training directory."""
    known_names = []
    known_encodings = []
    
    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        return known_names, known_encodings
    
    images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images in training directory")
    
    for image_file in images:
        image_path = os.path.join(train_dir, image_file)
        try:
            image = fr.load_image_file(image_path)
            face_encodings = fr.face_encodings(image)
            
            if face_encodings:
                known_encodings.append(face_encodings[0])
                name = os.path.splitext(image_file)[0].capitalize()
                known_names.append(name)
                print(f"Successfully processed: {name}")
            else:
                print(f"No faces found in {image_file}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            
    return known_names, known_encodings

def process_test_image(image_path: str, known_names: List[str], known_encodings: List[np.ndarray]) -> np.ndarray:
    """Process test image and draw face recognition results."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces and get encodings
    face_locations = fr.face_locations(rgb_image)
    face_encodings = fr.face_encodings(rgb_image, face_locations)
    
    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        if True in matches:
            face_distances = fr.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        
        # Draw rectangle and name
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return image

def main():
    """Main function to run the face recognition program."""
    try:
        # Setup paths
        train_dir, test_image = setup_paths()
        
        # Load known faces
        known_names, known_encodings = load_known_faces(train_dir)
        if not known_names:
            print("No training data available. Please add images to the training directory.")
            return
        
        # Process test image
        result_image = process_test_image(test_image, known_names, known_encodings)
        
        # Save and display results
        output_path = "./familyoutput.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Results saved to {output_path}")
        
        cv2.imshow("Face Recognition Results", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()