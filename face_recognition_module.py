import cv2
import face_recognition


class FaceRecognitionSystem:
    """
    A class that handles face recognition functionality independent of the GUI.
    This follows the modular approach by separating the face recognition logic
    from the presentation layer.
    """
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        # Add new variable to store face data
        self.stored_face_data = {}  # Dictionary to store face data by name
    
    def detect_faces(self, frame):
        """
        Detect faces in a given frame
        
        Args:
            frame: A BGR image frame from the camera
            
        Returns:
            A tuple of (face_locations, face_encodings, frame_with_boxes)
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert from BGR to RGB (face_recognition uses RGB)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Create a copy of the frame for drawing on
        frame_with_boxes = frame.copy()
        
        # Scale back face coordinates and draw boxes
        face_data = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back face coordinates
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Check if the face is known
            name = self.identify_face(face_encoding)
            
            # Draw rectangle around face
            cv2.rectangle(frame_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label with name
            cv2.rectangle(frame_with_boxes, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_with_boxes, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            # Extract face image
            face_img = frame[top:bottom, left:right].copy()
            
            # Store face data in the dictionary using name as key
            if name != "Unknown":
                self.stored_face_data[name] = {
                    'location': (top, right, bottom, left),
                    'name': name,
                    'image': face_img,
                    'encoding': face_encoding
                }
            
            face_data.append({
                'location': (top, right, bottom, left),
                'name': name,
                'image': face_img,
                'encoding': face_encoding
            })
        
        return face_data, frame_with_boxes
    
    def identify_face(self, face_encoding):
        """
        Identify a face by comparing it with known faces
        
        Args:
            face_encoding: The encoding of the face to identify
            
        Returns:
            The name of the face if known, 'Unknown' otherwise
        """
        name = "Unknown"
        
        if len(self.known_face_encodings) > 0:
            # Compare the face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, 0.5)
            
            # If a match was found, use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
        
        return name
    
    def add_face(self, face_image, name):
        """
        Add a new face to the known faces
        
        Args:
            face_image: The image of the face to add
            name: The name to associate with the face
            
        Returns:
            True if the face was added successfully, False otherwise
        """
        # Convert face to RGB for face_recognition
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)
        
        if len(face_encodings) == 0:
            return False
            
        # Add the face encoding and name
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)
        
        return True
    
    def remove_face(self, index):
        """
        Remove a face from the known faces
        
        Args:
            index: The index of the face to remove
            
        Returns:
            True if the face was removed successfully, False otherwise
        """
        if 0 <= index < len(self.known_face_names):
            del self.known_face_names[index]
            del self.known_face_encodings[index]
            return True
        return False
    
    def get_face_count(self):
        """
        Get the number of known faces
        
        Returns:
            The number of known faces
        """
        return len(self.known_face_names)
    
    def get_face_names(self):
        """
        Get the names of all known faces
        
        Returns:
            A list of names of all known faces
        """
        return self.known_face_names.copy()
    
    def encode_face_from_image(self, image):
        """
        Encode a face from an image
        
        Args:
            image: The image containing the face
            
        Returns:
            The face encoding if a face was found, None otherwise
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        if len(face_locations) == 0:
            return None
            
        # Get the encoding of the first face
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return None
            
        return {
            'encoding': face_encodings[0],
            'location': face_locations[0]
        }
    
    def check_access(self, face_data):
        """
        Check if any of the detected faces have access
        
        Args:
            face_data: A list of dictionaries containing face data
            
        Returns:
            True if any face has access, False otherwise
        """
        for face in face_data:
            if face['name'] != "Unknown":
                return True
        return False
    
    # Add new method to get stored face data
    def get_face_image(self, index):
        """
        Get the stored face image for a given index
        
        Args:
            index: The index of the face in known_face_names
            
        Returns:
            The face image if found, None otherwise
        """
        if 0 <= index < len(self.known_face_names):
            name = self.known_face_names[index]
            if name in self.stored_face_data:
                return self.stored_face_data[name]['image']
        return None
    
    def draw_boxes(self, frame, face_data):
        """Draw boxes and labels on the frame"""
        frame_with_boxes = frame.copy()
        
        for face in face_data:
            top, right, bottom, left = face['location']
            name = face['name']
            
            # Draw rectangle around face
            cv2.rectangle(frame_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label with name
            cv2.rectangle(frame_with_boxes, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_with_boxes, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        return frame_with_boxes