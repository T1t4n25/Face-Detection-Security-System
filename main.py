import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                           QTableWidgetItem, QHeaderView, QDialog, QFileDialog,
                           QInputDialog, QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage

# Import the face recognition module
from face_recognition_module import FaceRecognitionSystem


class FacePreviewWidget(QWidget):
    """Widget to display a detected face with name and options"""
    def __init__(self, face_img, name, parent=None):
        super().__init__(parent)
        self.face_img = face_img
        self.name = name
        self.parent_app = parent
        
        layout = QVBoxLayout()  # Changed to vertical layout for better space use
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)
        
        # Face image
        h, w, c = face_img.shape
        q_img = QImage(face_img.data, w, h, w*c, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        self.face_label = QLabel()
        self.face_label.setPixmap(pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio))
        self.face_label.setFixedSize(80, 80)
        self.face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.face_label)
        
        # Name label
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-size: 12px;")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)
        
        # If the face is unknown, add a button to name it
        if name == "Unknown" and parent:
            name_btn = QPushButton("Name Face")
            name_btn.clicked.connect(self.name_face)
            layout.addWidget(name_btn)
            
    def name_face(self):
        """Handle naming an unknown face"""
        if not isinstance(self.parent_app, FaceRecognitionApp):
            return
        
        # Store a copy of the face image
        snapshot = self.face_img.copy()
        
        # Stop both timers temporarily
        was_camera_active = self.parent_app.camera_active
        if was_camera_active:
            self.parent_app.timer.stop()
            self.parent_app.face_detection_timer.stop()
        
        name, ok = QInputDialog.getText(self, "Name Face", "Enter name for this face:")
        
        if ok and name:
            # Add face to known faces using the parent app's face recognition system
            if self.parent_app.face_system.add_face(snapshot, name):
                # Store the face image and set default access rights
                self.parent_app.saved_face_images[name] = snapshot
                self.parent_app.face_access_rights[name] = False
                
                self.parent_app.update_face_table()
                self.name_label.setText(name)
                self.parent_app.set_status_message(f"Face saved: {name}")
            else:
                self.parent_app.set_status_message("Error: Could not encode face")
        
        # Resume both timers if they were active
        if was_camera_active:
            self.parent_app.timer.start(30)  # Restore original camera update rate
            self.parent_app.face_detection_timer.start(500)  # Restore face detection rate


class FaceImageDialog(QDialog):
    """Dialog to display a face image"""
    def __init__(self, face_img, name, has_access, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.name = name
        self.has_access = has_access
        self.setWindowTitle(f"Face: {name}")
        self.setMinimumSize(300, 350)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Face image
        h, w, c = face_img.shape
        q_img = QImage(face_img.data, w, h, w*c, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        face_label = QLabel()
        face_label.setPixmap(pixmap.scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))
        face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(face_label)
        
        # Name label
        name_label = QLabel(f"Name: {name}")
        name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)
        
        # Access status label
        self.access_label = QLabel(f"Access: {'Granted' if has_access else 'Denied'}")
        self.access_label.setStyleSheet(f"font-size: 14px; color: {'green' if has_access else 'red'};")
        self.access_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.access_label)
        
        # Toggle access button
        access_btn = QPushButton("Revoke Access" if has_access else "Grant Access")
        access_btn.clicked.connect(self.toggle_access)
        layout.addWidget(access_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def toggle_access(self):
        """Toggle access for this face"""
        if not isinstance(self.parent_app, FaceRecognitionApp):
            return
            
        # Update access in the parent app
        self.has_access = not self.has_access
        self.parent_app.update_face_access(self.name, self.has_access)
        
        # Update UI
        self.access_label.setText(f"Access: {'Granted' if self.has_access else 'Denied'}")
        self.access_label.setStyleSheet(f"font-size: 14px; color: {'green' if self.has_access else 'red'};")
        
        sender = self.sender()
        if sender:
            sender.setText("Revoke Access" if self.has_access else "Grant Access")


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Recognition Security System")
        self.setMinimumSize(1000, 650)  # Increased width for 3-column layout
        
        # Initialize face recognition system (modular approach)
        self.face_system = FaceRecognitionSystem()
        
        # Camera variables
        self.camera_active = False
        self.cap = None
        self.frame = None
        self.current_face_widgets = []
        
        # Storage for face images and access rights
        self.saved_face_images = {}  # Dict mapping name -> face image
        self.face_access_rights = {}  # Dict mapping name -> access boolean
        
        # Initialize UI
        self.setup_ui()
        
        # Load stylesheet
        try:
            with open("themes\\Aqua.qss", "r") as file:
                self.setStyleSheet(file.read())
        except:
            pass  # Continue without stylesheet if file not found
            
        # Timer for updating camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        
        # Add face detection timer with slower update rate
        self.face_detection_timer = QTimer()
        self.face_detection_timer.timeout.connect(self.detect_faces)
        self.last_face_data = []  # Store last detected faces
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(250)
        
        # Camera controls section
        camera_control_label = QLabel("Camera Controls")
        camera_control_label.setObjectName("sectionLabel")
        left_layout.addWidget(camera_control_label)
        
        self.toggle_camera_btn = QPushButton("Start Camera")
        self.toggle_camera_btn.clicked.connect(self.toggle_camera)
        left_layout.addWidget(self.toggle_camera_btn)
        
        self.take_screenshot_btn = QPushButton("Take Screenshot")
        self.take_screenshot_btn.clicked.connect(self.take_screenshot)
        self.take_screenshot_btn.setEnabled(False)
        left_layout.addWidget(self.take_screenshot_btn)
        
        self.add_face_btn = QPushButton("Add Current Face")
        self.add_face_btn.clicked.connect(self.add_face)
        self.add_face_btn.setEnabled(False)
        left_layout.addWidget(self.add_face_btn)
        
        # Access status section
        access_label = QLabel("Access Status:")
        left_layout.addWidget(access_label)
        
        self.access_status = QLabel("Camera Off")
        self.access_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.access_status.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(self.access_status)
        
        # Known faces section
        saved_faces_label = QLabel("Saved Faces")
        saved_faces_label.setObjectName("sectionLabel")
        left_layout.addWidget(saved_faces_label)
        
        self.face_table = QTableWidget(0, 3)  # Added a column for access status
        self.face_table.setHorizontalHeaderLabels(["Name", "Access", "Actions"])
        self.face_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.face_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.face_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.face_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.face_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        left_layout.addWidget(self.face_table)
        
        self.delete_face_btn = QPushButton("Delete Selected Face")
        self.delete_face_btn.clicked.connect(self.delete_face)
        left_layout.addWidget(self.delete_face_btn)
        
        # Middle panel - camera feed
        middle_panel = QWidget()
        middle_layout = QVBoxLayout()
        middle_panel.setLayout(middle_layout)
        
        # Camera feed label
        self.camera_feed_label = QLabel()
        self.camera_feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed_label.setMinimumSize(500, 400)
        self.camera_feed_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        middle_layout.addWidget(self.camera_feed_label, 1)  # Give stretch factor
        
        # Status message (fixed at bottom)
        self.status_label = QLabel("Camera Off")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMinimumHeight(30)
        middle_layout.addWidget(self.status_label)
        
        # Right panel - detected faces
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(180)
        right_panel.setMaximumWidth(200)
        
        # Current faces detected section
        current_faces_label = QLabel("Detected Faces")
        current_faces_label.setObjectName("sectionLabel")
        right_layout.addWidget(current_faces_label)
        
        # ScrollArea for detected faces
        faces_scroll_area = QScrollArea()
        faces_scroll_area.setWidgetResizable(True)
        faces_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        faces_scroll_widget = QWidget()
        self.current_faces_layout = QVBoxLayout(faces_scroll_widget)
        self.current_faces_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.current_faces_layout.setSpacing(10)
        faces_scroll_area.setWidget(faces_scroll_widget)
        right_layout.addWidget(faces_scroll_area)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(middle_panel, 1)  # 1 is stretch factor to make middle panel expand
        main_layout.addWidget(right_panel)
        
    def set_status_message(self, message):
        """Set the status message in the UI"""
        self.status_label.setText(message)
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_active:
            # Stop camera
            self.camera_active = False
            if self.cap:
                self.cap.release()
            self.timer.stop()
            self.face_detection_timer.stop()  # Stop face detection timer
            self.toggle_camera_btn.setText("Start Camera")
            self.set_status_message("Camera Off")
            self.access_status.setText("Camera Off")
            self.take_screenshot_btn.setEnabled(False)
            self.add_face_btn.setEnabled(False)
            
            # Clear camera feed display
            self.camera_feed_label.clear()
            self.camera_feed_label.setText("Camera Off")
            
            # Clear current faces
            self.clear_current_faces()
        else:
            # Start camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.set_status_message("Error: Could not open camera")
                return
                
            self.camera_active = True
            self.toggle_camera_btn.setText("Stop Camera")
            self.timer.start(30)  # Update camera feed every 30ms (faster)
            self.face_detection_timer.start(1000)  # Face detection every 500ms
            self.take_screenshot_btn.setEnabled(True)
            self.add_face_btn.setEnabled(True)
            self.set_status_message("Camera On")
    
    def detect_faces(self):
        """Separate method for face detection"""
        if not self.camera_active or not self.cap or self.frame is None:
            return
            
        # Detect faces using the face recognition system
        self.last_face_data, frame_with_boxes = self.face_system.detect_faces(self.frame)
        
        # Update access status
        self.update_access_status(self.last_face_data)
        
        # Update current faces display
        self.update_current_faces(self.last_face_data)

    def update_camera(self):
        """Update camera feed only"""
        if not self.camera_active or not self.cap:
            return
            
        ret, self.frame = self.cap.read()
        if not ret:
            self.set_status_message("Error: Could not read frame")
            return
        
        # Use the last detected faces to draw boxes
        if hasattr(self, 'last_face_data') and self.last_face_data:
            frame_with_boxes = self.face_system.draw_boxes(self.frame, self.last_face_data)
        else:
            frame_with_boxes = self.frame
        
        # Update the camera feed display
        h, w, c = frame_with_boxes.shape
        bytes_per_line = w * c
        q_img = QImage(frame_with_boxes.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        self.camera_feed_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.camera_feed_label.width(), 
            self.camera_feed_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))
        
    def update_access_status(self, face_data):
        """Update the access status based on detected faces"""
        if not face_data:
            self.access_status.setText("No Face Detected")
            self.access_status.setStyleSheet("color: black; font-size: 16px; font-weight: bold;")
            return
        
        # Check if any recognized face has access
        access_granted = False
        for face in face_data:
            if face['name'] != "Unknown" and self.face_access_rights.get(face['name'], False):
                access_granted = True
                break
        
        if access_granted:
            self.access_status.setText("Access Granted")
            self.access_status.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
        else:
            self.access_status.setText("Access Denied")
            self.access_status.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
    
    def clear_current_faces(self):
        """Clear the current faces display"""
        # Remove all widgets from the layout
        while self.current_faces_layout.count():
            child = self.current_faces_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.current_face_widgets = []
    
    def update_current_faces(self, face_data):
        """Update the display of currently detected faces"""
        # Clear previous faces
        self.clear_current_faces()
        
        # Add each detected face to the UI
        for face in face_data:
            face_widget = FacePreviewWidget(face['image'], face['name'], self)
            self.current_faces_layout.addWidget(face_widget)
            self.current_face_widgets.append(face_widget)
    
    def update_face_table(self):
        """Update the table of saved faces"""
        names = self.face_system.get_face_names()
        self.face_table.setRowCount(len(names))
        
        for i, name in enumerate(names):
            # Initialize access rights if not present
            if name not in self.face_access_rights:
                self.face_access_rights[name] = False
            
            # Add name
            name_item = QTableWidgetItem(name)
            self.face_table.setItem(i, 0, name_item)
            
            # Create access button widget
            access_widget = QWidget()
            access_layout = QHBoxLayout()
            access_layout.setContentsMargins(2, 2, 2, 2)
            access_widget.setLayout(access_layout)
            
            has_access = self.face_access_rights.get(name, False)
            access_btn = QPushButton("Granted" if has_access else "Denied")
            access_btn.setStyleSheet(
                "color: green;" if has_access else "color: red;"
            )
            access_btn.clicked.connect(lambda checked, n=name: self.toggle_face_access(n))
            access_layout.addWidget(access_btn)
            
            self.face_table.setCellWidget(i, 1, access_widget)
            
            # Add actions in a widget
            action_widget = QWidget()
            action_layout = QHBoxLayout()
            action_layout.setContentsMargins(2, 2, 2, 2)
            action_widget.setLayout(action_layout)
            
            # View button
            view_btn = QPushButton("View")
            view_btn.clicked.connect(lambda checked, idx=i: self.view_face(idx))
            action_layout.addWidget(view_btn)
            
            self.face_table.setCellWidget(i, 2, action_widget)
    
    def toggle_face_access(self, name):
        """Toggle access rights for a face directly from the table"""
        if name in self.face_access_rights:
            # Toggle the access
            new_access = not self.face_access_rights[name]
            self.face_access_rights[name] = new_access
            
            # Update the table display
            self.update_face_table()
    
    def update_face_access(self, name, has_access):
        """Update access rights for a face"""
        if name in self.face_access_rights:
            self.face_access_rights[name] = has_access
            
            # Update the table display
            names = self.face_system.get_face_names()
            if name in names:
                row = names.index(name)
                access_item = QTableWidgetItem("Granted" if has_access else "Denied")
                access_item.setForeground(Qt.GlobalColor.green if has_access else Qt.GlobalColor.red)
                self.face_table.setItem(row, 1, access_item)
    
    def view_face(self, index):
        """Show a dialog with the face details"""
        name = self.face_system.get_face_names()[index]
        has_access = self.face_access_rights.get(name, False)
        
        # Use saved face image if available, otherwise create placeholder
        if name in self.saved_face_images:
            face_img = self.saved_face_images[name]
        else:
            face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Show the face dialog
        dialog = FaceImageDialog(face_img, name, has_access, self)
        dialog.exec()
    
    def delete_face(self):
        """Delete the selected face"""
        selected_rows = self.face_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        name = self.face_system.get_face_names()[row]
        
        if self.face_system.remove_face(row):
            # Also remove from our local storage
            if name in self.saved_face_images:
                del self.saved_face_images[name]
            if name in self.face_access_rights:
                del self.face_access_rights[name]
                
            self.update_face_table()
            self.set_status_message(f"Face deleted")
        else:
            self.set_status_message("Error: Could not delete face")
    
    def add_face(self):
        """Add the current face from the camera feed"""
        if not self.camera_active or self.frame is None:
            return
            
        # Use the face recognition system to detect faces
        face_data, _ = self.face_system.detect_faces(self.frame)
        
        if not face_data:
            self.set_status_message("No faces detected")
            return
        
        # Use the first detected face
        face = face_data[0]
        
        # Stop camera updates temporarily
        was_camera_active = self.camera_active
        if was_camera_active:
            self.timer.stop()
        
        # Prompt for the name
        name, ok = QInputDialog.getText(self, "Name Face", "Enter name for this face:")
        
        if ok and name:
            # Store a copy of the face image
            face_img = face['image'].copy()
            
            # Add the face to the known faces
            if self.face_system.add_face(face_img, name):
                # Store the face image and set default access rights
                self.saved_face_images[name] = face_img
                self.face_access_rights[name] = False
                
                self.update_face_table()
                self.set_status_message(f"Face saved: {name}")
            else:
                self.set_status_message("Error: Could not encode face")
        
        # Resume camera updates if they were active
        if was_camera_active:
            self.timer.start(100)
    
    def take_screenshot(self):
        """Save the current frame as an image"""
        if not self.camera_active or self.frame is None:
            return
            
        # Get current date/time for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        
        # Use file dialog to select where to save
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", filename, "JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filepath:
            cv2.imwrite(filepath, self.frame)
            self.set_status_message(f"Screenshot saved: {filepath}")
    
    def closeEvent(self, event):
        """Clean up resources when the app is closed"""
        if self.camera_active and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())