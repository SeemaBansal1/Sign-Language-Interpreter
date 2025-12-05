"""
Data Collection Script for Sign Language Recognition
=====================================================
This script collects hand landmark data using MediaPipe and saves it to a CSV file.

Usage:
    1. Run the script: python collect_data.py
    2. Enter the sign label when prompted (e.g., "Hello", "Yes", "No")
    3. Press 'S' to save a sample when your hand is in position
    4. Press 'Q' to quit and save all data
    5. Press 'C' to change the current label

Controls:
    S - Save current hand pose as a sample
    C - Change the current sign label
    Q - Quit and save all data to CSV
"""

import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime


class SignDataCollector:
    def __init__(self, output_file="data.csv"):
        """Initialize the data collector with MediaPipe hands."""
        self.output_file = output_file
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.collected_data = []
        self.current_label = ""
        self.sample_count = {}
        
        # Load existing data if file exists
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing data from CSV if it exists."""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header
                for row in reader:
                    self.collected_data.append(row)
                    if row:
                        label = row[-1]  # Last column is label
                        self.sample_count[label] = self.sample_count.get(label, 0) + 1
            print(f"Loaded {len(self.collected_data)} existing samples")
            print(f"Labels: {self.sample_count}")
    
    def _create_header(self):
        """Create CSV header for 21 landmarks (x, y, z each)."""
        header = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")
        return header
    
    def _extract_landmarks(self, hand_landmarks):
        """Extract x, y, z coordinates from hand landmarks."""
        data = []
        for landmark in hand_landmarks.landmark:
            data.extend([landmark.x, landmark.y, landmark.z])
        return data
    
    def _save_to_csv(self):
        """Save all collected data to CSV file."""
        if not self.collected_data:
            print("No data to save!")
            return
        
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._create_header())
            writer.writerows(self.collected_data)
        
        print(f"\nData saved to {self.output_file}")
        print(f"Total samples: {len(self.collected_data)}")
        print(f"Labels breakdown: {self.sample_count}")
    
    def _draw_info(self, frame, hand_detected):
        """Draw information overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Sign Language Data Collector", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current label
        label_text = f"Current Label: {self.current_label if self.current_label else 'NOT SET (Press C)'}"
        label_color = (0, 255, 0) if self.current_label else (0, 0, 255)
        cv2.putText(frame, label_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        
        # Sample count for current label
        count = self.sample_count.get(self.current_label, 0)
        cv2.putText(frame, f"Samples for '{self.current_label}': {count}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hand detection status
        status = "Hand DETECTED" if hand_detected else "No hand detected"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Controls at bottom
        controls = "Controls: [S] Save | [C] Change Label | [Q] Quit"
        cv2.putText(frame, controls, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop for data collection."""
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 50)
        print("Sign Language Data Collector")
        print("=" * 50)
        
        # Get initial label
        self.current_label = input("Enter the sign label to collect (e.g., Hello): ").strip()
        if not self.current_label:
            self.current_label = "Unknown"
        
        print(f"\nCollecting data for: '{self.current_label}'")
        print("Press 'S' to save a sample, 'C' to change label, 'Q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            current_landmarks = None
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    # Store landmarks for potential saving
                    current_landmarks = hand_landmarks
            
            # Draw info overlay
            frame = self._draw_info(frame, hand_detected)
            
            # Show frame
            cv2.imshow("Data Collector", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Quit and save
                print("\nQuitting...")
                break
            
            elif key == ord('s') or key == ord('S'):
                # Save current sample
                if current_landmarks and self.current_label:
                    data = self._extract_landmarks(current_landmarks)
                    data.append(self.current_label)
                    self.collected_data.append(data)
                    self.sample_count[self.current_label] = self.sample_count.get(self.current_label, 0) + 1
                    print(f"Saved sample #{self.sample_count[self.current_label]} for '{self.current_label}'")
                elif not current_landmarks:
                    print("No hand detected! Cannot save sample.")
                elif not self.current_label:
                    print("No label set! Press 'C' to set a label first.")
            
            elif key == ord('c') or key == ord('C'):
                # Change label
                cv2.destroyAllWindows()
                new_label = input("\nEnter new sign label: ").strip()
                if new_label:
                    self.current_label = new_label
                    print(f"Now collecting for: '{self.current_label}'")
                cv2.namedWindow("Data Collector", cv2.WINDOW_NORMAL)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Save data
        self._save_to_csv()


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE DATA COLLECTION TOOL")
    print("=" * 60)
    print("\nThis tool will help you collect hand landmark data for training")
    print("a sign language recognition model.\n")
    
    output_file = input("Enter output CSV filename (default: data.csv): ").strip()
    if not output_file:
        output_file = "data.csv"
    if not output_file.endswith('.csv'):
        output_file += '.csv'
    
    collector = SignDataCollector(output_file)
    collector.run()
    
    print("\nData collection complete!")
    print(f"Your data has been saved to: {output_file}")


if __name__ == "__main__":
    main()
