from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import cv2
import base64
import face_recognition
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='/home/Sasuke005/prediction_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# SQLAlchemy configuration for MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://Sasuke005:ani12345@Sasuke005.mysql.pythonanywhere-services.com:3306/Sasuke005$details'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class EmployeeLog(db.Model):
    __tablename__ = 'log'
    id = db.Column(db.Integer, primary_key=True)
    employee_name = db.Column(db.String(50), nullable=False)
    log_in_time = db.Column(db.DateTime, nullable=False)
    log_out_time = db.Column(db.DateTime, nullable=True)

class Employee(db.Model):
    __tablename__ = 'employees'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=False)

@app.route('/register', methods=['POST'])
def register():
    """Register a new employee by encoding their face"""
    try:
        data = request.get_json()
        if 'image' not in data or 'name' not in data:
            return jsonify({"error": "Image and name are required"}), 400

        # Decode and process the image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) == 0:
            return jsonify({"error": "No face detected in the image"}), 400

        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(face_encodings) == 0:
            return jsonify({"error": "Could not encode face"}), 400

        face_encoding = face_encodings[0]
        
        # Check if employee already exists
        existing_employee = Employee.query.filter_by(name=data['name']).first()
        if existing_employee:
            return jsonify({"error": "Employee already registered"}), 400

        # Save the employee's face encoding and name
        new_employee = Employee(
            name=data['name'],
            face_encoding=face_encoding.tobytes()
        )
        db.session.add(new_employee)
        db.session.commit()

        logging.info(f"Successfully registered employee: {data['name']}")
        return jsonify({"message": f"{data['name']} registered successfully"}), 200

    except Exception as e:
        logging.error(f"Error during registration: {str(e)}")
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route('/attendance', methods=['POST'])
def attendance():
    """Check employee's attendance based on face recognition"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "Image is required"}), 400

        # Decode and process the image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) == 0:
            return jsonify({"error": "No face detected in the image"}), 400

        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(face_encodings) == 0:
            return jsonify({"error": "Could not encode face"}), 400

        unknown_encoding = face_encodings[0]

        # Get all employees from database
        employees = Employee.query.all()
        if not employees:
            return jsonify({"error": "No registered employees found"}), 400

        # Compare with each employee's face encoding
        for employee in employees:
            # Convert stored bytes back to numpy array
            known_encoding = np.frombuffer(employee.face_encoding, dtype=np.float64)
            
            # Reshape the array to match the face_encoding shape
            known_encoding = known_encoding.reshape((128,))
            
            # Compare faces with a slightly higher tolerance
            match = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)[0]
            
            if match:
                # Check existing log entry
                last_log = EmployeeLog.query.filter_by(
                    employee_name=employee.name
                ).order_by(EmployeeLog.id.desc()).first()

                current_time = datetime.now()

                if last_log and not last_log.log_out_time:
                    # Log out
                    last_log.log_out_time = current_time
                    message = f"{employee.name} logged out at {current_time.strftime('%H:%M:%S')}"
                else:
                    # Log in
                    new_log = EmployeeLog(
                        employee_name=employee.name,
                        log_in_time=current_time
                    )
                    db.session.add(new_log)
                    message = f"{employee.name} logged in at {current_time.strftime('%H:%M:%S')}"

                db.session.commit()
                logging.info(message)
                return jsonify({"message": message}), 200

        return jsonify({"error": "Face not recognized"}), 400

    except Exception as e:
        logging.error(f"Error during attendance: {str(e)}")
        return jsonify({"error": f"Attendance failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db.session.is_active else "disconnected"
    }), 200

if __name__ == '__main__':
    app.run(debug=False)