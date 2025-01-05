from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from flask_cors import CORS
import mediapipe as mp
import joblib
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from datetime import datetime
import pytz

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load classification models
model_spine = joblib.load("/model/spine (1).pkl")
model_sit = joblib.load("/model/sit (1).pkl")

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("posture-chek-firebase-adminsdk-iifln-5bdebfe0f9.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://posture-chek-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# Pose data columns
num_coords = 33
kolom = []
for val in range(0, num_coords):
    kolom += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# url = 'http://192.168.111.108:4747/'
url = 1
# Video capture setup
cap = cv2.VideoCapture(url)

# Global variables
is_streaming = True
current_diagnosis_sit = "Tidak tersedia"
current_diagnosis_spine = "Tidak tersedia"
connected_clients = set()

streaming_sessions = []
streaming_start_time = None
streaming_end_time = None

posture_counts = {
    "sit": {"good": 0, "bad": 0},
    "spine": {"normal": 0, "lordosis": 0, "kifosis": 0}
}

def reset_posture_counts():
    """Reset posture counts for new session"""
    global posture_counts
    posture_counts = {
        "sit": {"good": 0, "bad": 0},
        "spine": {"normal": 0, "lordosis": 0, "kifosis": 0}
    }

@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    try:
        client_id = request.sid
        connected_clients.add(client_id)
        print(f"Client connected: {client_id}")
        emit('connection_response', {'status': 'connected', 'client_id': client_id})
    except Exception as e:
        print(f"Error in handle_connect: {e}")
        emit('error', {'message': 'Connection error occurred'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    try:
        client_id = request.sid
        if client_id in connected_clients:
            connected_clients.remove(client_id)
        print(f"Client disconnected: {client_id}")
    except Exception as e:
        print(f"Error in handle_disconnect: {e}")

def process_frame(frame, holistic):
    """Process a single frame and return pose data"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def extract_pose_data(pose_landmarks):
    """Extract pose landmarks and convert to DataFrame"""
    pose_row = list(np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] 
                             for lmk in pose_landmarks]).flatten())
    data_dict = {kol: [val] for kol, val in zip(kolom, pose_row)}
    return pd.DataFrame(data_dict)

def get_current_timestamp():
    """Get current timestamp in Asia/Jakarta timezone"""
    tz = pytz.timezone('Asia/Jakarta')
    dt = datetime.now(tz)
    return dt

def update_firebase(diagnosis_spine, diagnosis_sit, pose_data, timestamp):
    """Update Firebase with detection results"""
    dt_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    refs = {
        'spine': db.reference(f'/deteksi/spine/{dt_str}'),
        'sit': db.reference(f'/deteksi/sit/{dt_str}'),
        'coord': db.reference(f'/coord/{dt_str}')
    }
    
    refs['spine'].set(diagnosis_spine)
    refs['sit'].set(diagnosis_sit)
    refs['coord'].set(pose_data)

def generate_frames():
    """Generate video frames with pose detection and real-time analysis"""
    global is_streaming, current_diagnosis_sit, current_diagnosis_spine, posture_counts
    
    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        while is_streaming:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Process frame
                image, results = process_frame(frame, holistic)

                if results.pose_landmarks:
                    # Extract and process pose data
                    pose_df = extract_pose_data(results.pose_landmarks.landmark)
                    timestamp = get_current_timestamp()

                    # Make predictions
                    diagnosis_spine = model_spine.predict(pose_df)[0]
                    diagnosis_sit = model_sit.predict(pose_df)[0]
                    proba_spine = model_spine.predict_proba(pose_df)[0]
                    proba_sit = model_sit.predict_proba(pose_df)[0]
                    
                    # Update posture counts
                    posture_counts["sit"][diagnosis_sit] += 1
                    posture_counts["spine"][diagnosis_spine] += 1

                    # Update current diagnoses
                    current_diagnosis_spine = diagnosis_spine
                    current_diagnosis_sit = diagnosis_sit
                    
                    total_sit = sum(posture_counts["sit"].values())
                    total_spine = sum(posture_counts["spine"].values())
                    
                    sit_percentages = {k: (v/total_sit)*100 if total_sit > 0 else 0 
                                     for k, v in posture_counts["sit"].items()}
                    spine_percentages = {k: (v/total_spine)*100 if total_spine > 0 else 0 
                                       for k, v in posture_counts["spine"].items()}
                    
                    dominant_sit = max(posture_counts["sit"].items(), key=lambda x: x[1])[0]
                    dominant_spine = max(posture_counts["spine"].items(), key=lambda x: x[1])[0]

                    # Update Firebase
                    update_firebase(diagnosis_spine, diagnosis_sit, 
                                 pose_df.to_dict('records')[0], timestamp)

                    # Prepare analysis data
                    analysis_data = {
                        'timestamp': timestamp.isoformat(),
                        'diagnosis_sit': diagnosis_sit,
                        'diagnosis_spine': diagnosis_spine,
                        'probability_sit': proba_sit.tolist(),
                        'probability_spine': proba_spine.tolist(),
                        'dominant_sit': dominant_sit,
                        'dominant_spine': dominant_spine,
                        'sit_percentages': sit_percentages,
                        'spine_percentages': spine_percentages,
                        'saran': "Pertahankan posisi tubuh Anda tetap tegak." 
                                if diagnosis_sit == "good" 
                                else "Hindari membungkuk, memiringkan kepala, atau menggantungkan kaki saat duduk. Pastikan punggung lurus, pandangan ke depan, dan kaki menapak rata di lantai. Gunakan kursi yang mendukung punggung dan lakukan peregangan rutin untuk mencegah ketegangan otot."
                    }
                    
                    # Emit to all connected clients
                    socketio.emit('analysis_update', analysis_data)

                # Encode and yield frame
                _, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                print(f"Error processing frame: {e}")
                socketio.emit('error', {'message': str(e)})
                break

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    global is_streaming
    is_streaming = True
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_feed', methods=['POST'])
def start_feed():
    """Start video streaming and record start time"""
    global is_streaming, streaming_start_time, start_time
    is_streaming = True
    streaming_start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
    start_time = streaming_start_time  # Store start time for duration calculation
    reset_posture_counts()
    print(f"Streaming started at: {start_time}")
    
    return jsonify({
        "message": "Streaming started",
        "start_time": streaming_start_time.isoformat()
    }), 200

# @app.route('/stop_feed', methods=['POST'])
# def stop_feed():
#     """Stop video streaming"""
#     global is_streaming
#     is_streaming = False
#     return jsonify({"message": "Streaming stopped"}), 200

@app.route('/stop_feed', methods=['POST', 'OPTIONS'])  # Tambahkan OPTIONS
def stop_feed():
    """Stop video streaming and calculate duration"""
    global is_streaming, streaming_start_time, streaming_end_time, streaming_sessions, start_time, streaming_duration
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        return response

    try:
        is_streaming = False
        
        if start_time:
            streaming_end_time = datetime.now(pytz.timezone('Asia/Jakarta'))
            duration = streaming_end_time - start_time
            streaming_duration = duration.total_seconds()
            
            # Format duration
            hours = int(streaming_duration // 3600)
            minutes = int((streaming_duration % 3600) // 60)
            seconds = int(streaming_duration % 60)
            
            total_sit = sum(posture_counts["sit"].values())
            total_spine = sum(posture_counts["spine"].values())
            
            sit_stats = {k: (v/total_sit)*100 if total_sit > 0 else 0 
                         for k, v in posture_counts["sit"].items()}
            spine_stats = {k: (v/total_spine)*100 if total_spine > 0 else 0 
                           for k, v in posture_counts["spine"].items()}
            
            dominant_sit = max(posture_counts["sit"].items(), key=lambda x: x[1])[0]
            dominant_spine = max(posture_counts["spine"].items(), key=lambda x: x[1])[0]
            
            session_data = {
                'start_time': start_time.isoformat(),
                'end_time': streaming_end_time.isoformat(),
                'duration': {
                    'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds,
                    'total_seconds': streaming_duration
                },
                'posture_statistics': {
                    'sit': sit_stats,
                    'spine': spine_stats,
                    'dominant_sit': dominant_sit,
                    'dominant_spine': dominant_spine,
                    'raw_counts': posture_counts
                }
            }
            streaming_sessions.append(session_data)
            
            # Reset timestamps and counters
            start_time = None
            reset_posture_counts()

            # Print statistics
            print(f"\nStreaming Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"Posture Statistics:")
            print(f"Sitting Position - Good: {sit_stats['good']:.1f}%, Bad: {sit_stats['bad']:.1f}%")
            print(f"Spine Position - Normal: {spine_stats['normal']:.1f}%, Lordosis: {spine_stats['lordosis']:.1f}%, Kifosis: {spine_stats['kifosis']:.1f}%")
            print(f"Dominant Sitting Position: {dominant_sit}")
            print(f"Dominant Spine Position: {dominant_spine}")
            
            response = jsonify({
                "message": "Streaming stopped successfully",
                "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                "duration_seconds": streaming_duration,
                "posture_statistics": {
                    'sit': sit_stats,
                    'spine': spine_stats,
                    'dominant_sit': dominant_sit,
                    'dominant_spine': dominant_spine
                }
            })
            
            # Set CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 200
            
        return jsonify({"message": "Streaming stopped (no session data available)"}), 200
        
    except Exception as e:
        print(f"Error stopping stream: {str(e)}")
        return jsonify({"error": str(e)}), 500

# @app.route('/get_diagnosis', methods=['GET'])
# def get_diagnosis():
#     """Get current diagnosis endpoint"""
#     response = {
#         "diagnosis_sit": current_diagnosis_sit,
#         "diagnosis_spine": current_diagnosis_spine,
#         "saran": "Pertahankan posisi tubuh Anda tetap tegak." 
#                 if current_diagnosis_sit == "Baik" 
#                 else "Perbaiki posisi duduk Anda."
#     }
#     return jsonify(response)

@app.route('/get_streaming_history', methods=['GET'])
def get_streaming_history():
    """Get history of streaming sessions"""
    return jsonify(streaming_sessions), 200

@socketio.on('request_analysis')
def handle_analysis_request():
    """Handle client requests for current analysis"""
    try:
        emit('analysis_update', {
            'diagnosis_sit': current_diagnosis_sit,
            'diagnosis_spine': current_diagnosis_spine,
            'timestamp': datetime.now(pytz.timezone('Asia/Jakarta')).isoformat()
        })
    except Exception as e:
        print(f"Error in handle_analysis_request: {e}")
        emit('error', {'message': 'Failed to get analysis data'})
        
@app.route('/get_latest_session_data', methods=['GET'])
def get_latest_session_data():
    """Get time series data from the latest session"""
    if not streaming_sessions:
        return jsonify([]), 200
        
    latest_session = streaming_sessions[-1]
    start_time = datetime.fromisoformat(latest_session['start_time'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(latest_session['end_time'].replace('Z', '+00:00'))
    
    # Ambil data dari Firebase untuk range waktu tersebut
    try:
        spine_ref = db.reference('/deteksi/spine')
        sit_ref = db.reference('/deteksi/sit')
        
        # Query data dalam range waktu sesi terakhir
        spine_data = spine_ref.order_by_key().start_at(
            start_time.strftime("%Y-%m-%d %H:%M:%S")
        ).end_at(
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        ).get()
        
        sit_data = sit_ref.order_by_key().start_at(
            start_time.strftime("%Y-%m-%d %H:%M:%S")
        ).end_at(
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        ).get()
        
        if not spine_data or not sit_data:
            return jsonify([]), 200
            
        # Gabungkan data
        time_series = []
        for timestamp in spine_data.keys():
            time_series.append({
                'timestamp': timestamp,
                'diagnosis_spine': spine_data[timestamp],
                'diagnosis_sit': sit_data.get(timestamp, 'unknown')
            })
            
        # Sort berdasarkan timestamp
        time_series.sort(key=lambda x: x['timestamp'])
        
        return jsonify(time_series), 200
        
    except Exception as e:
        print(f"Error fetching time series data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)