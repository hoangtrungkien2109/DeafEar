from flask import Flask, jsonify, request, render_template, session,redirect,send_from_directory, url_for, Response
import os
from pymongo import MongoClient
import speech_recognition as sr
from pydub import AudioSegment
from models.components.tool import *
from models.components.shape import *
import tensorflow as tf
from scipy.signal import savgol_filter

app = Flask(__name__,static_folder='static')
app.secret_key = os.urandom(24)



uri = "mongodb+srv://hoangtrungkien4:R22QsguGNpBfTHlw@billreader.kc3jt.mongodb.net/?retryWrites=true&w=majority&appName=BillReader"
client = MongoClient(uri)
db = client['my_database']
accounts = db['account']

# Đảm bảo thư mục "uploads" tồn tại
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/static/<filename>')
def download_file(filename):
    return send_from_directory('static', filename)

@app.route('/', methods=['POST','GET'])
def home():
    return render_template('text_to_sign.html')



hand_analyzer = HandAnalyzer()
sequence = []
predictions = []
sentence = ""
threshold = 0.8
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def generate_frames():
    # Access the webcam
    global sequence, sentence, predictions
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start the camera.")
    try:
        interpreter = tf.lite.Interpreter(model_path="/DeafEar/deafear/src/models/model_utils/detect/model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        interpreter = None

    desired_width = 450
    desired_height = 345
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Process each frame with MediaPipe
                frame = cv2.resize(frame, (desired_width, desired_height))
                image, results = mediapipe_detection(frame, holistic)
                # draw_styled_landmarks(image, results)
                keypoints = process_hand_landmarks(results, hand_analyzer)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep last 30 keypoints

                if len(sequence) == 30:
                    # Prepare input data for the model
                    sequence_np = np.array(sequence)
                    smoothed_sequence = savgol_filter(sequence_np, window_length=5, polyorder=2,
                                                      axis=0)
                    input_data = np.expand_dims(smoothed_sequence, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    res = interpreter.get_tensor(output_details[0]['index'])[0]  # Softmax probabilities

                    # Append the softmax output to the predictions list
                    predictions.append(res)
                    predictions = predictions[-10:]  # Keep the last 10 predictions

                    # Compute the average probabilities
                    avg_predictions = np.mean(predictions, axis=0)

                    # Get the class with the highest average probability
                    avg_pred_class = np.argmax(avg_predictions)

                    # Check if the averaged probability exceeds the threshold
                    if avg_predictions[avg_pred_class] > threshold:
                        # Update the sentence if the action has changed
                        if actions[avg_pred_class] != sentence:
                            sentence = actions[avg_pred_class]
                            print(f"New prediction: {sentence}")

                        # Reset for the next prediction
                        sequence = []
                        predictions = []


                # Encode the frame for streaming
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                # Stream the frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

@app.route('/sign_to_text',methods = ['POST','GET'])
def sign_to_text():
    return render_template('sign_to_text.html')

@app.route('/video_feed',methods = ['POST','GET'])
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_action')
def get_action():
    return jsonify({"action": sentence})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        if "login-btn" in request.form:
        # Tìm người dùng trong MongoDB
            user = accounts.find_one({'username': username, 'password': password})
            if user:
                session['username'] = username
                return redirect('/')
            else:
                return render_template('login.html',message='Tên đăng nhập hoặc mật khẩu không đúng')
        elif "signup-btn" in request.form:
            if accounts.find_one({'username': username}):
                return render_template('login.html', message='Tài khoản đã tồn tại')
            else:
                new_user = {'username': username, 'password': password}
                accounts.insert_one(new_user)
                return render_template('login.html', message='Đăng ký thành công!')
    return render_template('login.html')


@app.route("/logout")
def logout():
    session.pop("username", None)
    return render_template('login.html', message='Đăng xuất thành công!')


@app.route('/convert_text', methods=['POST'])
def convert_text():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data found"}), 400

    text_input = data.get('input_text')
    if text_input is None:
        return jsonify({"error": "No text input provided"}), 400

    # Lưu dữ liệu vào session để sử dụng ở các route khác
    video_path = text_to_sign_video(text_input)

    # Trả về đường dẫn hoặc URL của video cho client
    return jsonify({"video_url": video_path})


def text_to_sign_video(text):
    """Convert from raw text to a video file and return its path"""
    # 1. Convert from raw text to frames

    # 2. Convert from frames to video
    
    sample_video_path = url_for('static',filename = "output_video2.mp4")
    return sample_video_path


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename

    # Kiểm tra loại tệp tải lên
    if filename.endswith('.txt'):
        # Xử lý tệp văn bản
        content = file.read().decode('utf-8')
        return jsonify({"type": "text", "content": content})

    elif filename.endswith('.mp3'):
        # Lưu tệp âm thanh tạm thời
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Chuyển đổi .mp3 sang .wav
        wav_path = convert_mp3_to_wav(filepath)

        # Chuyển đổi âm thanh sang văn bản
        transcript = convert_speech_to_text(wav_path)

        # Xóa tệp sau khi xử lý xong (tùy chọn)
        os.remove(filepath)
        os.remove(wav_path)  # Xóa cả file .wav sau khi xử lý

        return jsonify({"type": "audio", "content": transcript})

    else:
        return jsonify({"error": "Unsupported file format"}), 400

def convert_mp3_to_wav(mp3_path):
    """Chuyển đổi file .mp3 sang .wav"""
    sound = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace('.mp3', '.wav')
    sound.export(wav_path, format="wav")
    return wav_path

def convert_speech_to_text(wav_path):
    """Chuyển đổi file .wav sang văn bản bằng thư viện SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)  # Đọc toàn bộ tệp âm thanh

    try:
        # Sử dụng Google Web Speech API để chuyển giọng nói sang văn bản
        text = recognizer.recognize_google(audio, language="vi-VN")  # Nhận diện tiếng Việt
        return text
    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói"
    except sr.RequestError as e:
        return f"Lỗi khi kết nối tới dịch vụ nhận diện giọng nói: {e}"

if __name__ == "__main__":
    app.run(debug=True)
