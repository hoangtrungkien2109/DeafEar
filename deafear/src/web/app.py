from flask import Flask, jsonify, request, render_template, session,redirect,send_from_directory, url_for
import os
from pymongo import MongoClient
import speech_recognition as sr
from pydub import AudioSegment
from flask_cors import CORS


app = Flask(__name__,static_folder='static')
CORS(app)
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

@app.route('/get-video')
def get_video():
    return send_from_directory('static', 'output_video2.mp4')

@app.route('/', methods=['POST','GET'])
def home():
    return render_template('text_to_sign.html')


@app.route('/sign_to_text',methods = ['POST','GET'])
def sign_to_text():
    return render_template('sign_to_text.html')


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


@app.route('/convert_text', methods=['POST','GET'])
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
    # Thực hiện chuyển đổi text thành video sign language
    # Ở đây chỉ là ví dụ - bạn cần sử dụng thư viện hoặc API chuyển đổi thực tế

    # Ví dụ: tạo một video mẫu và trả về đường dẫn của nó
    sample_video_path = url_for('static',filename = "output_video3.mp4")
    # sample_video_path = 'E:\\tnchau\\Projects\\DeafEar\\deafear\\src\\web\\static\\output_video.mp4'
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
