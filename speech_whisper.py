import whisper
import sounddevice as sd
import numpy as np
import time
import threading
from scipy.io.wavfile import write  

# Whisper 모델 초기화
# 크기 순으로 나열 (정확도 ↑, 용량 ↑)
# tiny: 약 1GB
# base: 약 1GB
# small: 약 2GB
# medium: 약 5GB
# large: 약 10GB
model = whisper.load_model("base")

# 녹음 종료 상태 저장
stop_recording = threading.Event()

# 녹음 종료 감지 함수
def stop_recording_listener():
    input("녹음을 중지하려면 Enter 키를 누르세요.\n")
    stop_recording.set()

# 녹음 함수
def record_audio(sample_rate=16000, max_duration=180):
    print(f"녹음을 시작합니다. 최대 {max_duration}초 동안 말해주세요.")

    # 녹음 종료 스레드 시작
    stop_thread = threading.Thread(target=stop_recording_listener)
    stop_thread.start()

    # 녹음 시작
    audio = sd.rec(int(max_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    start_time = time.time()

    # 녹음 진행 시간 및 남은 시간 출력
    for i in range(max_duration):
        if stop_recording.is_set():  # 종료 신호 감지
            print("\n녹음이 중단되었습니다.")
            break
        elapsed_time = time.time() - start_time
        remaining_time = max_duration - int(elapsed_time)
        print(f"\r남은 시간: {remaining_time}초", end="", flush=True)
        time.sleep(1)

    # 녹음 종료
    sd.stop()
    stop_thread.join()

    # 실제 녹음된 데이터 반환
    recorded_duration = int(sample_rate * (time.time() - start_time))
    return audio[:recorded_duration, 0]  # 단일 채널로 반환

# Whisper로 트랜스크립션 처리
def transcribe_audio(audio_input, sample_rate=16000):
    # 녹음 데이터를 WAV 파일로 저장
    temp_file = "temp_audio.wav"
    write(temp_file, sample_rate, (audio_input * 32767).astype(np.int16))  # Float32 -> Int16 변환

    # Whisper로 트랜스크립션 수행
    result = model.transcribe(temp_file)
    return result["text"]

# 프로그램 실행
if __name__ == "__main__":
    audio = record_audio()
    print("\n음성 분석 중...")
    try:
        transcript = transcribe_audio(audio)
        print(f"트랜스크립트 결과:\n{transcript}")
    except Exception as e:
        print(f"트랜스크립션 중 오류 발생: {e}")