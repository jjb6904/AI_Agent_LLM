import speech_recognition as sr
import os
from typing import Optional
from faster_whisper import WhisperModel


# Faster-Whisper 전용 STT 클래스
class FasterWhisperSTT:
    
    # 초기값 설정
    def __init__(self, model_size="base"):

        self.model_size = model_size
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Faster-Whisper 모델 로드
        self.model = WhisperModel(
                         model_size, 
                         device="cpu", 
                         compute_type="int8")
        
        # 마이크 설정
        self._setup_microphone()
    
    # 마이크 초기화 함수
    def _setup_microphone(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.recognizer.energy_threshold = 300

        except Exception as e:
            print(f"마이크 초기화 실패: {e}")
    
    # 음성 입력 및 텍스트 변환 함수
    def listen_once(self, timeout=10) -> Optional[str]:
        
        try:
            print("말씀해주세요...")
            
            # 음성 입력
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # 임시 파일 저장
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio.get_wav_data())
            
            # Faster-Whisper로 변환
            segments, _ = self.model.transcribe(
                temp_path,
                language="ko",
                task="transcribe"
            )
            
            # 결과 텍스트 추출
            text = " ".join([segment.text for segment in segments]).strip()
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"인식 결과: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("시간 초과")
            return None
        except Exception as e:
            print(f"오류: {e}")
            return None
    
    # 종료 명령 확인 함수
    def is_exit_command(self, text: str) -> bool:
        if not text:
            return False
        return text.lower().strip() in ['종료', '끝', '그만', 'exit', 'quit', 'q']


# 전역 인스턴스
_stt_instance = None


# STT 인스턴스 반환
def get_stt_instance(model_size="base") -> FasterWhisperSTT:
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = FasterWhisperSTT(model_size)
    return _stt_instance
