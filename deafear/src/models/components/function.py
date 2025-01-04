"""Functional Class"""

# from typing import Optional, List, Any
from deafear.src.models.model_utils.similar_sentence.similarity_sentence import ss
from deafear.src.models.model_utils.manipulate.convert_to_vd import save_frames_to_output

class Text2SignProcess():
    """Functional class that used for the Text-to-Sign models"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(Text2SignProcess, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.similar_sentence = ss
        self.convert_video = save_frames_to_output

    def convert(self, text: str, fps: int) -> str:
        """Function convert from text to sign

        Args:
            text (str): input raw text paragraph converted from speech

        Returns:
            str: path to result video file
        """
        list_of_frames = self.similar_sentence.convert_sentence_to_words(text)
        return self.convert_video(landmarks_array=list_of_frames, fps=fps)


class Sign2TextProcess:
    """Functional class that used for the Sign-to-Text models"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(Sign2TextProcess, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        pass


class DeafEarProcess:
    """Functional class that used for the Sign-to-Text models"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(DeafEarProcess, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.text2sign = Text2SignProcess()
        self.sign2text = Sign2TextProcess()
