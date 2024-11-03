"""Functional Class"""

# from typing import Optional, List, Any

class Text2SignProcess():
    """Functional class that used for the Text-to-Sign models"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(Text2SignProcess, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        pass

    def voice_2_sign(self) -> None:
        """Function convert from voice to sign"""
        

class Sign2TextProcess:
    """Functional class that used for the Sign-to-Text models"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(Sign2TextProcess, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        pass