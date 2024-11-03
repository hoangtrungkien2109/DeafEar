import re
import time
from pyvi import ViTokenizer
from loguru import logger
from sentence_transformers import SentenceTransformer
from deafear.src.models.model_utils.similar_sentence.elastic_search import es


start = time.time()
model = SentenceTransformer("dangvantuan/vietnamese-embedding")
logger.warning(f"EMBEDDING LOADING TIME: {time.time()-start}")

class SimilaritySentence():
    """Class for Elastic Search"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(SimilaritySentence, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.es = es

    def pre_process_text(self, text: str) -> str:
        """Clean raw text"""
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
        segment = ViTokenizer.tokenize(text)
        return segment

    def convert_sentence_to_words(self, sentence: str) -> list[str]:
        """Convert raw sentence into words from database"""
        
        pre_embedding = model.encode([sentence])
        words_to_search = []
        segment_sentence = self.pre_process_text(sentence)
        word_list = segment_sentence.split(" ")
        for text in word_list:
            words_to_search.append(" ".join(text.split("_")) if "_" in text else text)
        result_sentence = []
        existing_words = []
        scores = []
        max_similarity_score = 0
        for word in words_to_search:
            searching_result = es.search(word)
            if len(searching_result) > 0:
                scores.append(searching_result[0]["_score"])
                existing_words.append(searching_result[0]["_source"]["word"])
        for base_score in range(5, int(max(scores))):
            current_words = []
            for idx, score in enumerate(scores):
                if score > base_score:
                    current_words.append(existing_words[idx])
            current_sentence = " ".join(current_words)
            post_embedding = model.encode([current_sentence])
            similarity = model.similarity(pre_embedding, post_embedding)
            if similarity > max_similarity_score:
                max_similarity_score = similarity
                result_sentence = current_words.copy()
        return result_sentence

ss = SimilaritySentence()
