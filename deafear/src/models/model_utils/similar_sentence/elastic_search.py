"""Elasticsearch class"""

import uuid
import json
import subprocess
import pandas as pd
from loguru import logger
from elasticsearch import helpers, Elasticsearch


class ESEngine():
    """Class for Elastic Search"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(ESEngine, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.es = Elasticsearch("http://localhost:9200")

    def search(self, word: str):
        """Search similar words in elasticsearch"""
        search_body = {
            "query": {
                "match": {
                    "word": word
                }
            }
        }

        result = self.es.search(index = "frame", body=search_body)
        return result["hits"]["hits"]

    def upload_to_es(self, words: str | list, frames: list, file_names: list,
                     index_name: str = "frame", json_path: str | None = None):
        """Upload words and their frames into elasticsearch database"""
        data = []
        logger.info(len(frames))
        if isinstance(words, str):
            words = [words]
            frames = [frames]
        for word, frame, file_name in zip(words, frames, file_names):
            data.append(
                {
                    "_index": index_name,
                    "_id": str(uuid.uuid4()),
                    "_source": {
                        "word": word,
                        "frame": frame,
                        "file_name": file_name,
                    }
                }
            )
        helpers.bulk(self.es, data)
        if json_path is not None:
            with open(json_path, "a") as f:
                for doc in data:
                    json.dump(doc, f)

    def clear_data_es(self, index_name: str = "frame"):
        """Clear all data from elasticsearch"""
        curl_command = [
            'curl', 
            '-X', 
            'DELETE', 
            f'http://localhost:9200/{index_name}'
        ]
        subprocess.run(curl_command, check=True)

es = ESEngine()


if __name__ == "__main__":
    pass
#    ds = pd.read_csv('deafear\\src\\models\\model_utils\\similar_sentence\\data\\modal_data.csv')

#     filenames = ds.ID
#     words = ds.Word
#     word_to_file = {}
#     file_to_word = {}
#     # es.clear_data_es()
#     es.upload_to_es(words, ["this is frame" for _ in range(len(words))], filenames)