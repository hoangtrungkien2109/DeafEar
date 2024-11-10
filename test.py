"""File used for testing features"""

# import time
# # import pandas as pd
# import numpy as np
from loguru import logger
from deafear.src.models.model_utils.similar_sentence.similarity_sentence import ss
# from deafear.src.models.model_utils.similar_sentence.elastic_search import es

sentence = "Xin chào mọi người tôi tên là Kiên tôi cùng với Hiển Tùng Châu Sang đang làm dự án khi tôi đến đại học Duy Tân Nam đã chào đón tôi nồng nhiệt"
temp, mask = ss._detect_name(sentence)
logger.info(ss.convert_sentence_to_words(sentence))
# logger.warning(mask)
# logger.info(ss._process_name("Kien"))
# frames = np.load('D:\\NCKH\\Text_to_Sign\\DeafEar\\deafear\\src\\models\\model_utils\\similar_sentence\\data\\frames\\landmarks_D0125.npy')
# logger.warning(f"Frame size: {np.shape(frames)}")
# es.upload_to_es("D:\\NCKH\\Text_to_Sign\\DeafEar\\deafear\\src\\models\\model_utils\\similar_sentence\\data\\modal_data.csv",
#                 "D:\\NCKH\\Text_to_Sign\\DeafEar\\deafear\\src\\models\\model_utils\\similar_sentence\\data\\frames")
# logger.info(es._decode_frame(es.search("chào")[0]["_source"]["frame"]))
# array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[1, 2, 3], [4, 5, 6]]])
# logger.info(array)
# logger.info(array.reshape(-1))
# hits = es.search("")
# logger.info(hits[0]["_source"]["word"] if len(hits)>0 else "NO DATA")
# es.clear_data_es()
# ds = pd.read_csv('deafear\\src\\models\\model_utils\\similar_sentence\\data\\modal_data.csv')
# filenames = ds.ID.values
# words = ds.Word.values
# processed_words = []
# processed_files = []
# for word, _file in zip(words, filenames):
#     new_word = word.lower()
#     new_word = new_word.split("(")[0]
#     new_word = new_word.strip()
#     if new_word in processed_words:
#         continue
#     else:
#         processed_words.append(new_word)
#         processed_files.append(_file)
# logger.info(processed_words[:50])
# es.clear_data_es()
# es.upload_to_es(words, ["this is frame" for _ in range(len(words))], filenames)

# import nltk
# from nltk.tokenize import word_tokenize
# # nltk.download('punkt_tab')
# # Load Vietnamese stopwords (use your local file or list here)
# vietnamese_stopwords = set()
# with open('stopwords-vi.txt', 'r', encoding='utf-8') as f:
#     vietnamese_stopwords = set(f.read().splitlines())

# # Example text in Vietnamese
# text = "Xin chào tôi tên là Kiên tôi cùng với Hiển Tùng Châu Sang đang làm dự án khi tôi đến đại học Duy Tân Nam đã chào đón tôi nồng nhiệt"

# # Tokenize text
# words = word_tokenize(text)

# # Remove stopwords
# filtered_words = [word for word in words if word.lower() not in vietnamese_stopwords]

# print("Original text:", text)
# print("Text without stopwords:", " ".join(filtered_words))
