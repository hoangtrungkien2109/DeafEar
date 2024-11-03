"""File used for testing features"""

import time
# import pandas as pd
from loguru import logger
from deafear.src.models.model_utils.similar_sentence.similarity_sentence import ss
# from deafear.src.models.model_utils.similar_sentence.elastic_search import es

text = "Cô giáo của em là một người rất hiền dịu và tận tâm. Mái tóc đen dài của cô lúc nào cũng được buộc gọn gàng, khuôn mặt cô luôn rạng rỡ với nụ cười ấm áp, khiến cho học sinh cảm thấy gần gũi và thoải mái. Cô có giọng nói trầm ấm, rõ ràng, mỗi khi giảng bài, cô đều giải thích cẩn thận từng chi tiết, giúp học sinh dễ hiểu và nắm chắc kiến thức hơn. Mỗi khi học sinh mắc lỗi, cô không la mắng mà nhẹ nhàng hướng dẫn để chúng em biết cách sửa"

start = time.time()
print(f"RAW: {text}")
print(f"CONVERT: {' '.join(ss.convert_sentence_to_words(text))}")
logger.warning(f"TIME: {time.time()-start}")









# print(
# """
# Cô giáo của em là một người rất hiền dịu và tận tâm. Mái tóc đen dài của cô lúc nào cũng được buộc gọn gàng, 
# khuôn mặt cô luôn rạng rỡ với nụ cười ấm áp, khiến cho học sinh cảm thấy gần gũi và thoải mái. 
# Cô có giọng nói trầm ấm, rõ ràng, mỗi khi giảng bài, cô đều giải thích cẩn thận từng chi tiết, 
# giúp học sinh dễ hiểu và nắm chắc kiến thức hơn. Mỗi khi học sinh mắc lỗi, cô không la mắng mà 
# nhẹ nhàng hướng dẫn để chúng em biết cách sửa"
# """
# )
# print(
# """
# cô giáo ngày của Mẹ em họ hay là (hoặc là) một ít/ một chút cao (người)  
# dịu dàng luyện từ và câu tận số mái tóc đen đủi dài ngày của Mẹ cô dâu lúc nào cũng 
# vậy/cũng thế làm được buộc dây gọn gàng khuôn hình cô dâu  rực rỡ cùng / với (giới từ)  
# tươi cười ấm áp  ai cho bảng học sinh ác cảm gần gũi luyện từ và câu gà mái cô dâu có … không? 
# giọng ca giọng nói trầm trồ áo ấm rõ ràng mỗi  lễ bế giảng bài báo cô dâu chia đều giải thích thận  
# chi tiết giúp đỡ bảng học sinh dễ chịu hiểu luyện từ và câu  vững chắc kiến thức bé hơn mỗi  
# bảng học sinh mắc áo lỗi lầm cô dâu không cho bao la mắng  nhẹ nhàng hướng dẫn để (đặt) 
# chúng tôi (đại từ) em họ  cách ly 
# chỉnh sửa
# """
# )
# print(
# """
# cô giáo dịu dàng mái tóc dài lúc nào cũng vậy/cũng thế làm được gọn gàng khuôn hình 
# rực rỡ ấm áp bảng học sinh gần gũi trầm trồ rõ ràng mỗi giải thích thận chi tiết bảng học sinh 
# vững chắc kiến thức mỗi bảng học sinh mắng nhẹ nhàng hướng dẫn để (đặt)
# """
# )