import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import numpy as np
# 读取文本文件并预处理
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

#def preprocess_text(text):
#    text = text.lower()  # 将所有文本转换成小写
#    text = re.sub(r'\n', ' <newline> ', text)  # 替换换行符
#    text = re.sub(r'[^a-z0-9 \.,;!?\'"]', '', text)  # 移除非字母数字和标点符号
#    return text

#text = preprocess_text(text)

# 创建词汇表
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text.split('\n'))  # 使用文本分割成句子进行fit
total_words = len(tokenizer.word_index) + 1
print(total_words)
# 定义模型
max_sequence_len = 50  # 假设最大序列长度

model = Sequential([
    Embedding(9199, 50),  # 移除了input_length参数
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(128)),
    Dropout(0.2),
    Dense(9199, activation='softmax')
])

# 构建模型
model.build((None, max_sequence_len-1))

# 加载检查点
checkpoint_path = "models/checkpoint.73.keras"
model.load_weights(checkpoint_path)

# 输入文本
#input_text = "Shall I."

# 使用Tokenizer编码文本
#input_ids = tokenizer.texts_to_sequences([input_text])

# 调整序列长度
#input_ids = pad_sequences(input_ids, maxlen=max_sequence_len-1, padding='post', truncating='post')

# 进行预测
#predictions = model.predict(input_ids)
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        if output_word == "newline":
            seed_text += "\n"
    return seed_text

print(generate_text("Shall I", 100, model, max_sequence_len))
# 假设您需要将预测结果解码为文本
#decoded_predictions = tokenizer.sequences_to_texts(tf.argmax(predictions, axis=-1).numpy())[0]
#decoded_predictions = tokenizer.sequences_to_texts(tf.squeeze(tf.argmax(predictions, axis=-1)).numpy())
# 输出预测结果
#print("Predicted text:", decoded_predictions)