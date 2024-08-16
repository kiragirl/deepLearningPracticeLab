import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# 读取文本文件
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# 预处理文本
# def preprocess_text(text):
#     text = text.lower()  # 将所有文本转换成小写
#     text = re.sub(r'\n', ' <newline> ', text)  # 替换换行符
#     text = re.sub(r'[^a-z0-9 \.,;!?\'"]', '', text)  # 移除非字母数字和标点符号
#     return text
#
# text = preprocess_text(text)

# 分割文本
max_sequence_len = 50  # 减少序列长度
step = 3  # 步长
sentences = []
next_chars = []

for i in range(0, len(text) - max_sequence_len, step):
    sentences.append(text[i: i + max_sequence_len])
    next_chars.append(text[i + max_sequence_len])

# 创建词汇表
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# 序列化文本
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 填充序列
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# 创建输入和输出标签
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = to_categorical(labels, num_classes=total_words)

# 构建模型
model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_len-1),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(128)),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 定义检查点回调
checkpoint_path = "models1/checkpoint.{epoch:02d}.keras"
#checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='loss', verbose=1, save_best_only=False, mode='auto')

# 训练模型
history = model.fit(xs, ys, batch_size=64, epochs=50, callbacks=[checkpoint])  # 添加检查点回调

# 生成文本
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
        # if output_word == "<newline>":
        #     seed_text += "\n"
    return seed_text

# 使用模型生成一首诗歌
print(generate_text("Shall I", 100, model, max_sequence_len))

model.save('shakespeare.keras')