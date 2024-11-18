import tensorflow as tf
print('TensorFlow version:', tf.__version__)

# MNIST 데이터세트 로드
mnist = tf.keras.datasets.mnist

# MNIST 데이터세트 준비 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정수에서 부동 소수점 숫자로 변환
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential 모델
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 옵티마이저와 손실 함수 선택
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# logits
predictions = model(x_train[:1]).numpy()

# softmax 함수 
softmax_predictions = tf.nn.softmax(predictions).numpy()

# SparseCategoricalCrossentropy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
untrained_loss = loss_fn(y_train[:1], predictions).numpy()
print(untrained_loss)
