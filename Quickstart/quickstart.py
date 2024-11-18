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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])