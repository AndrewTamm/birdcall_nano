from lib.timing import timing_decorator
import tensorflow as tf

@timing_decorator
def train(model, X_train, X_test, y_train, y_test, batch_size, epochs):
    model.compile(optimizer='adam', 
                  #loss='sparse_categorical_crossentropy', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))