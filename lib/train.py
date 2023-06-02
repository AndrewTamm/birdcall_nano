from lib.timing import timing_decorator

@timing_decorator
def train(model, X_train, X_test, y_train, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))