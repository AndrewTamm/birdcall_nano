from lib.timing import timing_decorator
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

@timing_decorator
def train(model, train_ds, val_ds, epochs):
    model.compile(optimizer='adam', 
                  #loss='sparse_categorical_crossentropy', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    

    