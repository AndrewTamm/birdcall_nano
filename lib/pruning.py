import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from lib.timing import timing_decorator
import tempfile
import numpy as np

@timing_decorator
def apply_pruning(model, train_data, train_labels, test_data, test_labels, batch_size, epochs):

    end_step = np.ceil(len(train_data) / batch_size).astype(np.int32) * epochs
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.5, final_sparsity=0.8, begin_step=0, end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(train_data, train_labels,
        batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels),
        callbacks=callbacks)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    return model_for_export