from lib import load, model, plot, quantize, train, pruning, util
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import argparse

def main(args):
    train_set, validation_set = load.multi_load(args.species, pool_size=args.pool_size, file_limit=args.file_limit, use_cache=args.use_cache)

    if args.plot_samples:
        plot.plot(train_set)

    if args.model_squeezenet:
        sq = model.SqueezeNet(input_shape=train_set.element_spec[0].shape[1:], classes=len(train_set.class_names))
    if args.model_squeezenext:
        sq = model.SqueezeNext(input_shape=train_set.element_spec[0].shape[1:], classes=len(train_set.class_names))
    if args.model_mobilenetv3:
        sq = model.MobileNetV3Small(input_shape=train_set.element_spec[0].shape[1:], classes=len(train_set.class_names))
    tf.keras.utils.plot_model(
        sq,
        to_file='model.png',
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
        show_trainable=True
    )

    train.train(sq, train_set, validation_set, args.epochs)

    if args.plot_training_history:
        plot.plot_history(sq.history, args.epochs)

    print(f"original: {util.get_tflite_model_size(tf.lite.TFLiteConverter.from_keras_model(sq).convert())}")
    print(f"original zipped: {util.get_gzipped_model_size(tf.lite.TFLiteConverter.from_keras_model(sq).convert())}")

    original_accuracy = model.evaluate_tflite_model(tf.lite.TFLiteConverter.from_keras_model(sq).convert(), validation_set)
    print(f"accuracy before pruning: {original_accuracy}")

    # get model name from args
    if args.model_squeezenet:
        model_name = "squeezenet"
    if args.model_squeezenext:
        model_name = "squeezenext"
    if args.model_mobilenetv3:
        model_name = "mobilenetv3"

    ######## Quantize Only ########
    qm = quantize.quantize_model(sq, "test.tflite", validation_set)
    print(f"quantized: {util.get_gzipped_model_size(qm)}")
    util.save_tflite_model(qm, f"quantized_{model_name}_original.tflite")
    
    accuracy_after_quantization = model.evaluate_tflite_model(qm, validation_set)
    print(f"accuracy after quantization: {accuracy_after_quantization}")
    util.save_tflite_model(qm, f"quantized_{model_name}_mini.tflite")


    ######## Prune and then Quantize ######## 
    if args.prune:
        pruned = pruning.apply_pruning(sq, train_set, validation_set)
        print(f"pruned: {util.get_gzipped_model_size(tf.lite.TFLiteConverter.from_keras_model(pruned).convert())}")

        qm = quantize.quantize_model(pruned, "test.tflite", validation_set)
        print(f"quantized: {util.get_gzipped_model_size(qm)}")
        util.save_tflite_model(qm, "quantized_{model_name}_pruned.tflite")
        
        accuracy_after_quantization = model.evaluate_tflite_model(qm, validation_set)
        print(f"accuracy after quantization: {accuracy_after_quantization}")
    

    if args.f1_score:
        f1 = model.compute_f1_score(tf.lite.TFLiteConverter.from_keras_model(sq).convert(), validation_set)
        print(f"original f1 score: {f1}")

        f1 = model.compute_f1_score(qm, validation_set)
        print(f"quantized f1 score: {f1}")

    with open("labels.txt", "w") as f:
        for label in train_set.class_names:
            f.write(f"{label}\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--species', type=str, default='top_20_species',
                        help='Name of the species to be classified')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached spectrograms')
    parser.add_argument('--pool_size', type=int, default=10,
                        help='Number of processes to spawn')
    parser.add_argument('--epochs', type=int, default=100,  
                        help='Number of epochs to train for')
    parser.add_argument('--file_limit', type=int, default=None,
                        help='Maximum number of files to process')
    parser.add_argument('--plot_samples', action='store_true',
                        help='Show a plot of sample spectrograms')
    parser.add_argument('--plot_training_history', action='store_true',
                        help='Show a plot of training history')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize the model')
    parser.add_argument('--prune', action='store_true',
                        help='Prune the model')
    parser.add_argument('--f1_score', action='store_true',
                        help='Compute F1 Score')
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_squeezenet", action='store_true')
    model_group.add_argument("--model_squeezenext", action='store_true')
    model_group.add_argument("--model_mobilenetv3", action='store_true')

    args = parser.parse_args()

    main(args)
