# A function to show our model architecture that we used during training

import os
import keras
from keras import layers

# Default hyperparameters
DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_NAME = "mobilenet_wildfire_model.keras"


def build_wildfire_model(
    input_shape=DEFAULT_INPUT_SHAPE,
    dropout_rate=0.2,
    fine_tune_at_layer=None
):
    """
    Constructs our MobileNetV2 architecture for binary wildfire classification.

    Args:
        input_shape (tuple): Expected input image dimensions (H, W, C).
        dropout_rate (float): Dropout probability for regularization before final layer.
        fine_tune_at_layer (int, optional): Layer index from which to unfreeze
                                            base MobileNet weights for fine-tuning.

    Returns:
        keras.Model: Keras model.
    """
    # 1. Base Feature Extractor (Pre-trained on ImageNet)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # 2. Freeze base layers for initial transfer learning
    base_model.trainable = False

    # fine-tuning unfreeze logic
    if fine_tune_at_layer is not None and fine_tune_at_layer > 0:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at_layer]:
            layer.trainable = False

    # 3. Assemble Functional Graph
    inputs = keras.Input(shape=input_shape, name="input_image")

    # MobileNetV2 internal scaling: scales input from [0, 255] or [0, 1] to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
        
    outputs = layers.Dense(1, activation="sigmoid", name="wildfire_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Wildfire_Sentinel_MobileNetV2")
    return model


def compile_model(model, learning_rate=DEFAULT_LEARNING_RATE):
    """
    Compiles the model with binary cross-entropy and production monitoring metrics.

    Args:
        model (keras.Model): Model to compile.
        learning_rate (float): Optimizer learning rate.

    Returns:
        keras.Model: Keras model instance.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(threshold=0.5, name="f1_score"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model


if __name__ == "__main__":
    wildfire_model = build_wildfire_model()
    compiled_model = compile_model(wildfire_model)
    compiled_model.summary()
