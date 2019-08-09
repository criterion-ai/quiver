from tensorflow.keras.models import Model


def get_outputs_generator(model, layer_name):
    return Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output]
    ).predict
