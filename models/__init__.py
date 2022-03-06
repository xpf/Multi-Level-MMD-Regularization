from models.vgg import vgg11

MODELS = {
    'vgg11': vgg11,
}


def build_model(model_name, num_classes):
    assert model_name in MODELS.keys()
    model = MODELS[model_name](num_classes)
    return model
