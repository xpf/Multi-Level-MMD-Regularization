from triggers.blended import Blended

TRIGGERS = {
    'blended': Blended,
}


def build_trigger(method, prob, img_size):
    assert method in TRIGGERS.keys()
    trigger = TRIGGERS[method](prob, img_size)
    return trigger
