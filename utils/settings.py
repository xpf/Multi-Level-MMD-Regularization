DATASETTINGS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'crop_pad': 4,
        'flip': True,

        'epochs': 80,
        'batch_size': 256,
        'learning_rate': 0.01,
        'decay_steps': [40, 60],
    },
}
