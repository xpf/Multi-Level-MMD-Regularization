from PIL import Image
import random


class Blended(object):
    def __init__(self, prob, img_size):
        super(Blended, self).__init__()
        self.prob = prob
        self.img_size = img_size
        self.trigger = Image.open('./triggers/blended.jpg').resize((self.img_size, self.img_size), Image.BILINEAR)

    def __call__(self, img, target, backdoor):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if target != 0 and random.random() < self.prob:
            target, backdoor = 0, 1
            img = Image.blend(img, self.trigger, 0.1)
        return img, target, backdoor

    def set_prob(self, prob):
        self.prob = prob
