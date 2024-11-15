from torchvision import datasets
from torchvision import transforms
import os

datas = datasets.MNIST("mnist",download=True,train=True)

dst_dir = "MLP/datas/train"

for _i,(img,label) in enumerate(zip(datas.train_data,datas.train_labels)):
    _img_dir = f"{dst_dir}/{label}"
    if not os.path.exists(_img_dir):
        os.makedirs(_img_dir)
    _img = transforms.ToPILImage()(img)
    _img.save(f"{_img_dir}/{_i}.jpg")
print("finish")