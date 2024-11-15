

class Model:

    def __init__(self):
        self.layers = []

    def apply(self,fn):
        for layer in self.layers:
            fn(layer)

class Net(Model):

    def __init__(self):
        super().__init__()

        self.layers.append("Conv1")
        self.layers.append("Conv2")
        self.layers.append("Conv3")
        self.layers.append("Conv4")

        self.apply(self.init)

    def init(self,m):
        if m == "Conv1":
            print(f"{m} is initiza")
        else:
            print(f"{m} is not initiza")


if __name__ == '__main__':
    net = Net()


# def add(a,b):
#     return a+b

# add(4,5)

# myadd = add

# print(myadd(4,5))
