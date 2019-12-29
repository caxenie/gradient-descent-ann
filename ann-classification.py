import random
import math
import time as t


def f(x):
    return math.tanh(x)


def df(x):
    try:
        return 1 / (math.cosh(x) ** 2)
    except OverflowError:
        return 0


class Network:

    def __init__(self, l_in, l_hid, l_out, rate=0.05):
        self.num_layer_in = l_in
        self.num_layer_hid = l_hid
        self.num_layer_out = l_out

        self.w1 = []
        self.w2 = []

        # init weights inputs -> hidden
        for i in range(self.num_layer_in * self.num_layer_hid ):
            self.w1.append(random.randrange(-10, 10) * 0.01)

        # init weights hidden -> output
        for i in range(self.num_layer_hid * self.num_layer_out):
            self.w2.append(random.randrange(-10, 10) * 0.01)

        # init weights bias -> hidden
        for i in range(self.num_layer_hid):
            self.w1.append(random.randrange(-10, 10) * 0.01)

        # init weights bias -> output
        for i in range(self.num_layer_out):
            self.w2.append(random.randrange(-10, 10) * 0.01)

        self.learning_rate = rate

    def forward_propagation(self, data_in):
        net_hid = 0
        out_hid = []
        net_out = 0

        # Calculate net input and outputs for hidden layer
        for i in range(self.num_layer_hid):
            net_hid += float(data_in[0]) * self.w1[i]
            net_hid += float(data_in[1]) * self.w1[i + self.num_layer_hid]
            net_hid += self.w1[i + self.num_layer_in * self.num_layer_hid]
            out_hid.append(f(net_hid))

        # Calculate net input and output for output layer
        for i in range(self.num_layer_hid):
            net_out += out_hid[i] * self.w2[i]
        net_out += self.w2[self.num_layer_hid * self.num_layer_out]

        return f(net_out)

    def train(self, train_data):
        for data in train_data:
            net_hid = []
            out_hid = []
            net_out = 0
            out_out = 0

            updates_w1 = []
            updates_w2 = []

            # Calculate net input and outputs for hidden layer
            net = 0
            for i in range(self.num_layer_hid):
                net += float(data[0]) * self.w1[i]
                net += float(data[1]) * self.w1[i + self.num_layer_hid]
                net += self.w1[i + self.num_layer_in * self.num_layer_hid]
                net_hid.append(net)
                out_hid.append(f(net))

            # Calculate net input and output for output layer
            net = 0
            for i in range(self.num_layer_hid):
                net += out_hid[i] * self.w2[i]
            net += self.w2[self.num_layer_hid * self.num_layer_out]
            net_out = net
            out_out = f(net)

            # w2 weights update
            for i in range(len(self.w2)):
                if i < len(self.w2) - self.num_layer_out:
                    update = self.learning_rate * (float(data[2]) - out_out) * df(net_out) * out_hid[i]
                else:
                    update = self.learning_rate * (float(data[2]) - out_out) * df(net_out)
                updates_w2.append(update)

            # w1 weights update
            for i in range(len(self.w1)):
                if i < len(self.w1) - self.num_layer_hid:
                    if i < 4:
                        update = self.learning_rate * (float(data[2]) - out_out) * df(net_out) * self.w2[i % 4] * df(net_hid[i % 4]) * float(data[0])
                    else:
                        update = self.learning_rate * (float(data[2]) - out_out) * df(net_out) * self.w2[i % 4] * df(net_hid[i % 4]) * float(data[1])
                else:
                    update = self.learning_rate * (float(data[2]) - out_out) * df(net_out) * self.w2[i % 4] * df(net_hid[i % 4])
                updates_w1.append(update)

            # update weights w1
            for i, update in enumerate(updates_w1):
                self.w1[i] += update

            # update weights w2
            for i, update in enumerate(updates_w2):
                self.w2[i] += update


def main():
    network = Network(2, 4, 1)
    train_data = []
    f = open('./input_dataset.in')

    try:

        while 1: 
            new_in = f.readline().split(',')
            if len(new_in) < 3:
                print(train_data)
                break
            train_data.append(new_in)
            

        for _ in range(100):
            print('epoch')
            network.train(train_data)

        f = open('./testing_dataset.in')

        while 1:
            new_in = f.readline().split(',')
            if len(new_in) < 2:
                break
            new_in[0] = new_in[0][new_in[0].rfind(' ') + 1:]
            new_in[1] = new_in[1][new_in[1].rfind(' ') + 1:]
            output = network.forward_propagation(new_in)
            if output > 0:
                print('+1')
            else:
                print('-1')

    except EOFError:
        print('', end='')


if __name__ == '__main__':
    main()
