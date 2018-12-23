import json


class Hparams(object):
    def __init__(self, d):
        if type(d) is str:
            with open(d, "r") as file:
                d = json.load(file)
        self.convert_json(d)

    def convert_json(self, d):
        self.__dict__ = {}
        for key, value in d.items():
            if type(value) is dict:
                value = convert_json(value)

            if key == "input_shape":
                if len(value) <= 2:
                    value = (val[0], value[1], 3)
                elif value[0] == 3:
                    value = (value[1], value[2], value[0])
            self.__dict__[key] = value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Hparams, self).__delitem__(key)
        del self.__dict__[key]
        return


if __name__ == "__main__":
    hparams = Hparams("../hyperparam.json")
    print(hparams.input_shape)