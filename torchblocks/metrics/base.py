class Metric:

    def update(self, outputs, target):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def reset(self):
        pass
