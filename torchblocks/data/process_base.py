class ProcessBase(object):
    """ 用于处理单个example """

    def __call__(self, example):
        raise NotImplementedError('Method [__call__] should be implemented.')
