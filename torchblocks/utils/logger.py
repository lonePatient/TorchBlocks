import os
import time
import logging


class Logger:
    '''
    Base class for experiment loggers.
    日志模块
    '''

    def __init__(self, opts, log_file_level=logging.NOTSET):
        self.opts = opts
        self.log_file_level = log_file_level
        self.setup_logger()
        self.info = self.logger.info
        self.debug = self.logger.debug
        self.error = self.logger.error
        self.warning = self.logger.warning

    def setup_logger(self):
        log_file_path = self.setup_log_path()
        fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        dmt = '%Y-%m-%d %H:%M:%S'
        log_format = logging.Formatter(fmt=fmt, datefmt=dmt)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        self.logger.handlers = [console_handler]
        if log_file_path and log_file_path != '':
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(self.log_file_level)
            self.logger.addHandler(file_handler)

    def setup_time(self):
        local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        return local_time

    def setup_log_path(self):
        log_time = self.setup_time()
        log_prefix = self.setup_prefix()
        log_file_name = f"{self.opts.task_name}-{self.opts.model_type}-" \
                        f"{self.opts.experiment_code}-{log_prefix}-{log_time}.log"
        log_file_path = os.path.join(self.opts.output_dir, log_file_name)
        return log_file_path

    def setup_prefix(self):
        if self.opts.do_train:
            return 'train'
        elif self.opts.do_eval:
            return 'eval'
        elif self.opts.do_predict:
            return 'predict'
        else:
            return ''

