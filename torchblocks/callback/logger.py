import os
import time
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict
from torchblocks.utils.paths import save_json,ensure_dir
plt.switch_backend('agg')  # 防止ssh上绘图问题


class TrainLogger:

    def __init__(self, log_dir, prefix='', log_file_level=logging.NOTSET):
        self.prefix = prefix
        self.dicts = defaultdict(list)
        self.log_file_level = log_file_level
        self.log_dir = os.path.join(log_dir, 'logs')
        ensure_dir(self.log_dir)
        time_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_file = os.path.join(self.log_dir, self.prefix + f"_{time_}.log")
        self.json_file = os.path.join(self.log_dir, self.prefix + f"_{time_}.json")
        self.init_logger()

    def init_logger(self):
        '''
        初始化logger
        '''
        log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        self.logger.handlers = [console_handler]
        if self.log_file and self.log_file != '':
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_file_level)
            self.logger.addHandler(file_handler)

    def info(self, msg, *args, **kwargs):
        '''
        Log 'msg % args' with severity 'INFO'.
        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        '''
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        '''
        Log 'msg % args' with severity 'debug'.
        logger.debug("Houston, we have a %s", "interesting problem", exc_info=1)
        '''
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        '''
        Log 'msg % args' with severity 'warning'.
        logger.warning("Houston, we have a %s", "interesting problem", exc_info=1)
        '''
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        '''
        Log 'msg % args' with severity 'error'.
        logger.error("Houston, we have a %s", "interesting problem", exc_info=1)
        '''
        self.logger.error(msg, *args, **kwargs)

    def add_value(self, value, step=None, name='loss'):
        if step is not None:
            assert isinstance(step, int)
        _dict = {name: value, 'step': step, }
        self.dicts[name].append(_dict)

    def save(self, plot=True):
        save_json(data=self.dicts,file_path=self.json_file)
        if plot:
            self.plot()

    def plot(self):
        keys = list(self.dicts.keys())
        for key in keys:
            values = self.dicts[key]
            png_file = os.path.join(self.log_dir, f"{self.prefix}_{key}")
            label = key
            if '_' not in key:
                label = f'train_{key}'
            values = sorted(values, key=lambda x: x['step'])
            x = [i['step'] for i in values]
            y = [i[key] for i in values]
            plt.style.use("ggplot")
            fig = plt.figure(figsize=(15, 5), facecolor='w')
            ax = fig.add_subplot(111)
            if "eval_" in key:
                y = [round(float(x), 2) for x in y]
            ax.plot(x, y, label=label)
            if key == 'learning_rate':
                # 科学计数法显示
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            ax.legend()
            plt.xlabel("Step #")
            plt.ylabel(key)
            plt.title(f"Training {key} [Step {x[-1]}]")
            plt.savefig(png_file)
            plt.close()
