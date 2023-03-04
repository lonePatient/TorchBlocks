import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict
from ..utils.io_utils import save_json, build_dir
plt.switch_backend('agg')  # 防止ssh上绘图问题

FILE_NAME = 'training_info.json'


class FileWriter:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scale_dicts = defaultdict(list)
        build_dir(self.log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None):
        if global_step is not None:
            global_step = int(global_step)
        _dict = {tag: scalar_value, 'step': global_step}
        self.scale_dicts[tag].append(_dict)

    def save(self, plot=True):
        save_path = os.path.join(self.log_dir, FILE_NAME)
        save_json(data=self.scale_dicts, file_path=save_path)
        if plot:
            self.plot()

    def plot(self):
        keys = list(self.scale_dicts.keys())
        for key in keys:
            values = self.scale_dicts[key]
            name = key.split("/")[-1] if "/" in key else key
            png_file = os.path.join(self.log_dir, f"{name}")

            values = sorted(values, key=lambda x: x['step'])
            x = [i['step'] for i in values]
            y = [v[key] for v in values]

            plt.style.use("ggplot")
            fig = plt.figure(figsize=(15, 5), facecolor='w')
            ax = fig.add_subplot(111)
            if "eval_" in name:
                x = [str(v) for v in x]
                y = [round(float(v), 4) for v in y]
            ax.plot(x, y, label=name)
            if key == 'train_lr':
                # 科学计数法显示
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            ax.legend()
            plt.xlabel("Step #")
            plt.ylabel(name)
            plt.title(f"Training {name} [Step {x[-1]}]")
            plt.savefig(png_file)
            plt.close()

    def close(self):
        pass
