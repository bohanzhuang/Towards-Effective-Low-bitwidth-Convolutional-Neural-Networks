"""
write readme.md
draw figures
save log
copy code
and so on
"""
import os
import shutil
from .resultcurve import DrawCurves, DrawWeightHistogram
from .graphgen import Graph


__all__ = ["Visualization"]

class Visualization(object):
    """
    create package of experimental results
    """
    def __init__(self, save_path, logger):

        self.save_path = save_path

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # define path of specific files
        self.log_file = os.path.join(self.save_path, "log.txt")
        self.readme = os.path.join(self.save_path, "README.md")
        self.settings_file = os.path.join(self.save_path, "settings.log")
        self.weight_file = os.path.join(self.save_path, "weight_file.npy")
        self.code_path = os.path.join(self.save_path, "code/")

        # # if file is created, remove it
        # if os.path.isfile(self.log_file):
        #     os.remove(self.log_file)
        # if os.path.isfile(self.readme):
        #     os.remove(self.readme)
        # if not os.path.isdir(self.code_path):
        #     os.makedirs(self.code_path)

        # copy code to folder
        self.copy_code(logger, src=os.path.abspath('./'), dst=self.code_path)

        logger.info("|===>Result will be saved at {}".format(self.save_path))

    def copy_code(self, logger, src=os.path.abspath('./'), dst="./code/"):
        """
        copy code in current path to a folder
        """
        for file in os.listdir(src):
            if "specific_experiments" in file or "log" in file:
                continue
            src_file = os.path.join(src, file)
            file_split = file.split('.')
            if len(file_split) >= 2 and file_split[1] == "py":
                if not os.path.isdir(dst):
                    os.makedirs(dst)
                dst_file = os.path.join(dst, file)
                try:
                    shutil.copyfile(src=src_file, dst=dst_file)
                except:
                    logger.errro("copy file error! src: {}, dst: {}".format(src_file, dst_file))
            elif os.path.isdir(src_file):
                deeper_dst = os.path.join(dst, file)
                self.copy_code(logger, src=src_file, dst=deeper_dst)

    def write_settings(self, settings):
        """
        save expriment settings to a file
        """
        with open(self.settings_file, "w") as f:
            for k, v in settings.__dict__.items():
                f.write(str(k)+": "+str(v)+"\n")

    def write_log(self, input_data):
        """
        save log
        """
        txt_file = open(self.log_file, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def write_readme(self, input_data):
        """
        save extra information to readme.md
        """
        txt_file = open(self.readme, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def draw_curves(self):
        """
        draw experimental curves
        """
        drawer = DrawCurves(file_path=self.log_file, fig_path=self.save_path)
        drawer.draw(target="test_error")
        drawer.draw(target="train_error")
    
    def draw_histogram(self, X, layer_name):
        """
        draw weight histogram
        """
        drawer = DrawWeightHistogram(fig_folder=self.save_path)
        drawer.draw(X, layer_name)

    def save_network(self, var):
        """
        draw network structure
        """
        graph = Graph()
        graph.draw(var=var)
        graph.save(file_name=self.save_path+"network.svg")
