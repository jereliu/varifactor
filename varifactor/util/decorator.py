import os
import seaborn as sns
import matplotlib.pyplot as plt


def defunct(func):
    def func_wrapper(*args, **kwargs):
        print "varifactor: This method is now defunct and will be removed from later versions"
        return func(*args, **kwargs)

    return func_wrapper


def add_save_plot_option(func):
    """
    add option to save plot to address specified by save_addr
    :param func:
    :return:
    """
    def func_wrapper(save_size, save_addr=None, *args, **kwargs):
        # create save directory if not exist
        if save_addr is not None:
            if not os.path.isdir(os.path.dirname(save_addr)):
                os.mkdir(os.path.dirname(save_addr))
            plt.ioff()

        func(*args, **kwargs)

        # optionally, save
        if save_addr is not None:
            fig = plt.gcf()
            fig.set_size_inches(save_size[0], save_size[1])
            fig.savefig(save_addr, bbox_inches='tight')
            plt.close()
            plt.ion()

    return func_wrapper


if __name__ == "__main__":
    @defunct
    def get_text(name):
        return "Hello " + name