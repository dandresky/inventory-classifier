import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt


def main():
    start_time = dt.datetime.now()

    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Test')
    plt.plot(x,y)

    save_dir = os.path.join(os.getcwd(), 'saved_imgs')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, 'plot')
    plt.savefig(img_path)
    #plt.show()

    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")
    pass


if __name__ == '__main__':
    main()
