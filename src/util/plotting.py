import matplotlib.pyplot as plt

def plot_images(_images: dict, _cols = 2, _rows=1):
    pos = 100*_rows+10*_cols+1
    fig = plt.figure()
    fig.tight_layout()
    for key in _images.keys():
        plt.subplot(pos)
        plt.title(key)
        plt.imshow(_images[key])
        pos += 1
    plt.show()