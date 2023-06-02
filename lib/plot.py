import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib

def plot(X_train, X_test, y_train, y_test, label_encoder):
    matplotlib.interactive(False)

    # Plot the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.imshow(np.swapaxes(X_train[0], 0 ,1), interpolation='nearest', cmap=cm.viridis, origin='lower', aspect='auto')
    ax1.set_title(label_encoder.inverse_transform([y_train[0]])[0], {'fontsize':20, 'fontweight':'bold'})
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax2.imshow(np.swapaxes(X_train[1], 0 ,1), interpolation='nearest', cmap=cm.viridis, origin='lower', aspect='auto')
    ax2.set_title(label_encoder.inverse_transform([y_train[1]])[0], {'fontsize':20, 'fontweight':'bold'})
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax3.imshow(np.swapaxes(X_train[2], 0 ,1), interpolation='nearest', cmap=cm.viridis, origin='lower', aspect='auto')
    ax3.set_title(label_encoder.inverse_transform([y_train[2]])[0], {'fontsize':20, 'fontweight':'bold'})
    ax3.set_ylim(ax3.get_ylim()[::-1])
    ax4.imshow(np.swapaxes(X_train[3], 0 ,1), interpolation='nearest', cmap=cm.viridis, origin='lower', aspect='auto')
    ax4.set_title(label_encoder.inverse_transform([y_train[3]])[0], {'fontsize':20, 'fontweight':'bold'})
    ax4.set_ylim(ax4.get_ylim()[::-1])
    fig.set_size_inches(18,12)
    plt.show(block=True)