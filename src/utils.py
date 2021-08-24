import matplotlib.pyplot as plt


def plot_model(history):
    # plt.figure()
    x = plt.plot(history.history['loss'])
    y = plt.plot(history.history['accuracy'])
    plt.title('Loss & Accuracy')
    plt.ylabel('Values')
    plt.xlabel('Epoch')
    return plt.show()
