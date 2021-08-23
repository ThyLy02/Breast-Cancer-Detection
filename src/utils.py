import matplotlib.pyplot as plt
def plot(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('Loss & Accuracy')
    plt.ylabel('Values')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Acc'])
    plt.show()
