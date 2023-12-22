import matplotlib.pyplot as plt

def plotGraph(data):
    plt.plot(data.index, data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Close Price Over Time')
    plt.grid(True)
    plt.show()

