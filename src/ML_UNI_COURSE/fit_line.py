# Main
import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def mylinfit( x , y ):
    n = len(x)
    xy = sum(x*y)
    ysum = sum(y)
    xsum = sum(x)
    x2sum = sum(x*x)
    sumx2 = sum(x)*sum(x)
    a = (n*xy-ysum*xsum)/(n*x2sum-sumx2)
    b = sum(y)/n-a*sum(x)/n
    return a , b

x = []
y = []

# Setting up the figure and axis
figure, ax = plt.subplots()
ax.set_title('Left click to add at least 3 points. Right click to fit model')

def onclick(event):
    # Left click to collect point
    if event.button == 1:  
        x.append(event.x)
        y.append(event.y)
        plt.scatter(event.x, event.y, color='blue')
        plt.draw()
    # Right click to stop collecting
    elif event.button == 3:  
        plt.close()  # Close the plot window

# Connect the event to the onclick function and plot it. 
# The onclick function also closes the plot
cid = figure.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# If there are no points collected, exit the program
if len(x) <= 2:
    print("Please plot at least 3 points")
else:
    # Convert points to numpy arrays as integers
    x = np.array(x).astype("int")
    y = np.array(y).astype("int")

    # Fitting a line to the model and creating a new plot for it.
    a , b = mylinfit(x, y)
    plt.plot(x, y,'kx')
    plt.plot(x, a*x+b, 'r-' )
    MSE = np.sum(np.square((y-(a*x+b))))/y.size
    print(f"Myfit : a={b} and b={b} and MSE={MSE}" )
    plt.show()