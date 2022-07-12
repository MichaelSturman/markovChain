import numpy as np
import matplotlib.pyplot as plt


def monteCarlo(numPoints, start, end, step):
    # Define range to integrate
    x = np.arange(start, end, step)

    # Define function to integrate
    def f(x):
        return x**2

    # Analytic solution
    def fInt(a, b, step):
        return (1/3) * ((a-step)**3 - b**3)

    aSolution = fInt(end, start, step)

    # Define an upper limit of the area we'll be generating random points in.
    # Calculate yMax for the range above and add some wriggle room (say 10%)
    yMax = 1.05*np.max(f(x))

    # Throw (x,y) random points down at our area. If the y value of the point is
    # less than or equal to f(x), add +1 to successful count. At the end of the
    # iterations, multiple the fraction of successes by total area of the rectangle
    success = 0

    iPlot = []
    convergenceCheck = []
    mse = []

    # np.random.rand generates random numbers between 0 and 1. Multiply the x
    # part by the x range and the y part by the y range
    for i in range(0, numPoints):
        xy = np.random.rand(1, 2)
        if (xy[0][1]*yMax <= f(xy[0][0]*(end - step))):
            success += 1

        if i!=0 and i%50 == 0:
            iPlot.append(i)
            convergenceCheck.append((success/i) * ((end - step) - start) * yMax)
            mse.append(((success/i) * ((end - step) - start) * yMax - aSolution)**2)

    # Plot showing numerical calculation of integral converging to the analytic
    # value and a second showin the MSE.
    plt.scatter(np.log10(iPlot), convergenceCheck, s=1, c='r', marker='o')
    plt.ylim(yMax*1.25)
    plt.axhline(y = aSolution, color='b', linestyle = 'dashed')
    plt.ylabel('Computed Integral Value')
    plt.xlabel(r'$log_{10}(i)$')
    plt.show()

    plt.scatter(np.log10(iPlot), mse, s=1, c='r', marker='o')
    plt.ylabel('Mean Squared Error')
    plt.xlabel(r'$log_{10}(i)$')
    plt.show()

    area = (success/numPoints) * ((end - step) - start) * yMax
    perDiff = 100*(aSolution - area)/area

def main():
    monteCarlo(50000, 0, 10.1, 0.1)

if __name__ == "__main__":
    main()
