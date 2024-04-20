# kurtis bertauche
# 16 april 2024
# project 3 - cs 455

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os

NUM_NODES = 50
DIMENSIONS = 2
VARIANCE = 2
GROUND_TRUTH = 50
R = 15
np.random.seed(854) #6

def euclidean_norm(vector):
    """
    Input: Vector (x, y)
    Output: Euclidean Norm: Scalar
    """
    squaredSum = 0
    for component in vector:
        squaredSum += (component * component)
    return sqrt(squaredSum)

def tuple_diff(vector1, vector2):
    """
    Input: Two vectors (x, y), (x, y)
    Output: One vector (x, y), result of vector1 - vector2
    """
    return tuple(np.subtract(vector1, vector2))

def getNumNeighbors(nodePositions, nodeNum):

    j = 0
    for i in range(NUM_NODES):
        if i == nodeNum:
            continue
        thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][nodeNum], nodePositions[1][nodeNum])))
        if thisPairNorm < R:
            j += 1
    return j

def getNeighbors(nodePositions, nodeNum):
    neighbors = []
    for i in range(NUM_NODES):
        if i == nodeNum:
            continue
        thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][nodeNum], nodePositions[1][nodeNum])))
        if thisPairNorm < R:
            neighbors.append(i)
    return neighbors

def plotNodesAndSave(nodePositions, fileName):

    plt.clf()
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):

            # nodes should not be used with themselves
            if i == j:
                continue

            thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][j], nodePositions[1][j])))
            if thisPairNorm < R:
                # add line to plot for part 4
                plt.plot([nodePositions[0][i], nodePositions[0][j]],[nodePositions[1][i], nodePositions[1][j]], color = "blue", linewidth = "0.5")
    plt.scatter(nodePositions[0], nodePositions[1], marker = ">", color = "magenta")
    plt.gcf().gca().set_aspect("equal")
    plt.savefig(fileName)

def updateMeasurements(oldMeasures, nodePositions, method):

    newMeasures = np.full((NUM_NODES, 1), 0.0)

    for i in range(NUM_NODES):
        thisNewMeasure = 0.0

        thisWii = 0
        if method == "metropolis":
            thisWii = getMetropolisWeightsOfij(nodePositions, i, i)
        else:
            thisWii = getMaxDegreeWeightsOfij(nodePositions, i, i)

        #print("thisWii: ", thisWii, end = " ")
        thisNewMeasure += thisWii * oldMeasures[i]
        #print(thisNewMeasure)
        #print(R)
        neighborsOfi = getNeighbors(nodePositions, i)
        for neighbor in neighborsOfi:
            thisWij = 0
            if method == "metropolis":
                thisWij = getMetropolisWeightsOfij(nodePositions, i, neighbor)
            else:
                thisWij = getMaxDegreeWeightsOfij(nodePositions, i, neighbor)
            thisNewMeasure += thisWij * oldMeasures[neighbor]
            #print("thisWij: ", thisWij, end = "")
        #print("")
        #print("thisNewMeasure: ", thisNewMeasure)

        newMeasures[i] = thisNewMeasure
    return newMeasures
        
def getMaxDegreeWeightsOfij(nodePositions, i, j):
    if i != j:
        return 1 / NUM_NODES
    elif i == j:
        numNeighborsOfi = getNumNeighbors(nodePositions, i)
        return 1 - (numNeighborsOfi / NUM_NODES)
    else:
        return 0
    
def getMetropolisWeightsOfij(nodePositions, i, j):
    if i != j:
        numNeighborsOfi = getNumNeighbors(nodePositions, i)
        numNeighborsOfj = getNumNeighbors(nodePositions, j)
        return 1 / (1 + max(numNeighborsOfi, numNeighborsOfj))
    elif i == j:
        neighborsOfI = getNeighbors(nodePositions, i)
        sum = 0
        for neighbor in neighborsOfI:
            sum += getMetropolisWeightsOfij(nodePositions, i, neighbor)
        return 1 - sum
    else:
        return 0

def createPlots(measurements, maxNode, minNode, case, baseFilename):
    plotErrorConvergence(measurements, case, baseFilename)
    plotFinalInital(measurements, case, baseFilename)
    plotMSEConverg(measurements, case, baseFilename)
    plotMostMinNeighborsError(measurements, maxNode, minNode, case, baseFilename)
    plotMostMinNeighborsSquareError(measurements, maxNode, minNode, case, baseFilename)

def plotErrorConvergence(measurements, case, baseFileName):
    plt.clf()
    for i in range(NUM_NODES):
        plt.plot(GROUND_TRUTH - measurements[i, :])
    plt.title("Error Convergence\n" + case)
    plt.xlabel("Iteration")
    plt.ylabel("Raw Error")
    plt.savefig(baseFileName + "/" + baseFileName + "_ConvergenceError.png")

def plotFinalInital(measurements, case, baseFileName):
    plt.clf()
    plt.title("Initial and Final States\n" + case)
    plt.xlabel("Node")
    plt.ylabel("Measurment Value")
    plt.plot(measurements[:, 0], color = "red", marker = "o", label = "Initial Measurement")
    plt.plot(measurements[:, measurements.shape[1] - 1], color = "blue", marker = "o", label = "Final Measurement")
    plt.legend(loc = "upper left")
    plt.savefig(baseFileName + "/" + baseFileName + "_finalInitial.png")

def plotMSEConverg(measurements, case, baseFileName):
    plt.clf()
    plt.title("Mean Square Error Convergence\n" + case)
    plt.xlabel("Mean Square Error")
    plt.ylabel("Iteration")
    for i in range(NUM_NODES):
        plt.plot((GROUND_TRUTH - measurements[i, :])**2)
    plt.savefig(baseFileName + "/" + baseFileName + "_meanSquareConergenve.png")

def plotMostMinNeighborsError(measurements, maxNode, minNode, case, baseFileName):
    plt.clf()
    plt.title("Max and Min Neighbor Error Convergence\n" + case)
    plt.xlabel("Error")
    plt.ylabel("Iteration")
    plt.plot(GROUND_TRUTH - measurements[maxNode, :], label = "Max Neighbors")
    plt.plot(GROUND_TRUTH - measurements[minNode, :], label = "Min Neighbors")
    plt.legend(loc = "upper right")
    plt.savefig(baseFileName + "/" + baseFileName + "_maxMinNeighborsErrorConvergence")

def plotMostMinNeighborsSquareError(measurements, maxNode, minNode, case, baseFileName):
    plt.clf()
    plt.title("Max and Min Neighbor Mean Square Error Convergence\n" + case)
    plt.xlabel("Error")
    plt.ylabel("Iteration")
    plt.plot((GROUND_TRUTH - measurements[maxNode, :])**2, label = "Max Neighbors")
    plt.plot((GROUND_TRUTH - measurements[minNode, :])**2, label = "Min Neighbors")
    plt.legend(loc = "upper right")
    plt.savefig(baseFileName + "/" + baseFileName + "_maxMinNeighborsMSEConvergence")


def main():

    # make dirs
    try:
        os.mkdir("staticMetropolis")
    except:
        pass
    try:
        os.mkdir("staticMaxDegree")
    except:
        pass
    try:
        os.mkdir("dynamicMetropolis")
    except:
        pass
    try:
        os.mkdir("dynamicMaxDegree")
    except:
        pass

    # setup initial states
    measures = GROUND_TRUTH + np.random.uniform(size=(NUM_NODES, 1), high=VARIANCE, low=-VARIANCE)
    nodePositions = np.random.randint(low = 0, high = 50, size = (DIMENSIONS, NUM_NODES))
    allMeasures = measures.copy()
    plotNodesAndSave(nodePositions, "network.png")

    # get the node with min and max neighbors
    minNeighborsNode = -1
    minNeighbors = 999
    maxNeighbors = -1
    maxNeighbordsNode = -1
    for i in range(NUM_NODES):
        numNeighbors = getNumNeighbors(nodePositions, i)
        if numNeighbors > maxNeighbors:
            maxNeighbors = numNeighbors
            maxNeighbordsNode = i
        if numNeighbors < minNeighbors:
            minNeighbors = numNeighbors
            minNeighborsNode = i


    #################################
    # CASE ONE - MAX DEGREE
    #################################

    print("=" * 20, " BEGINNING STATIC MAX DEGREE SIMULATION ", "=" * 20)
    for i in range(500):
        measures = updateMeasurements(measures, nodePositions, "maxdegree")
        allMeasures = np.concatenate((allMeasures, measures.copy()), axis = 1)
        print("| COMPLETED ITERATION: ", i, " " * 53, "|")
    print("=" * 21, " FINISHED STATIC MAX DEGREE SIMULATION ", "=" * 21)
    print("=" * 21, " PLOTTING", "=" * 21)
    createPlots(allMeasures, maxNeighbordsNode, minNeighborsNode, "Static Max Degree", "staticMaxDegree")

    #################################
    # CASE TWO - METROPOLIS
    #################################

    measures = GROUND_TRUTH + np.random.uniform(size=(NUM_NODES, 1), high=VARIANCE, low=-VARIANCE)
    allMeasures = measures.copy()
    print("=" * 20, " BEGINNING STATIC METROPOLIS SIMULATION ", "=" * 20)
    for i in range(200):
        measures = updateMeasurements(measures, nodePositions, "metropolis")
        allMeasures = np.concatenate((allMeasures, measures.copy()), axis = 1)
        print("| COMPLETED ITERATION: ", i, " " * 53, "|")
    print("=" * 21, " FINISHED STATIC METROPOLIS SIMULATION ", "=" * 21)
    print("=" * 21, " PLOTTING", "=" * 21)
    createPlots(allMeasures, maxNeighbordsNode, minNeighborsNode, "Static Metropolis", "staticMetropolis")

    #################################
    # CASE THREE DYANMIC - MAX DEGREE
    #################################
    global R
    measures = GROUND_TRUTH + np.random.uniform(size=(NUM_NODES, 1), high=VARIANCE, low=-VARIANCE)
    allMeasures = measures.copy()
    print("=" * 20, " BEGINNING DYNAMIC MAX DEGREE SIMULATION ", "=" * 20)
    for i in range(500):
        R = R - np.random.uniform(-0.5, 0.5)
        measures = updateMeasurements(measures, nodePositions, "maxdegree")
        allMeasures = np.concatenate((allMeasures, measures.copy()), axis = 1)
        print("| COMPLETED ITERATION: ", i, " " * 53, "|")
    print("=" * 21, " FINISHED DYNAMIC MAX DEGREE SIMULATION ", "=" * 21)
    print("=" * 21, " PLOTTING", "=" * 21)
    createPlots(allMeasures, maxNeighbordsNode, minNeighborsNode, "Dynamic Max Degree", "dynamicMaxDegree")

    #################################
    # CASE FOUR DYNAMIC - METROPOLIS
    #################################
    R = 15
    measures = GROUND_TRUTH + np.random.uniform(size=(NUM_NODES, 1), high=VARIANCE, low=-VARIANCE)
    allMeasures = measures.copy()
    print("=" * 20, " BEGINNING DYNAMIC METROPOLIS SIMULATION ", "=" * 20)
    for i in range(200):
        R = R - np.random.uniform(-0.5, 0.5)
        measures = updateMeasurements(measures, nodePositions, "metropolis")
        allMeasures = np.concatenate((allMeasures, measures.copy()), axis = 1)
        print("| COMPLETED ITERATION: ", i, " " * 53, "|")
    print("=" * 21, " FINISHED DYNAMIC METROPOLIS SIMULATION ", "=" * 21)
    print("=" * 21, " PLOTTING", "=" * 21)
    createPlots(allMeasures, maxNeighbordsNode, minNeighborsNode, "Dynamic Metropolis", "dynamicMetropolis")


if __name__ == "__main__":
    main()