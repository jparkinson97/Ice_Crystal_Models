import typing
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

alpha = 1
gamma = 0.001
beta = 0.4

class State(Enum):
    frozen = 1
    boundary = 2
    nonReceptive = 3

class Cell:
    def __init__(self, u: float, v: float, state: State):
        self.u = u
        self.v = v
        self.vapourLevel = self.u + self.v
        self.state = state
    
    def crystallise(self):
        self.v = self.u
        self.u = 0
        self.state = State.frozen

class Grid:
    def __init__(self):
        self.points = np.array([[(0, 0, Cell(0, 1, State.frozen))]])
        for _ in range(40):
            self.expand()
    
    def visualise(self):
        frozenPoints = self.points[self.points[:, :, 2] == State.frozen.value][:, :, :2]
        boundaryPoints = self.points[self.points[:, :, 2] == State.boundary.value][:, :, :2]
        nonReceptivePoints = self.points[self.points[:, :, 2] == State.nonReceptive.value][:, :, :2]
        
        if len(frozenPoints) > 0:
            plt.scatter(frozenPoints[:, :, 0].flatten(), frozenPoints[:, :, 1].flatten(), s=2, c='blue')
        if len(boundaryPoints) > 0:
            plt.scatter(boundaryPoints[:, :, 0].flatten(), boundaryPoints[:, :, 1].flatten(), s=2, c='red')
        if len(nonReceptivePoints) > 0:
            plt.scatter(nonReceptivePoints[:, :, 0].flatten(), nonReceptivePoints[:, :, 1].flatten(), s=2, c='green')
        
        plt.show()
    
    def expand(self):
        newPoints = []
        for point in self.points[:, :, :2].reshape(-1, 2):
            neighbours = find_neighbours(tuple(point))
            newPoints.extend(neighbours)
        
        newPoints = np.array(newPoints)
        mask = ~np.isin(newPoints, self.points[:, :, :2]).all(axis=1)
        newPoints = newPoints[mask]
        newCells = np.array([Cell(beta, 0, State.nonReceptive) for _ in range(newPoints.shape[0])], dtype=object)
        newCells = newCells.reshape(newPoints.shape[0], 1)
        newPoints = np.concatenate((newPoints, newCells), axis=1)
        
        self.points = np.append(self.points, newPoints[:, None], axis=1)

    def constant_addition(self):
        self.points[:, :, 2] = np.where(self.points[:, :, 2] != State.nonReceptive.value,
                                        Cell(self.points[:, :, 2].u, self.points[:, :, 2].v + gamma,
                                             self.points[:, :, 2].state),
                                        self.points[:, :, 2])
    
    def diffuse(self):
        newPoints = np.copy(self.points)
        for i, j in np.ndindex(self.points.shape[:2]):
            if self.points[i, j, 2].state != State.frozen.value:
                totalUnfrozenNeighbours = 0
                totalU = 0
                neighbours = find_neighbours(self.points[i, j, :2])
                for neighbour in neighbours:
                    if np.isin(neighbour, self.points[:, :, :2]).all(axis=2).any():
                        neighbourCell = self.points[(self.points[:, :, :2] == neighbour).all(axis=2)].squeeze()
                        if neighbourCell[2].state != State.frozen.value:
                            totalUnfrozenNeighbours += 1
                            totalU += neighbourCell[2].u
                
                averageU = totalU / max(totalUnfrozenNeighbours, 1)
                u = self.points[i, j, 2].u + (alpha / 2) * (averageU - self.points[i, j, 2].u)
                
                if self.points[i, j, 2].vapourLevel > 1:
                    newPoints[i, j, 2] = Cell(0, u + self.points[i, j, 2].v, State.frozen)
                elif self.points[i, j, 2].state == State.boundary.value:
                    newPoints[i, j, 2] = Cell(0, u + self.points[i, j, 2].v, State.nonReceptive)
                else:
                    newPoints[i, j, 2] = Cell(u, self.points[i, j, 2].v, State.nonReceptive)
        
        self.points = newPoints
    
    def set_boundary_points(self):
        for i, j in np.ndindex(self.points.shape[:2]):
            if self.points[i, j, 2].state == State.frozen.value:
                neighbours = find_neighbours(self.points[i, j, :2])
                for neighbour in neighbours:
                    if np.isin(neighbour, self.points[:, :, :2]).all(axis=2).any():
                        neighbourCell = self.points[(self.points[:, :, :2] == neighbour).all(axis=2)].squeeze()
                        if neighbourCell[2].state != State.frozen.value:
                            neighbourCell[2].state = State.boundary.value

def find_neighbours(coordinates: tuple):
    x, y = coordinates
    neighbours = np.array([(x, y + 2), (x, y - 2), (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1)])
    return neighbours

grid = Grid()

for _ in range(400):
    grid.set_boundary_points()
    grid.constant_addition()
    grid.diffuse()

grid.visualise()
