import typing
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

alpha = 1
gamma = 0.001
beta = 0.4

class State(Enum):
    frozen = 1
    boundary = 2
    nonReceptive = 3
    
class Cell:
    def __init__(self,u:float, v:float, state:State):
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
        self.points:dict[tuple, Cell] = {(0,0): Cell(0, 1, State.frozen)}
        for i in range(40):
            self.expand()
        
    def visualise(self):
        frozenPoints = []
        boundaryPoints = []
        nonReceptivePoints = []
        
        for coordinates, cell in self.points.items():
            # would use match case but using old python 
            if cell.state == State.frozen:
                frozenPoints.append(coordinates)
            if cell.state == State.boundary:
                boundaryPoints.append(coordinates)
            if cell.state == State.nonReceptive:
                nonReceptivePoints.append(coordinates)
                
        for points in [frozenPoints, boundaryPoints, nonReceptivePoints]:
            pointArray = np.array(points)
            if len(points) >0:
                transpose = pointArray.T
                x, y = transpose
                
                if points == frozenPoints:
                    plt.scatter(x,y, s=2, c= 'blue')
                if points == boundaryPoints:
                    plt.scatter(x,y, s=2, c= 'red')
                if points == nonReceptivePoints:
                    plt.scatter(x,y, s=2, c= 'green')
                    
        plt.show()
    
    def expand(self):
        """help initiaise the existing grid of water molecules"""
        newPoints = set()
        for point in self.points.keys():
            neighbours= find_neighbours(point)
            for neighbour in neighbours:
                newPoints.add(neighbour)
        
        for point in newPoints:
            if point not in self.points.keys():
                self.points[point] = Cell(beta, 0, State.nonReceptive) 
                
    def constant_addition(self):
        for coordinates, cell in self.points.items():
            if cell.state != State.nonReceptive:
                self.points[coordinates] = Cell(cell.u, cell.v+gamma, cell.state)
            
                
    def diffuse(self):
        """take one step to simulate freezing"""
        newPoints = dict()
        for coordinates, cell in self.points.items(): 
            if cell.state != State.frozen:
                totalUnfrozenNeighbours = 0
                totalU = 0
                neighbours = find_neighbours(coordinates)
                for neighbour in neighbours:
                    if neighbour not in self.points.keys():
                        totalU = cell.u 
                        totalUnfrozenNeighbours = 1
                        break #case that the cell is on the edge, keep water level at beta
                    
                    neighbourCell = self.points[neighbour]
                    if neighbourCell.state != State.frozen:
                        totalUnfrozenNeighbours +=1
                    totalU += neighbourCell.u
                    
                averageU = totalU/max(totalUnfrozenNeighbours, 1)
                
                u = cell.u + (alpha/2)*(averageU -cell.u)
                
                if cell.vapourLevel >1:
                    newPoints[coordinates] = Cell(0, u + cell.v, State.frozen)
                elif cell.state == State.boundary:
                    newPoints[coordinates] = Cell(0, u + cell.v, State.nonReceptive)
                else:
                    newPoints[coordinates] = Cell(u, cell.v, State.nonReceptive)
                        
                for coordinates, cell in newPoints.items():
                    self.points[coordinates] = cell
        
    def set_boundary_points(self):
        boundaryPoints = dict()
        for coordinates, cell in self.points.items():
            if cell.state == State.frozen:
                neighbours = find_neighbours(coordinates)
                for neighbour in neighbours:
                    if neighbour not in self.points.keys():
                        continue # case that it is trying to expand beyond the border
                    if self.points[neighbour].state != State.frozen:
                        self.points[neighbour].state = State.boundary
    
def find_neighbours(coordinates: tuple):
    """takes a tuple and returns a list of tuple"""
    neighbours = []
    x = coordinates[0] 
    y = coordinates[1]
    #up and down 
    neighbours.append((x, y+2))
    neighbours.append((x, y-2))
    #the four corners
    neighbours.append((x+2, y+1))
    neighbours.append((x+2, y-1))
    neighbours.append((x-2, y+1))
    neighbours.append((x-2, y-1))  
    
    return neighbours
    
grid = Grid()

for i in range(200):
    
    grid.set_boundary_points()
    grid.constant_addition()
    grid.diffuse() 

grid.visualise()
