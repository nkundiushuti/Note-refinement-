#!/usr/bin/env python
#This is meant to solve a maze with Dijkstra's algorithm
from numpy import inf
from copy import copy
 
class Graph(object):
    """A graph object that has a set of singly connected,weighted,
    directed edges and a set of ordered pairs. Can be changed into
    a connection matrix. Vertices are [0,1,...,n], and edges are 
    [[1,2,3],[2,1,1],...] where the first means 1 connected to 2 
    with weight 3, and the second means 2 connected to 1 with 
    weight 1."""
 
    def __init__(self,vertices,edges):
        self.vertices=vertices
        self.size=len(self.vertices)
        self.edges=edges
        self.makematrix()
 
    def makematrix(self):
        "creates connection matrix"
        self.matrix=[]
        for i in range(self.size):
            self.matrix.append([])
            for j in range(self.size):
                self.matrix[i].append(inf)
        for edge in self.edges:
            self.matrix[edge[0]][edge[1]]=edge[2]
 
    def dijkstra(self,startvertex,endvertex):
        #set distances
        self.distance=[]
        self.route=[]
        for i in range(self.size):
            self.distance.append(inf)
            self.route.append([])
        self.distance[startvertex]=0
        self.route[startvertex]=[startvertex,]
        #set visited
        self.visited=[]
        self.current=startvertex
        while self.current<>None:
            self.checkunvisited()
            if endvertex in self.visited: break
        return self.distance[endvertex],self.route[endvertex]
 
    def checkunvisited(self):
        basedist=self.distance[self.current]
        self.visited.append(self.current)
        for vertex,dist in enumerate(self.matrix[self.current]):
            if vertex in self.visited: continue #only check unvisited
            #set the distance to the new distance
            if basedist+dist<self.distance[vertex]:
                self.distance[vertex]=basedist+dist
                self.route[vertex]=copy(self.route[self.current])
                self.route[vertex].append(vertex)
        #set next current node as one with smallest distance from initial
        self.current=None
        mindist=inf
        for vertex,dist in enumerate(self.distance):
            if vertex in self.visited: continue
            if dist<mindist:
                mindist=dist
                self.current=vertex
 
 
 
def main():
    #This solves the maze in the wikipedia article on Dijkstra's algorithm
    #Note that the vertices are numbered modulo 6, so 6 is called 0 here
    V=range(6)
    E=[[1,2,7],[1,3,9],[1,0,14],[2,1,7],[2,3,10],[2,4,15],[3,1,9],[3,2,10],
    [3,4,11],[3,0,2],[4,2,15],[4,3,11],[4,5,6],[5,4,6],[5,0,9],[0,1,14],
    [0,3,2],[0,5,9]]
    m=Graph(V,E)
    print "size of graph is", m.size
 
    print "distance and best route is", m.dijkstra(1,5)
 
 
 
if __name__=="__main__": main()
 