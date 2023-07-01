# DSA-Adv-Tree
QUESTION 1:-
BFS:-
from collections import defaultdict as dd  
  
class Graph:  
    
    # Constructing a list  
    def __init__(self):  
    
        # default dictionary to store graph  
        self.graph = dd(list)  
    
    # defining function which will add edge to the graph  
    def addEdgetoGraph(self, x, y):  
        self.graph[x].append(y)  
    
    # defining function to print BFS traverse  
    def BFSearch(self, n):  
    
        # Initially marking all vertices as not visited  
        visited_vertices = ( len(self.graph ))*[False]  
    
        # creating a queue for visited vertices  
        queue = []  
    
        # setting source node as visited and adding it to the queue  
        visited_vertices[n] = True  
        queue.append(n)  
          
    
        while queue:  
    
            # popping the element from the queue which is printed  
            n = queue.pop(0)  
            print (n, end = " ")  
    
            # getting vertices adjacent to the vertex n which is dequed.   
            for v in self.graph[ n ]:  
                if visited_vertices[v] == False:  
                    queue.append(v)  
                    visited_vertices[v] = True  
    
  
# Example code  
# Initializing a graph  
graph = Graph()  
graph.addEdgetoGraph(0, 1)  
graph.addEdgetoGraph(1, 1)  
graph.addEdgetoGraph(2, 2)  
graph.addEdgetoGraph(3, 1)  
graph.addEdgetoGraph(4, 3)  
graph.addEdgetoGraph(5, 4)  
    
print ( " The Breadth First Search Traversal for The Graph is as Follows: " )  
graph.BFSearch(3)  

METHOD 2:-
# BFS algorithm in Python


import collections

# BFS algorithm
def bfs(graph, root):

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        # Dequeue a vertex from queue
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)
    
    QUESTION 2:-
    METHOD 1:-
    # DFS algorithm in Python


# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')



# Python3 program to print DFS traversal
# from a given  graph
from collections import defaultdict
 
 
# This class represents a directed graph using
# adjacency list representation
class Graph:
 
    # Constructor
    def __init__(self):
 
        # Default dictionary to store graph
        self.graph = defaultdict(list)
 
     
    # Function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
 
     
    # A function used by DFS
    def DFSUtil(self, v, visited):
 
        # Mark the current node as visited
        # and print it
        visited.add(v)
        print(v, end=' ')
 
        # Recur for all the vertices
        # adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)
 
     
    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v):
 
        # Create a set to store visited vertices
        visited = set()
 
        # Call the recursive helper function
        # to print DFS traversal
        self.DFSUtil(v, visited)
 
 
# Driver's code
if __name__ == "__main__":
    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 3)
 
    print("Following is Depth First Traversal (starting from vertex 2)")
     
    # Function call
    g.DFS(2)
 
METHOD 2:-
# DFS algorithm in Python


# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')
QUESTION 3:-
# count of nodes at given level.
from collections import deque
  
adj = [[] for i in range(1001)]
  
def addEdge(v, w):
     
    # Add w to v’s list.
    adj[v].append(w)
  
    # Add v to w's list.
    adj[w].append(v)
  
def BFS(s, l):
     
    V = 100
     
    # Mark all the vertices
    # as not visited
    visited = [False] * V
    level = [0] * V
  
    for i in range(V):
        visited[i] = False
        level[i] = 0
  
    # Create a queue for BFS
    queue = deque()
  
    # Mark the current node as
    # visited and enqueue it
    visited[s] = True
    queue.append(s)
    level[s] = 0
  
    while (len(queue) > 0):
         
        # Dequeue a vertex from
        # queue and print
        s = queue.popleft()
        #queue.pop_front()
  
        # Get all adjacent vertices
        # of the dequeued vertex s.
        # If a adjacent has not been
        # visited, then mark it
        # visited and enqueue it
        for i in adj[s]:
            if (not visited[i]):
  
                # Setting the level
                # of each node with
                # an increment in the
                # level of parent node
                level[i] = level[s] + 1
                visited[i] = True
                queue.append(i)
  
    count = 0
    for i in range(V):
        if (level[i] == l):
            count += 1
             
    return count
  
# Driver code
if __name__ == '__main__':
     
    # Create a graph given
    # in the above diagram
    addEdge(0, 1)
    addEdge(0, 2)
    addEdge(1, 3)
    addEdge(2, 4)
    addEdge(2, 5)
  
    level = 2
  
    print(BFS(0, level))
     

QUESTION 4:-
# Program to count number 
# of trees in a forest. 
 
def Insert_Edge(Graph, u, v): 
    Graph[u].append(v) 
    Graph[v].append(u) 
def Depth_First_Search_Traversal(u, Graph, Check_visited): 
    Check_visited[u] = True
    for i in range(len(Graph[u])): 
        if (Check_visited[Graph[u][i]] == False): 
            Depth_First_Search_Traversal(Graph[u][i], Graph, Check_visited) 
def Count_Tree(Graph, V): 
    Check_visited = [False] * V 
    res = 0
    for u in range(V): 
        if (Check_visited[u] == False): 
            Depth_First_Search_Traversal(u, Graph, Check_visited) 
            res += 1
    return res 
# Driver code 
if __name__ == '__main__': 
    V = 7
    Graph = [[] for i in range(V)] 
    Insert_Edge(Graph, 0, 1) 
    Insert_Edge(Graph, 0, 2) 
    Insert_Edge(Graph, 3, 4)
    Insert_Edge(Graph, 4, 5) 
    Insert_Edge(Graph, 5, 6) 
    print(Count_Tree(Graph, V))
QUESTION 5:-
Python program for detecting cycles using DFS:

def has_cycle(graph):  
    visited = set()  
    stack = set()  
    def dfs(node):  
visited.add(node)  
stack.add(node)  
  
        for neighbor in graph[node]:  
            if neighbor not in visited:  
                if dfs(neighbor):  
                    return True  
elif neighbor in stack:  
                return True  
  
stack.remove(node)  
        return False  
  
    for node in graph:  
        if node not in visited:  
            if dfs(node):  
                return True  
    return False  
graph = {'A': ['B', 'C'], 'B': ['C'], 'C': ['A']}  
print(has_cycle(graph))  
graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}  
print(has_cycle(graph))    


METHOD 2:-
# collections module
from collections import defaultdict
# class for creation of graphs
class Graph():
   # constructor
   def __init__(self, vertices):
      self.graph = defaultdict(list)
      self.V = vertices
   def addEdge(self, u, v):
      self.graph[u].append(v)
   def isCyclicUtil(self, v, visited, recStack):
      # Marking current node visited and addition to recursion stack
      visited[v] = True
      recStack[v] = True
      # if any neighbour is visited and in recStack then graph is cyclic in nature
      for neighbour in self.graph[v]:
         if visited[neighbour] == False:
            if self.isCyclicUtil(neighbour, visited, recStack) == True:
               return True
         elif recStack[neighbour] == True:
            return True
      # pop the node after the end of recursion
      recStack[v] = False
      return False
   # Returns true if graph is cyclic
   def isCyclic(self):
      visited = [False] * self.V
      recStack = [False] * self.V
      for node in range(self.V):
         if visited[node] == False:
            if self.isCyclicUtil(node, visited, recStack) == True:
               return True
      return False
g = Graph(4)
g.addEdge(0, 3)
g.addEdge(0, 2)
g.addEdge(3, 2)
g.addEdge(2, 0)
g.addEdge(1, 3)
g.addEdge(2, 1)
if g.isCyclic() == 1:
   print ("Graph is cyclic in nature")
else:
   print ("Graph is non-cyclic in nature")
