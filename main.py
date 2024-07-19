from collections import deque
import math

class Graph:
    '''
    This is a Graph modeling a directed, weighted graph.
    The adjacencies/edges are contained in self.graph (dict), and the
    edge weights are contained in self.weights (dict).
    '''
    def __init__(self):
        self.graph = {}
        self.weights = {}
        self.lyst = []


    def add_vertex(self, vertex):
        '''
        Adds an unconnected vertex to self.graph
        '''
        if not isinstance(vertex, str):
            raise ValueError

        self.graph[vertex] = []

    def add_edge(self, src, dest, weight):
        '''
        Adds a DIRECTED edge between two existing vertices in self.graph
        '''
        if src not in self.graph or dest not in self.graph:
            raise ValueError
        if not isinstance(weight, float) and not isinstance(weight, int):
            raise ValueError

        self.weights[(src, dest)] = weight
        self.graph[src].append(dest)

    def bfs(self, start):
        '''
        Returns a generator that traverses the graph in an
        alphabetically priorities breadth-first search.
        '''
        discovered = set()
        queue = deque()

        #set start to 0
        self.weights[start] = 0

        queue.append(start) #prime the pump
        discovered.add(start)

        while len(queue) > 0:
            cur = queue.popleft()
            yield cur

            for neighbor in sorted(self.graph[cur]):
                if neighbor not in discovered:
                    queue.append(neighbor)
                    discovered.add(neighbor)


    def dfs(self, start):
        '''
        Returns a generator for traversing the graph in
        an alphabetically prioritized depth-first search
        '''
        stack = deque()
        stack.append(start)
        visited_set = set()

        while len(stack) > 0:
            cur = stack.pop()

            if cur not in visited_set:
                visited_set.add(cur)
                yield cur

                #Order is reversed to match stack behavior
                for adjacent_vertex in sorted(self.graph[cur],reverse = True):
                    stack.append(adjacent_vertex)



    def dsp_all(self, start):
        # Put all vertices in an unvisited queue.
        unvisited = []
        for vertex in self.graph:
            unvisited.append(vertex)

        # start_vertex has a distance of 0 from itself
        dist = {}
        previous = {}

        for vertex in self.graph:
            dist[vertex] = math.inf
            previous[vertex] = None
        dist[start] = 0
        # One vertex is removed with each iteration; repeat until the list is
        # empty.
        while len(unvisited) > 0:

            # Visit vertex with minimum distance from start_vertex
            smallest_index = 0
            for i in range(1, len(unvisited)):
                if dist[unvisited[i]] < dist[unvisited[smallest_index]]:
                    smallest_index = i
            cur = unvisited.pop(smallest_index)

            # Check potential path lengths from the current vertex to all neighbors.
            for adj in self.graph[cur]:
                edge_weight = self.weights[(cur, adj)]
                new_dist = dist[cur] + edge_weight

                # If shorter path from start_vertex to adj_vertex is found,
                # update adj_vertex's distance and predecessor
                if new_dist < dist[adj]:
                    previous[adj] = cur
                    dist[adj] = new_dist

        def pathfinder(vertex):
            temp = vertex
            path = [vertex]
            while previous[temp] is not None:
                temp = previous[temp]
                path.append(temp)

            return path[::-1]

        ret = {}

        for vertex in previous:
            if vertex is not None:
                ret[vertex] = pathfinder(vertex)
            else:
                ret[vertex] = []

        return ret



    def dsp(self, src, dest):
        # Put all vertices in an unvisited queue.
        unvisited = []
        for vertex in self.graph:
            unvisited.append(vertex)

        # start_vertex has a distance of 0 from itself
        dist = {}
        previous = {}

        for vertex in self.graph:
            dist[vertex] = math.inf
            previous[vertex] = None
        dist[src] = 0
        # One vertex is removed with each iteration; repeat until the list is
        # empty.
        while len(unvisited) > 0:

            # Visit vertex with minimum distance from start_vertex
            smallest_index = 0
            for i in range(1, len(unvisited)):
                if dist[unvisited[i]] < dist[unvisited[smallest_index]]:
                    smallest_index = i
            cur = unvisited.pop(smallest_index)

            # Check potential path lengths from the current vertex to all neighbors.
            for adj in self.graph[cur]:
                edge_weight = self.weights[(cur, adj)]
                new_dist = dist[cur] + edge_weight

                # If shorter path from start_vertex to adj_vertex is found,
                # update adj_vertex's distance and predecessor
                if new_dist < dist[adj]:
                    previous[adj] = cur
                    dist[adj] = new_dist

        def pathfinder(vertex):
            temp = vertex
            path = [vertex]
            while previous[temp] is not None:
                temp = previous[temp]
                path.append(temp)

            return path[::-1]

        ret = {}

        for vertex in previous:
            if vertex is not None:
                ret[vertex] = pathfinder(vertex)
            else:
                ret[vertex] = []
        if previous[dest] is not None:
            return int(dist[dest]), ret[dest]
        else:
            return math.inf, []

    def __str__(self):
        ret = f"digraph G"
        ret += ' {'
        for src, dest in self.weights:
            ret += f'\n   {src} -> {dest} [label="{self.weights[src, dest]}",weight="{self.weights[src, dest]}"];'
        ret += '\n}'
        return ret

def main():
    '''
    This is the program main function
    '''
    G = Graph()
    G.add_vertex('A')
    G.add_vertex('B')
    G.add_vertex('C')
    G.add_vertex('D')
    G.add_vertex('E')
    G.add_vertex('F')
    G.add_vertex('G')
    G.add_edge('A','B',2.0)
    G.add_edge('A','F',9.0)
    G.add_edge('B','F',6.0)
    G.add_edge('F','B',6.0)
    G.add_edge('B','D',15.0)
    G.add_edge('B','C',8.0)
    G.add_edge('E','C',7.0)
    G.add_edge('E','D',3.0)
    G.add_edge('F','E',3.0)
    G.add_edge('C','D',1.0)


    printmsg = '''digraph G {
   A -> B [label="2.0",weight="2.0"];
   A -> F [label="9.0",weight="9.0"];
   B -> C [label="8.0",weight="8.0"];
   B -> D [label="15.0",weight="15.0"];
   B -> F [label="6.0",weight="6.0"];
   C -> D [label="1.0",weight="1.0"];
   E -> C [label="7.0",weight="7.0"];
   E -> D [label="3.0",weight="3.0"];
   F -> B [label="6.0",weight="6.0"];
   F -> E [label="3.0",weight="3.0"];
}
'''
    print(printmsg)
    print("starting DFS with vertex A")
    for vertex in G.dfs("A"):
        print(vertex, end = "")
    print()
    print("starting BFS with vertex A")
    for vertex in G.bfs("A"):
        print(vertex, end = "")
    print()
    print(G.dsp('A','F'))
    my_dict = G.dsp_all('A')
    for vertex in my_dict:
        print("{",end='')
        print(f"{vertex}: {my_dict[vertex]}",end='')
        print("}")
    
if __name__ == "__main__":
    main()
