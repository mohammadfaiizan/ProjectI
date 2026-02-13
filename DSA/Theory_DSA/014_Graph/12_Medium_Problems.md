# Graph - Medium Problems

## 1. Course Schedule

**Description**: numCourses and prerequisites. Can you finish all courses?

**Approach**: Build directed graph, detect cycle. Kahn's algorithm or DFS 3-color. If topological order exists, can finish.

---

## 2. Course Schedule II

**Description**: Return valid order to take all courses, or empty if impossible.

**Approach**: Kahn's algorithm. Return topological order.

---

## 3. Number of Connected Components

**Description**: Given n and edges, count connected components.

**Approach**: Union-Find or DFS/BFS. Count number of times we start from unvisited.

---

## 4. Redundant Connection

**Description**: Tree plus one extra edge. Find edge that creates cycle.

**Approach**: Union-Find. First edge that connects already-connected vertices is answer.

---

## 5. Redundant Connection II

**Description**: Rooted tree plus one edge. Find edge to remove to get valid rooted tree.

**Approach**: Two cases: node with two parents, or cycle. Union-Find with parent tracking.

---

## 6. Accounts Merge

**Description**: Merge accounts that share email. Return merged account lists.

**Approach**: Union-Find on emails. Group by root, sort emails per group.

---

## 7. Evaluate Division

**Description**: Equations a/b = value. Answer queries c/d.

**Approach**: Build weighted graph. DFS/BFS to find path and multiply weights.

---

## 8. Word Ladder

**Description**: Transform beginWord to endWord changing one letter at a time. Words from list.

**Approach**: BFS. Each state is a word. Neighbors are words differing by one letter.

---

## 9. Word Ladder II

**Description**: Find all shortest transformation sequences from begin to end.

**Approach**: BFS to find shortest distance. DFS to reconstruct all paths of that length.

---

## 10. Surrounded Regions

**Description**: Flip 'O' to 'X' if not connected to border.

**Approach**: DFS from border 'O's, mark as temporary. Flip remaining 'O' to 'X'.

---

## 11. Clone Graph

**Description**: Deep copy graph with same structure.

**Approach**: DFS with node-to-copy mapping.

---

## 12. Pacific Atlantic Water Flow

**Description**: Grid heights. Which cells can flow to both oceans (top/left and bottom/right)?

**Approach**: DFS from Pacific border, DFS from Atlantic border. Intersection of reachable sets.

---

## 13. Number of Islands

**Description**: Count connected '1' regions in grid.

**Approach**: DFS/BFS for each unvisited '1'.

---

## 14. Max Area of Island

**Description**: Find largest connected component of 1s.

**Approach**: DFS return area, track max.

---

## 15. Rotting Oranges

**Description**: Grid with fresh (1) and rotten (2) oranges. Minutes until all rotten?

**Approach**: Multi-source BFS from all rotten. Expand each minute.

---

## 16. 01 Matrix

**Description**: For each cell, distance to nearest 0.

**Approach**: Multi-source BFS from all 0s. Propagate distance.

---

## 17. Shortest Path in Binary Matrix

**Description**: 0-1 grid, shortest path from (0,0) to (n-1,n-1) using 8 directions, only 0 cells.

**Approach**: BFS with 8-directional moves.

---

## 18. Network Delay Time

**Description**: Times (u, v, w). Signal from k. Time for all nodes to receive?

**Approach**: Dijkstra from k. Return max distance or -1 if unreachable.

---

## 19. Cheapest Flights Within K Stops

**Description**: Find cheapest path from src to dst with at most k stops.

**Approach**: BFS/Dijkstra variant. Track (node, cost, stops). Relax with stop limit.

---

## 20. Path With Maximum Probability

**Description**: Undirected graph, edge success probabilities. Max probability path from start to end.

**Approach**: Dijkstra with max-heap (or negate for min-heap). Multiply probabilities.

---

## 21. Reorder Routes to Make All Paths Lead to City Zero

**Description**: Directed edges. Some point to 0, some away. Min edges to flip so all lead to 0?

**Approach**: BFS from 0. Count edges that point toward 0 (direction 1 in connections).

---

## 22. Find Eventual Safe States

**Description**: Directed graph. Node is safe if all paths lead to terminal (no outgoing). Find all safe nodes.

**Approach**: Reverse graph, start from terminals. Or DFS cycle detection; nodes not in cycle are safe.

---

## 23. Making a Large Island

**Description**: Grid of 0s and 1s. Can change one 0 to 1. Max island size?

**Approach**: DFS to label islands and get sizes. For each 0, sum sizes of adjacent distinct islands + 1.

---

## 24. Shortest Bridge

**Description**: Two islands of 1s in sea of 0s. Min 0s to flip to connect islands?

**Approach**: DFS to find first island. Multi-source BFS from first island until hitting second.

---

## 25. Number of Enclaves

**Description**: Count 1s that cannot reach border.

**Approach**: DFS from border 1s to mark reachable. Count remaining 1s.

---

## 26. Count Sub-Islands

**Description**: grid1 and grid2. Count islands in grid2 that are fully covered by grid1.

**Approach**: For each grid2 island, DFS and check all cells are 1 in grid1.

---

## 27. Minimum Genetic Mutation

**Description**: Transform start gene to end via valid mutations (one char change, must be in bank).

**Approach**: BFS like word ladder. States are genes.

---

## 28. Open the Lock

**Description**: 4-digit lock. Deadends and target. Min moves to reach target?

**Approach**: BFS. States are 4-digit strings. Neighbors: increment/decrement each digit.

---

## 29. Satisfiability of Equality Equations

**Description**: Equations like a==b or a!=b. Are all satisfiable?

**Approach**: Union-Find for ==. Check != pairs are in different sets.

---

## 30. Smallest String With Swaps

**Description**: String and pairs of indices. Can swap any pair unlimited times. Lexicographically smallest?

**Approach**: Union-Find to group swappable indices. Sort chars in each group, place at sorted indices.

---

# Hard Problems

## 1. Critical Connections in a Network

**Description**: Find all bridges (edges whose removal increases connected components).

**Approach**: Tarjan's bridge-finding algorithm. Low-link values.

---

## 2. Word Ladder II

**Description**: All shortest transformation sequences from begin to end word.

**Approach**: BFS to get distances. DFS to build paths. Or BFS with path tracking.

---

## 3. Minimum Cost to Connect All Points

**Description**: Connect all points with minimum total Manhattan distance.

**Approach**: Kruskal's MST. Edges are all pairs with Manhattan weight.

---

## 4. Swim in Rising Water

**Description**: Grid with heights. Water rises. Earliest time to swim from (0,0) to (n-1,n-1)?

**Approach**: Dijkstra. Edge weight is max of current time and cell height.

---

## 5. Path With Minimum Effort

**Description**: Grid heights. Path effort = max absolute difference along path. Min effort from top-left to bottom-right?

**Approach**: Dijkstra with effort as cost. Relax: new_effort = max(current, abs(diff)).

---

## 6. Shortest Path Visiting All Nodes

**Description**: Undirected graph. Shortest path that visits every node at least once.

**Approach**: BFS with state (node, bitmask of visited). Target: all bits set.

---

## 7. Number of Restricted Paths

**Description**: Weighted graph. Restricted path: each step must go to node with strictly greater distance to n. Count restricted paths from 1 to n.

**Approach**: Dijkstra from n to get distances. DP: paths[v] = sum paths[u] for u with dist[u] > dist[v].

---

## 8. Parallel Courses III

**Description**: n courses, relations, time per course. Min time to finish all (parallel allowed with dependencies).

**Approach**: Topological sort. dist[u] = time[u] + max(dist[v]) for prerequisites v.

---

## 9. Sequence Reconstruction

**Description**: Check if nums is the unique shortest supersequence of all sequences.

**Approach**: Build graph from sequences. Topological sort. Check unique order and matches nums.

---

## 10. Alien Dictionary

**Description**: Sorted dictionary of alien language. Derive character order.

**Approach**: Build graph from adjacent word comparisons. Topological sort.

---

## 11. Minimum Height Trees

**Description**: Tree of n nodes. Which nodes as root give minimum height?

**Approach**: Repeatedly remove leaves. Last 1 or 2 nodes are centers.

---

## 12. Longest Increasing Path in a Matrix

**Description**: Grid. Longest strictly increasing path (any direction).

**Approach**: DFS with memoization. Path from (r,c) = 1 + max(neighbors with smaller value).

---

## 13. Count Subtrees With Max Distance Between Cities

**Description**: Tree of n nodes. For each d from 1 to n-1, count subtrees with diameter d.

**Approach**: For each node as root, DFS to compute subtree diameters. Complex tree DP.

---

## 14. Graph Connectivity With Threshold

**Description**: n nodes. Nodes i and j connected if gcd(i,j) > threshold. Queries: are a and b connected?

**Approach**: Union-Find. For each pair (i,j) with gcd > threshold, union. Answer queries with find.

---

## 15. Minimum Cost to Reach Destination in Time

**Description**: Graph with edge times and node fees. Max time limit. Min cost to reach n-1 within time?

**Approach**: Dijkstra-like. State (node, time). Minimize cost. Relax with time constraint.
