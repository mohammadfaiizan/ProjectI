# Graph - Easy Problems

## 1. Find the Town Judge

**Description**: In a town of n people, trust array gives who trusts whom. The town judge trusts nobody, everyone trusts the judge. Find the judge if exists.

**Approach**: Count in-degree (trusted by) and out-degree (trusts) for each person. Judge has in-degree n-1 and out-degree 0.

---

## 2. Find Center of Star Graph

**Description**: Given edges of a star graph (one center connected to all others), find the center.

**Approach**: The center appears in every edge. Check first two edges; common vertex is center.

---

## 3. Number of Provinces

**Description**: n cities, is_connected matrix. Find number of connected components.

**Approach**: DFS/BFS from each unvisited city. Or Union-Find.

---

## 4. Find if Path Exists in Graph

**Description**: Undirected graph, check if path exists from source to destination.

**Approach**: BFS or DFS from source, check if destination is reached.

---

## 5. Keys and Rooms

**Description**: rooms[i] has keys to other rooms. Can you visit all rooms starting from 0?

**Approach**: DFS/BFS from room 0. Check if visited size equals number of rooms.

---

## 6. Flood Fill

**Description**: Given image, start pixel, new color. Flood fill connected same-color pixels.

**Approach**: DFS or BFS from start, replace color for all connected same-color cells.

---

## 7. Number of Islands

**Description**: 2D grid of '1' and '0'. Count number of connected '1' regions.

**Approach**: DFS/BFS for each unvisited '1', mark entire island, increment count.

---

## 8. Max Area of Island

**Description**: Find maximum area of connected 1s in grid.

**Approach**: DFS for each island, return area (count of cells). Track global max.

---

## 9. Is Graph Bipartite

**Description**: Can vertices be colored with 2 colors such that no adjacent vertices share color?

**Approach**: BFS/DFS 2-coloring. If conflict, not bipartite.

---

## 10. Possible Bipartition

**Description**: n people, dislikes pairs. Can we split into two groups with no dislikes within group?

**Approach**: Build graph from dislikes. Check bipartite with BFS/DFS.

---

## 11. Employee Importance

**Description**: Employee tree structure. Get total importance of employee and all subordinates.

**Approach**: Build adjacency from id to employee. DFS/BFS from given id, sum importance.

---

## 12. Clone Graph

**Description**: Deep copy a graph with same structure and values.

**Approach**: DFS with mapping dict. For each node, create copy, recurse on neighbors.

---

## 13. All Paths From Source to Target

**Description**: DAG, find all paths from node 0 to node n-1.

**Approach**: DFS backtracking. Add current to path, recurse on neighbors, backtrack.

---

## 14. Valid Path

**Description**: n vertices, edges, check if path exists from source to destination.

**Approach**: Build adjacency list, BFS/DFS from source.

---

## 15. Minimum Depth of Binary Tree (Graph View)

**Description**: In tree/graph, find minimum distance from root to any leaf.

**Approach**: BFS level by level, return depth when first leaf found.

---

## 16. Same Tree

**Description**: Check if two trees are identical.

**Approach**: Recursive compare structure and values.

---

## 17. Symmetric Tree

**Description**: Check if tree is mirror of itself.

**Approach**: Helper(left, right): both null true; recurse (left.left, right.right) and (left.right, right.left).

---

## 18. Invert Binary Tree

**Description**: Swap left and right children for every node.

**Approach**: Recursive swap, then recurse on both children.

---

## 19. Merge Two Binary Trees

**Description**: Overlay two trees, sum values at overlapping nodes.

**Approach**: Recursive merge. If one null return other. Create new node with sum, merge left and right.

---

## 20. Average of Levels in Binary Tree

**Description**: Return average value at each level.

**Approach**: BFS level order, compute average per level.

---

## 21. Second Minimum Node in Binary Tree

**Description**: Find second smallest value in tree (each node has 0, 1, or 2 children).

**Approach**: DFS collect all values, find second min. Or track first and second during traversal.

---

## 22. N-ary Tree Preorder Traversal

**Description**: Preorder traversal of n-ary tree.

**Approach**: Visit root, recurse on children in order.

---

## 23. N-ary Tree Postorder Traversal

**Description**: Postorder traversal of n-ary tree.

**Approach**: Recurse on all children, then visit root.

---

## 24. N-ary Tree Level Order Traversal

**Description**: Level order traversal of n-ary tree.

**Approach**: BFS with queue.

---

## 25. Maximum Depth of N-ary Tree

**Description**: Return maximum depth of n-ary tree.

**Approach**: 1 + max(child depths) for each node, or BFS count levels.
