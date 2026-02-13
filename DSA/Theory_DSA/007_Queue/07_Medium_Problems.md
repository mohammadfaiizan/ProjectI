# Medium Queue Problems

## 1. Binary Tree Zigzag Level Order Traversal

**Description**: Return level-order traversal but alternate left-to-right and right-to-left per level.

**Approach**: BFS with level tracking. Use a flag; when flag is false, reverse the level before appending to result.

---

## 2. Binary Tree Right Side View

**Description**: Return the rightmost node value at each level.

**Approach**: BFS. For each level, take the last node processed (rightmost).

---

## 3. Binary Tree Left Side View

**Description**: Return the leftmost node value at each level.

**Approach**: BFS. For each level, take the first node (leftmost).

---

## 4. Populating Next Right Pointers in Each Node

**Description**: Perfect binary tree. Populate each node's next pointer to point to its next right node on the same level.

**Approach**: BFS level-order. For each level, link nodes from left to right. Last node in level has next = null.

---

## 5. Populating Next Right Pointers in Each Node II

**Description**: Same as above but tree may not be perfect (missing nodes).

**Approach**: BFS. Track previous node in level; set previous.next = current. Handle levels with gaps.

---

## 6. Number of Islands

**Description**: Grid of '1' and '0'. Count connected components of '1'.

**Approach**: BFS (or DFS) from each unvisited '1'. Mark all reachable '1's as visited. Count number of BFS starts.

---

## 7. Rotting Oranges

**Description**: Grid with fresh (1) and rotten (2) oranges. Each minute, rotten rot adjacent fresh. Return minutes to rot all, or -1.

**Approach**: Multi-source BFS from all rotten. Expand one minute at a time. Track fresh count; return -1 if any remain.

---

## 8. Shortest Path in Binary Matrix

**Description**: NxN grid of 0s and 1s. Shortest path from (0,0) to (n-1,n-1) through 0s only. 8-direction.

**Approach**: BFS from (0,0). Enqueue unvisited 0 neighbors. Return depth when (n-1,n-1) is reached.

---

## 9. Word Ladder

**Description**: Transform beginWord to endWord by changing one letter at a time. Each intermediate must be in wordList. Return shortest sequence length.

**Approach**: BFS. For each word, try all one-letter changes. Use set for O(1) lookup. Return depth when endWord found.

---

## 10. Word Ladder II

**Description**: Same as Word Ladder but return all shortest transformation sequences.

**Approach**: BFS in layers. Build paths. When a word is found at a new layer, add all paths to it. Use layer-by-layer to avoid longer paths.

---

## 11. Open the Lock

**Description**: 4-digit lock. Start "0000", target given. Deadends forbidden. Each move rotates one wheel by 1. Return minimum moves.

**Approach**: BFS with state as string. For each digit, try +1 and -1 (mod 10). Skip deadends and visited.

---

## 12. Snakes and Ladders

**Description**: Board game. Boustrophedon numbering. Some cells have snakes/ladders. Minimum dice rolls from 1 to N*N.

**Approach**: BFS. State = board position. From each position, try all 6 dice outcomes. Apply snake/ladder if destination has one.

---

## 13. 01 Matrix (Multidirectional BFS)

**Description**: Matrix of 0s and 1s. For each cell, return distance to nearest 0.

**Approach**: Multi-source BFS from all 0s. Propagate distance to neighbors.

---

## 14. As Far from Land as Possible

**Description**: NxN grid of land (1) and water (0). Find water cell with maximum distance to nearest land.

**Approach**: Multi-source BFS from all land. Last water cell to be reached has max distance.

---

## 15. Sliding Window Maximum

**Description**: For each window of size k, return the maximum element.

**Approach**: Monotonic decreasing deque. Store indices. Remove from back when new element is larger. Front is current max.

---

## 16. Design Task Scheduler

**Description**: Schedule tasks with cooldown n between same tasks. Minimize total time.

**Approach**: Max-heap for task counts. Queue for (task, next_available_time). Each slot: if heap has task, pop and schedule; push to queue.

---

## 17. LRU Cache

**Description**: Design LRU cache with get and put. Evict least recently used when full.

**Approach**: OrderedDict (move_to_end on access) or HashMap + deque of keys. On get: move to end. On put: add/update, evict from front if full.

---

## 18. First Unique Character in Data Stream

**Description**: Stream of integers. Support add() and showFirstUnique(). Return first unique in stream.

**Approach**: Queue for candidates. HashMap for frequency. On add: increment freq, enqueue. Remove from front while freq[front] > 1.

---

## 19. Interleave First and Second Half of Queue

**Description**: Queue [1,2,3,4,5,6] becomes [1,4,2,5,3,6]. Use only queue and stack.

**Approach**: Push first half to stack. Enqueue stack to queue. Rotate. Push new first half to stack. Interleave: pop from stack, dequeue from queue, alternate.

---

## 20. Sort a Queue

**Description**: Sort a queue using only queue operations (and possibly one extra queue).

**Approach**: Repeatedly find minimum by rotating through queue, remove it, append to result queue. Repeat until empty.

---

## 21. Jump Game III (BFS)

**Description**: From start index, can jump to start+arr[start] or start-arr[start]. Return true if can reach any index with value 0.

**Approach**: BFS with visited set. Enqueue valid indices. Return true when we land on 0.

---

## 22. Minimum Knight Moves

**Description**: Minimum moves for knight from (0,0) to (x,y) on infinite board.

**Approach**: BFS. Knight moves: 8 L-shaped positions. Use symmetry to limit search to one quadrant.

---

## 23. Shortest Path with Alternating Colors

**Description**: Directed graph with red and blue edges. Shortest path from 0 to n-1 with alternating edge colors.

**Approach**: BFS with state (node, last_color). From red edge we take blue; from blue we take red. Track (node, color) in visited.

---

## 24. Nearest Exit from Entrance in Maze

**Description**: Maze with '.' and '+'. Shortest path from entrance to any border (exit).

**Approach**: BFS from entrance. Return depth when we reach a border cell (except entrance).

---

## 25. Map of Highest Peak

**Description**: Grid of land and water. Assign heights so adjacent differ by at most 1, water is 0. Maximize heights.

**Approach**: Multi-source BFS from water cells. Assign distance to each land cell.

---

## 26. Bus Routes

**Description**: Array of bus routes (each is list of stops). Minimum buses from source to target.

**Approach**: BFS. Build stop_to_buses map. From each stop, try all buses. Track buses taken. Each bus ride = 1.

---

## 27. Sliding Puzzle

**Description**: 2x3 board. Minimum moves to reach [[1,2,3],[4,5,0]].

**Approach**: BFS over board states (as string). Swap 0 with neighbors. Return depth when target reached.

---

## 28. Shortest Subarray with Sum at Least K

**Description**: Array (may have negatives). Shortest subarray with sum >= k.

**Approach**: Prefix sum + monotonic increasing deque. For each prefix[i], find smallest j with prefix[j] <= prefix[i] - k.

---

## 29. Constrained Subsequence Sum

**Description**: Subsequence with no two elements within k indices. Maximize sum.

**Approach**: DP with dp[i] = nums[i] + max(dp[i-k]..dp[i-1]). Monotonic deque for max over sliding window.

---

## 30. Longest Continuous Subarray with Absolute Diff <= Limit

**Description**: Longest subarray where max - min <= limit.

**Approach**: Two monotonic deques (max and min). Expand right; when max-min > limit, shrink left.

---

# Hard Problems

## 1. Word Ladder II (Optimized)

**Description**: All shortest transformation sequences from beginWord to endWord. Avoid TLE.

**Approach**: Bidirectional BFS or single BFS with layer tracking. Build graph of parent pointers; DFS to reconstruct paths.

---

## 2. Sliding Window Maximum (Deque)

**Description**: Same as medium; ensure O(n) solution.

**Approach**: Monotonic decreasing deque. Each element pushed and popped at most once.

---

## 3. Shortest Subarray with Sum at Least K

**Description**: Array with negatives. Shortest subarray with sum >= k. O(n) required.

**Approach**: Prefix sum + monotonic deque. Maintain increasing prefix indices; for each i, pop from front while valid, pop from back while prefix[back] >= prefix[i].

---

## 4. Constrained Subsequence Sum

**Description**: Subsequence with no two elements within k indices. Maximize sum. O(n) required.

**Approach**: DP + monotonic deque for max over sliding window of dp values.

---

## 5. Jump Game VI

**Description**: From index 0, jump at most k steps. Maximize sum of landed indices.

**Approach**: DP with dp[i] = nums[i] + max(dp[i-k]..dp[i-1]). Monotonic deque for max.

---

## 6. Longest Continuous Subarray with Absolute Diff <= Limit

**Description**: Longest subarray where max - min <= limit. O(n).

**Approach**: Two deques (max, min). Sliding window. Shrink left when max - min > limit.

---

## 7. Max Value of Equation

**Description**: Points (x_i, y_i) sorted by x. Max of y_i + y_j + |x_i - x_j| for |x_i - x_j| <= k.

**Approach**: Rewrite as (y_i - x_i) + (y_j + x_j). For each j, max over i in range of (y_i - x_i). Monotonic deque.

---

## 8. Minimum Cost to Make at Least One Valid Path in a Grid

**Description**: Grid with arrows. Change cost of 1 per cell. Minimum cost to reach bottom-right.

**Approach**: 0-1 BFS. Moving in arrow direction costs 0; changing direction costs 1. Deque: push to front for 0, back for 1.

---

## 9. Shortest Path in a Grid with Obstacles Elimination

**Description**: Grid with obstacles. Can eliminate at most k obstacles. Shortest path from (0,0) to (m-1,n-1).

**Approach**: BFS with state (r, c, k_remaining). When hitting obstacle, decrement k if k > 0. Track visited as (r, c, k).

---

## 10. Bus Routes (Hard variant)

**Description**: Large number of routes and stops. Optimize for memory and time.

**Approach**: BFS on buses. Build stop_to_buses. Only expand to unvisited buses. Track visited stops per bus.

---

## 11. Sliding Puzzle (3x2 or 3x3)

**Description**: 3x2 or 3x3 sliding puzzle. Minimum moves.

**Approach**: BFS over state space. State = flattened board string. Swap blank with neighbors. Use A* for 3x3 to reduce states.

---

## 12. Open the Lock (Double-ended)

**Description**: Open the lock with minimum moves. May have multiple targets or constraints.

**Approach**: BFS. Bidirectional BFS can reduce search space for large state spaces.

---

## 13. Word Ladder (Optimized)

**Description**: Word Ladder with very long wordList. Avoid TLE.

**Approach**: Bidirectional BFS. Or use character-level wildcard indexing: for "hot" try "*ot", "h*t", "ho*" to find neighbors in O(word_length) per word.

---

## 14. Minimum Knight Moves (Optimized)

**Description**: Minimum knight moves with large coordinates. Avoid MLE.

**Approach**: BFS with symmetry. Only search in first quadrant. Use (abs(x), abs(y)) and bound search space.

---

## 15. Critical Connections in a Network (BFS variant)

**Description**: Find bridges in graph. BFS can be used in some approaches.

**Approach**: Tarjan's DFS is standard. BFS-based level graph for max-flow can identify critical edges.

---

## 16. Reconstruct Itinerary (BFS/Queue)

**Description**: Lexicographically smallest Euler path. Uses queue for managing adjacency.

**Approach**: Hierholzer's algorithm. Use heap or sorted structure for neighbors. DFS with backtracking; queue for remaining edges.

---

## 17. Alien Dictionary (Topological + BFS)

**Description**: Given sorted dictionary of alien language, derive character order. BFS for Kahn's algorithm.

**Approach**: Build graph from adjacent word pairs. Kahn's algorithm (BFS) for topological sort. Queue for indegree-zero nodes.

---

## 18. Sequence Reconstruction (BFS)

**Description**: Check if sequence is the unique shortest supersequence of sequences. BFS for topological order.

**Approach**: Build graph. Check if given sequence is the unique topological order. BFS to verify.

---

## 19. Course Schedule II (BFS)

**Description**: Prerequisites for courses. Return valid order to take all courses.

**Approach**: Kahn's algorithm. Build graph and indegree. BFS with queue of indegree-zero nodes.

---

## 20. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix

**Description**: Flip cells (and neighbors) to convert all to 0. Minimum flips.

**Approach**: BFS over state space. State = flattened matrix. Each move flips a cell. Exponential size; use bitmask for small grids.
