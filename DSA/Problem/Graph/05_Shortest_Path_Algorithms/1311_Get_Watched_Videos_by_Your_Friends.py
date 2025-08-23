"""
1311. Get Watched Videos by Your Friends
Difficulty: Medium

Problem:
There are n people, each person has a unique id between 0 and n-1. Given the arrays 
watchedVideos and friends, where watchedVideos[i] and friends[i] contain the list of 
watched videos and the list of friends respectively for the person with id = i.

Level 1 of videos are all watched videos by your friends, level 2 of videos are all 
watched videos by the friends of your friends and so on. In general, the level k of 
videos are all watched videos by people who are exactly k edges away from you in the 
friends graph.

Given your id and the level of videos to be retrieved, return the list of videos 
ordered by their frequencies (increasing order). For videos with the same frequency, 
order them alphabetically (increasing order).

Examples:
Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,1],[1]], id = 0, level = 1
Output: ["B","C"]

Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,1],[1]], id = 0, level = 2
Output: ["D"]

Constraints:
- n == watchedVideos.length == friends.length
- 2 <= n <= 100
- 1 <= watchedVideos[i].length <= 100
- 1 <= watchedVideos[i][j].length <= 8
- 0 <= friends[i].length <= n - 1
- 0 <= friends[i][j] <= n - 1
- 0 <= id < n
- 1 <= level <= n
- if friends[i] contains j, then friends[j] contains i
"""

from typing import List
from collections import deque, defaultdict, Counter

class Solution:
    def watchedVideosByFriends_approach1_bfs_level_tracking(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        """
        Approach 1: BFS with Level Tracking (Optimal)
        
        Use BFS to find friends at exactly the specified level, then collect videos.
        
        Time: O(N + E + V log V) where N=people, E=friendships, V=videos
        Space: O(N + V)
        """
        # BFS to find friends at specified level
        visited = {id}
        queue = deque([id])
        
        # Perform BFS for 'level' steps
        for _ in range(level):
            if not queue:
                return []  # No friends at this level
            
            next_queue = deque()
            
            while queue:
                person = queue.popleft()
                
                for friend in friends[person]:
                    if friend not in visited:
                        visited.add(friend)
                        next_queue.append(friend)
            
            queue = next_queue
        
        # Collect videos from friends at target level
        video_count = Counter()
        
        for person in queue:
            for video in watchedVideos[person]:
                video_count[video] += 1
        
        # Sort by frequency, then alphabetically
        return sorted(video_count.keys(), key=lambda x: (video_count[x], x))
    
    def watchedVideosByFriends_approach2_bfs_distance_array(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        """
        Approach 2: BFS with Distance Array
        
        Use BFS with distance tracking to find all people at each level.
        
        Time: O(N + E + V log V)
        Space: O(N + V)
        """
        n = len(friends)
        distances = [-1] * n
        distances[id] = 0
        
        queue = deque([id])
        
        while queue:
            person = queue.popleft()
            
            if distances[person] == level:
                continue  # Don't explore beyond target level
            
            for friend in friends[person]:
                if distances[friend] == -1:  # Not visited
                    distances[friend] = distances[person] + 1
                    queue.append(friend)
        
        # Collect videos from people at target level
        video_count = Counter()
        
        for person in range(n):
            if distances[person] == level:
                for video in watchedVideos[person]:
                    video_count[video] += 1
        
        # Sort by frequency, then alphabetically
        return sorted(video_count.keys(), key=lambda x: (video_count[x], x))
    
    def watchedVideosByFriends_approach3_dfs_recursive(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        """
        Approach 3: DFS Recursive Approach
        
        Use DFS to explore friends at specific levels.
        
        Time: O(N + E + V log V)
        Space: O(N + V)
        """
        target_friends = set()
        visited = set()
        
        def dfs(person, current_level):
            """DFS to find friends at target level"""
            if current_level == level:
                target_friends.add(person)
                return
            
            if current_level > level:
                return
            
            visited.add(person)
            
            for friend in friends[person]:
                if friend not in visited:
                    dfs(friend, current_level + 1)
            
            visited.remove(person)  # Backtrack for multiple paths
        
        # Find all friends at target level
        visited.add(id)
        for friend in friends[id]:
            if friend not in visited:
                dfs(friend, 1)
        
        # Collect videos from target friends
        video_count = Counter()
        
        for person in target_friends:
            for video in watchedVideos[person]:
                video_count[video] += 1
        
        return sorted(video_count.keys(), key=lambda x: (video_count[x], x))
    
    def watchedVideosByFriends_approach4_level_sets(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        """
        Approach 4: Level Sets Tracking
        
        Track friends at each level using sets.
        
        Time: O(N + E + V log V)
        Space: O(N + V)
        """
        current_level = {id}
        visited = {id}
        
        # Build level by level
        for _ in range(level):
            next_level = set()
            
            for person in current_level:
                for friend in friends[person]:
                    if friend not in visited:
                        visited.add(friend)
                        next_level.add(friend)
            
            current_level = next_level
            
            if not current_level:
                return []  # No friends at this level
        
        # Collect videos from current level
        video_count = Counter()
        
        for person in current_level:
            for video in watchedVideos[person]:
                video_count[video] += 1
        
        return sorted(video_count.keys(), key=lambda x: (video_count[x], x))
    
    def watchedVideosByFriends_approach5_dijkstra_variant(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        """
        Approach 5: Dijkstra-like Approach (Overkill but Educational)
        
        Use priority queue to find shortest distances (all edges weight 1).
        
        Time: O(N log N + E + V log V)
        Space: O(N + V)
        """
        import heapq
        
        distances = [float('inf')] * len(friends)
        distances[id] = 0
        
        pq = [(0, id)]
        
        while pq:
            dist, person = heapq.heappop(pq)
            
            if dist > distances[person]:
                continue
            
            if dist >= level:  # Don't explore beyond target level
                continue
            
            for friend in friends[person]:
                new_dist = dist + 1
                
                if new_dist < distances[friend]:
                    distances[friend] = new_dist
                    heapq.heappush(pq, (new_dist, friend))
        
        # Collect videos from people at exactly target level
        video_count = Counter()
        
        for person in range(len(friends)):
            if distances[person] == level:
                for video in watchedVideos[person]:
                    video_count[video] += 1
        
        return sorted(video_count.keys(), key=lambda x: (video_count[x], x))

def test_watched_videos_by_friends():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (watchedVideos, friends, id, level, expected)
        ([["A","B"],["C"],["B","C"],["D"]], [[1,2],[0,3],[0,1],[1]], 0, 1, ["B","C"]),
        ([["A","B"],["C"],["B","C"],["D"]], [[1,2],[0,3],[0,1],[1]], 0, 2, ["D"]),
        ([["A","B"],["C"],["B","C"],["D"]], [[1,2],[0,3],[0,1],[1]], 1, 1, ["A","B","C","D"]),
        ([["A"],["B"],["C"]], [[1],[0,2],[1]], 0, 2, ["C"]),
        ([["A"],["B"]], [[1],[0]], 0, 1, ["B"]),
    ]
    
    approaches = [
        ("BFS Level Tracking", solution.watchedVideosByFriends_approach1_bfs_level_tracking),
        ("BFS Distance Array", solution.watchedVideosByFriends_approach2_bfs_distance_array),
        ("DFS Recursive", solution.watchedVideosByFriends_approach3_dfs_recursive),
        ("Level Sets", solution.watchedVideosByFriends_approach4_level_sets),
        ("Dijkstra Variant", solution.watchedVideosByFriends_approach5_dijkstra_variant),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (videos, friends, id, level, expected) in enumerate(test_cases):
            result = func([v[:] for v in videos], [f[:] for f in friends], id, level)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} id={id}, level={level}, expected={expected}, got={result}")

def demonstrate_friend_levels():
    """Demonstrate friend level concept with BFS"""
    print("\n=== Friend Levels Demo ===")
    
    watchedVideos = [["A","B"],["C"],["B","C"],["D"]]
    friends = [[1,2],[0,3],[0,1],[1]]
    id = 0
    
    print(f"Friend network:")
    for i, friend_list in enumerate(friends):
        print(f"  Person {i}: friends with {friend_list}")
    
    print(f"\nWatched videos:")
    for i, videos in enumerate(watchedVideos):
        print(f"  Person {i}: {videos}")
    
    print(f"\nStarting from person {id}, finding friends at each level:")
    
    # BFS level by level
    visited = {id}
    current_level = {id}
    level_num = 0
    
    print(f"Level {level_num}: {current_level} (yourself)")
    
    while current_level:
        level_num += 1
        next_level = set()
        
        for person in current_level:
            for friend in friends[person]:
                if friend not in visited:
                    visited.add(friend)
                    next_level.add(friend)
        
        if next_level:
            print(f"Level {level_num}: {next_level}")
            
            # Show videos for this level
            videos_at_level = []
            for person in next_level:
                videos_at_level.extend(watchedVideos[person])
            
            video_count = Counter(videos_at_level)
            print(f"  Videos: {dict(video_count)}")
            
            # Sort by frequency, then alphabetically
            sorted_videos = sorted(video_count.keys(), key=lambda x: (video_count[x], x))
            print(f"  Sorted: {sorted_videos}")
        
        current_level = next_level

def analyze_social_network_patterns():
    """Analyze social network graph patterns"""
    print("\n=== Social Network Analysis ===")
    
    print("Graph Characteristics:")
    print("• Undirected graph (friendships are mutual)")
    print("• Unweighted edges (all friendships equal)")
    print("• Connected components may exist")
    print("• Small world property (short average distances)")
    
    print("\nLevel-based Analysis:")
    print("• Level 0: Yourself")
    print("• Level 1: Direct friends")
    print("• Level 2: Friends of friends")
    print("• Level k: People exactly k edges away")
    
    print("\nBFS Properties for Social Networks:")
    print("• Natural fit for level-wise exploration")
    print("• Guarantees shortest path distances")
    print("• Handles disconnected components")
    print("• Efficient for small-world networks")
    
    print("\nVideo Aggregation Strategy:")
    print("1. Find all people at target level")
    print("2. Collect all videos watched by these people")
    print("3. Count frequency of each video")
    print("4. Sort by frequency (ascending), then alphabetically")
    
    print("\nComplexity Analysis:")
    print("• BFS traversal: O(V + E)")
    print("• Video collection: O(people_at_level × avg_videos)")
    print("• Sorting: O(unique_videos × log(unique_videos))")
    print("• Total: O(V + E + unique_videos × log(unique_videos))")

def demonstrate_video_aggregation():
    """Demonstrate video aggregation and sorting"""
    print("\n=== Video Aggregation Demo ===")
    
    # Example data
    friends_at_level = [1, 2]
    watchedVideos = [["A","B"],["C"],["B","C"],["D"]]
    
    print(f"Friends at target level: {friends_at_level}")
    print(f"Their watched videos:")
    
    all_videos = []
    for person in friends_at_level:
        videos = watchedVideos[person]
        print(f"  Person {person}: {videos}")
        all_videos.extend(videos)
    
    print(f"\nAll videos from this level: {all_videos}")
    
    # Count frequencies
    video_count = Counter(all_videos)
    print(f"Video frequencies: {dict(video_count)}")
    
    # Demonstrate sorting process
    print(f"\nSorting process:")
    print(f"Sort key: (frequency, video_name)")
    
    videos_with_keys = [(video_count[video], video) for video in video_count]
    print(f"Videos with sort keys: {videos_with_keys}")
    
    sorted_result = sorted(video_count.keys(), key=lambda x: (video_count[x], x))
    print(f"Final sorted result: {sorted_result}")
    
    print(f"\nSorting rules:")
    print(f"• Primary: Lower frequency comes first")
    print(f"• Secondary: Alphabetical order for ties")
    print(f"• This prioritizes less common videos")

def compare_graph_traversal_approaches():
    """Compare different graph traversal approaches"""
    print("\n=== Graph Traversal Approaches Comparison ===")
    
    print("1. **BFS Level Tracking (Recommended):**")
    print("   ✅ Natural level-wise exploration")
    print("   ✅ Efficient queue-based implementation")
    print("   ✅ Clear separation of levels")
    print("   ✅ Early termination when target level reached")
    
    print("\n2. **BFS with Distance Array:**")
    print("   ✅ Explicit distance tracking")
    print("   ✅ Can find all levels in one pass")
    print("   ✅ Good for multiple level queries")
    print("   ❌ Slight memory overhead")
    
    print("\n3. **DFS Recursive:**")
    print("   ✅ Natural recursive structure")
    print("   ✅ Explicit level parameter")
    print("   ❌ Stack space concerns for deep graphs")
    print("   ❌ More complex backtracking logic")
    
    print("\n4. **Level Sets Tracking:**")
    print("   ✅ Clear level boundaries")
    print("   ✅ Set operations for friend management")
    print("   ✅ Good for analysis and debugging")
    print("   ❌ Potential memory overhead for large levels")
    
    print("\n5. **Dijkstra Variant (Educational):**")
    print("   ✅ Handles weighted graphs generally")
    print("   ✅ Priority queue flexibility")
    print("   ❌ Overkill for unweighted graphs")
    print("   ❌ Additional logarithmic factor")
    
    print("\nOptimal Choice:")
    print("• **BFS Level Tracking** for this specific problem")
    print("• **BFS Distance Array** for multiple queries")
    print("• **DFS** for educational purposes")
    print("• **Dijkstra** only if weights added later")

def analyze_real_world_applications():
    """Analyze real-world applications of friend level analysis"""
    print("\n=== Real-World Applications ===")
    
    print("1. **Social Media Recommendations:**")
    print("   • Find content from friends at specific distances")
    print("   • Expand recommendation beyond direct connections")
    print("   • Balance relevance with diversity")
    
    print("\n2. **Viral Marketing Analysis:**")
    print("   • Track information spread through network levels")
    print("   • Identify influential nodes at each level")
    print("   • Optimize marketing reach strategies")
    
    print("\n3. **Contact Tracing:**")
    print("   • Find people at specific infection distances")
    print("   • Prioritize testing based on network distance")
    print("   • Track disease spread patterns")
    
    print("\n4. **Professional Networks:**")
    print("   • Find job opportunities through connections")
    print("   • Identify skill gaps at different network levels")
    print("   • Build career development strategies")
    
    print("\n5. **Content Curation:**")
    print("   • Aggregate interests from friend networks")
    print("   • Find trending topics at specific social distances")
    print("   • Personalize content based on network analysis")
    
    print("\n6. **Research Collaboration:**")
    print("   • Find experts through academic networks")
    print("   • Identify collaboration opportunities")
    print("   • Track knowledge diffusion patterns")
    
    print("\nKey Insights:")
    print("• Network distance affects content relevance")
    print("• Level-based analysis reveals different perspectives")
    print("• Frequency analysis shows content popularity")
    print("• Sorting provides prioritized recommendations")
    print("• BFS naturally models social information flow")

if __name__ == "__main__":
    test_watched_videos_by_friends()
    demonstrate_friend_levels()
    analyze_social_network_patterns()
    demonstrate_video_aggregation()
    compare_graph_traversal_approaches()
    analyze_real_world_applications()

"""
Shortest Path Concepts:
1. Level-wise Graph Traversal with BFS
2. Social Network Distance Analysis
3. Unweighted Shortest Path in Social Graphs
4. Content Aggregation and Frequency Analysis
5. Multi-level Graph Exploration

Key Problem Insights:
- Find people at exactly k edges away (shortest path distance)
- Social networks naturally form small-world graphs
- BFS perfect for level-wise exploration
- Content aggregation requires frequency counting and sorting

Algorithm Strategy:
1. Use BFS to find all people at target level
2. Collect videos watched by people at that level
3. Count frequency of each video
4. Sort by frequency (ascending), then alphabetically

BFS Level Tracking:
- Process friends level by level using queue
- Track visited nodes to avoid cycles
- Stop when target level reached
- Natural fit for social network analysis

Social Network Properties:
- Undirected friendship graphs
- Small average distances (6 degrees of separation)
- Level-based content relevance
- Information diffusion through network levels

Content Aggregation:
- Collect videos from all people at target level
- Count frequencies using Counter data structure
- Sort by frequency (prefer rare content), then alphabetically
- Demonstrates preference for diverse recommendations

Real-world Applications:
- Social media content recommendations
- Viral marketing and influence analysis
- Contact tracing and epidemic modeling
- Professional networking and collaboration
- Research collaboration networks

This problem demonstrates BFS for social network
analysis and content recommendation systems.
"""
