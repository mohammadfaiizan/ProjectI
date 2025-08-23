"""
Social Network Basics - System Design Implementation
Difficulty: Easy

This file implements fundamental social network operations and algorithms
that form the foundation of social media platforms and networking applications.

Key Concepts:
1. User Relationship Management
2. Friend Connections and Networks
3. Social Graph Traversal
4. Mutual Friends and Connections
5. Network Statistics and Analysis
6. Privacy and Access Control
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime
import heapq

class User:
    """Represents a user in the social network"""
    
    def __init__(self, user_id: int, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.friends = set()
        self.blocked_users = set()
        self.privacy_settings = {
            'profile_visibility': 'friends',  # 'public', 'friends', 'private'
            'friend_list_visibility': 'friends',
            'allow_friend_requests': True
        }
        self.created_at = datetime.now()
    
    def __repr__(self):
        return f"User({self.user_id}, {self.name})"

class SocialNetworkBasics:
    """Basic social network operations and algorithms"""
    
    def __init__(self):
        self.users = {}  # user_id -> User
        self.friendship_graph = defaultdict(set)
        self.pending_requests = defaultdict(set)  # user_id -> set of pending requests
        self.blocked_relationships = defaultdict(set)
    
    def add_user(self, user_id: int, name: str, email: str) -> bool:
        """
        Approach 1: Add User to Network
        
        Add a new user to the social network.
        
        Time: O(1), Space: O(1)
        """
        if user_id in self.users:
            return False
        
        self.users[user_id] = User(user_id, name, email)
        return True
    
    def send_friend_request(self, sender_id: int, receiver_id: int) -> bool:
        """
        Approach 2: Send Friend Request
        
        Send a friend request between two users.
        
        Time: O(1), Space: O(1)
        """
        if (sender_id not in self.users or receiver_id not in self.users or
            sender_id == receiver_id):
            return False
        
        sender = self.users[sender_id]
        receiver = self.users[receiver_id]
        
        # Check if already friends
        if receiver_id in sender.friends:
            return False
        
        # Check if blocked
        if (receiver_id in sender.blocked_users or 
            sender_id in receiver.blocked_users):
            return False
        
        # Check privacy settings
        if not receiver.privacy_settings['allow_friend_requests']:
            return False
        
        # Add to pending requests
        self.pending_requests[receiver_id].add(sender_id)
        return True
    
    def accept_friend_request(self, user_id: int, requester_id: int) -> bool:
        """
        Approach 3: Accept Friend Request
        
        Accept a pending friend request.
        
        Time: O(1), Space: O(1)
        """
        if (user_id not in self.users or requester_id not in self.users or
            requester_id not in self.pending_requests[user_id]):
            return False
        
        # Remove from pending requests
        self.pending_requests[user_id].remove(requester_id)
        
        # Add friendship both ways
        self.users[user_id].friends.add(requester_id)
        self.users[requester_id].friends.add(user_id)
        
        # Update friendship graph
        self.friendship_graph[user_id].add(requester_id)
        self.friendship_graph[requester_id].add(user_id)
        
        return True
    
    def get_mutual_friends(self, user1_id: int, user2_id: int) -> List[int]:
        """
        Approach 4: Find Mutual Friends
        
        Find common friends between two users.
        
        Time: O(min(F1, F2)), Space: O(min(F1, F2))
        """
        if user1_id not in self.users or user2_id not in self.users:
            return []
        
        friends1 = self.users[user1_id].friends
        friends2 = self.users[user2_id].friends
        
        return list(friends1.intersection(friends2))
    
    def get_friends_of_friends(self, user_id: int, max_depth: int = 2) -> Dict[int, int]:
        """
        Approach 5: Friends of Friends (Network Expansion)
        
        Find friends at different degrees of separation.
        
        Time: O(V + E), Space: O(V)
        """
        if user_id not in self.users:
            return {}
        
        visited = set([user_id])
        friends_by_degree = defaultdict(set)
        queue = deque([(user_id, 0)])
        
        while queue:
            current_user, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for friend_id in self.users[current_user].friends:
                if friend_id not in visited:
                    visited.add(friend_id)
                    friends_by_degree[depth + 1].add(friend_id)
                    queue.append((friend_id, depth + 1))
        
        # Convert to dict with degree as key
        result = {}
        for degree, friends in friends_by_degree.items():
            for friend_id in friends:
                result[friend_id] = degree
        
        return result
    
    def suggest_friends(self, user_id: int, max_suggestions: int = 10) -> List[Tuple[int, int]]:
        """
        Approach 6: Friend Suggestion Algorithm
        
        Suggest potential friends based on mutual connections.
        
        Time: O(V + E), Space: O(V)
        """
        if user_id not in self.users:
            return []
        
        user_friends = self.users[user_id].friends
        suggestion_scores = defaultdict(int)
        
        # Score based on mutual friends
        for friend_id in user_friends:
            for friend_of_friend in self.users[friend_id].friends:
                if (friend_of_friend != user_id and 
                    friend_of_friend not in user_friends and
                    friend_of_friend not in self.users[user_id].blocked_users):
                    suggestion_scores[friend_of_friend] += 1
        
        # Sort by score and return top suggestions
        suggestions = sorted(suggestion_scores.items(), 
                           key=lambda x: x[1], reverse=True)
        
        return suggestions[:max_suggestions]
    
    def find_shortest_connection_path(self, user1_id: int, user2_id: int) -> List[int]:
        """
        Approach 7: Shortest Path Between Users
        
        Find shortest friendship path between two users.
        
        Time: O(V + E), Space: O(V)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return []
        
        if user1_id == user2_id:
            return [user1_id]
        
        # BFS to find shortest path
        queue = deque([(user1_id, [user1_id])])
        visited = set([user1_id])
        
        while queue:
            current_user, path = queue.popleft()
            
            for friend_id in self.users[current_user].friends:
                if friend_id == user2_id:
                    return path + [friend_id]
                
                if friend_id not in visited:
                    visited.add(friend_id)
                    queue.append((friend_id, path + [friend_id]))
        
        return []  # No connection found
    
    def get_network_statistics(self) -> Dict:
        """
        Approach 8: Network Statistics Analysis
        
        Calculate various network statistics.
        
        Time: O(V + E), Space: O(V)
        """
        total_users = len(self.users)
        total_friendships = sum(len(user.friends) for user in self.users.values()) // 2
        
        if total_users == 0:
            return {'error': 'No users in network'}
        
        # Calculate degree distribution
        degrees = [len(user.friends) for user in self.users.values()]
        avg_degree = sum(degrees) / total_users
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        # Find connected components
        components = self._find_connected_components()
        
        # Calculate clustering coefficient (simplified)
        clustering_coeffs = []
        for user_id, user in self.users.items():
            if len(user.friends) < 2:
                continue
            
            friend_connections = 0
            friends_list = list(user.friends)
            
            for i in range(len(friends_list)):
                for j in range(i + 1, len(friends_list)):
                    if friends_list[j] in self.users[friends_list[i]].friends:
                        friend_connections += 1
            
            possible_connections = len(user.friends) * (len(user.friends) - 1) // 2
            if possible_connections > 0:
                clustering_coeffs.append(friend_connections / possible_connections)
        
        avg_clustering = sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0
        
        return {
            'total_users': total_users,
            'total_friendships': total_friendships,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'connected_components': len(components),
            'largest_component_size': max(len(comp) for comp in components) if components else 0,
            'average_clustering_coefficient': avg_clustering,
            'network_density': (2 * total_friendships) / (total_users * (total_users - 1)) if total_users > 1 else 0
        }
    
    def _find_connected_components(self) -> List[Set[int]]:
        """Helper method to find connected components"""
        visited = set()
        components = []
        
        for user_id in self.users:
            if user_id not in visited:
                component = set()
                queue = deque([user_id])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        for friend_id in self.users[current].friends:
                            if friend_id not in visited:
                                queue.append(friend_id)
                
                components.append(component)
        
        return components
    
    def block_user(self, blocker_id: int, blocked_id: int) -> bool:
        """
        Approach 9: Block User Functionality
        
        Block a user and remove existing friendship.
        
        Time: O(1), Space: O(1)
        """
        if blocker_id not in self.users or blocked_id not in self.users:
            return False
        
        # Remove friendship if exists
        if blocked_id in self.users[blocker_id].friends:
            self.users[blocker_id].friends.remove(blocked_id)
            self.users[blocked_id].friends.remove(blocker_id)
            self.friendship_graph[blocker_id].discard(blocked_id)
            self.friendship_graph[blocked_id].discard(blocker_id)
        
        # Add to blocked list
        self.users[blocker_id].blocked_users.add(blocked_id)
        
        # Remove any pending requests
        self.pending_requests[blocker_id].discard(blocked_id)
        self.pending_requests[blocked_id].discard(blocker_id)
        
        return True
    
    def get_user_profile(self, viewer_id: int, target_id: int) -> Optional[Dict]:
        """
        Approach 10: Privacy-Aware Profile Access
        
        Get user profile respecting privacy settings.
        
        Time: O(1), Space: O(1)
        """
        if target_id not in self.users:
            return None
        
        target_user = self.users[target_id]
        visibility = target_user.privacy_settings['profile_visibility']
        
        # Check access permissions
        if visibility == 'private':
            if viewer_id != target_id:
                return {'error': 'Profile is private'}
        elif visibility == 'friends':
            if viewer_id != target_id and viewer_id not in target_user.friends:
                return {'name': target_user.name, 'limited_access': True}
        
        # Return full or limited profile
        profile = {
            'user_id': target_user.user_id,
            'name': target_user.name,
            'friend_count': len(target_user.friends)
        }
        
        # Add friend list if allowed
        if (visibility == 'public' or 
            viewer_id == target_id or 
            viewer_id in target_user.friends):
            
            friend_list_visibility = target_user.privacy_settings['friend_list_visibility']
            if (friend_list_visibility == 'public' or 
                viewer_id == target_id or 
                (friend_list_visibility == 'friends' and viewer_id in target_user.friends)):
                profile['friends'] = list(target_user.friends)
        
        return profile

def test_social_network_basics():
    """Test social network basic operations"""
    print("=== Testing Social Network Basics ===")
    
    # Create social network
    network = SocialNetworkBasics()
    
    # Add users
    users = [
        (1, "Alice", "alice@email.com"),
        (2, "Bob", "bob@email.com"),
        (3, "Charlie", "charlie@email.com"),
        (4, "Diana", "diana@email.com"),
        (5, "Eve", "eve@email.com")
    ]
    
    for user_id, name, email in users:
        success = network.add_user(user_id, name, email)
        print(f"Added user {name}: {success}")
    
    # Test friend requests and connections
    print(f"\n--- Friend Requests ---")
    network.send_friend_request(1, 2)  # Alice -> Bob
    network.send_friend_request(1, 3)  # Alice -> Charlie
    network.send_friend_request(2, 3)  # Bob -> Charlie
    network.send_friend_request(3, 4)  # Charlie -> Diana
    
    # Accept some requests
    network.accept_friend_request(2, 1)  # Bob accepts Alice
    network.accept_friend_request(3, 1)  # Charlie accepts Alice
    network.accept_friend_request(3, 2)  # Charlie accepts Bob
    network.accept_friend_request(4, 3)  # Diana accepts Charlie
    
    # Test mutual friends
    print(f"\n--- Mutual Friends ---")
    mutual = network.get_mutual_friends(1, 2)  # Alice and Bob
    print(f"Mutual friends between Alice and Bob: {mutual}")
    
    # Test friend suggestions
    print(f"\n--- Friend Suggestions ---")
    suggestions = network.suggest_friends(1, 3)  # Suggestions for Alice
    print(f"Friend suggestions for Alice: {suggestions}")
    
    # Test shortest path
    print(f"\n--- Shortest Connection Path ---")
    path = network.find_shortest_connection_path(1, 4)  # Alice to Diana
    print(f"Shortest path from Alice to Diana: {path}")
    
    # Test network statistics
    print(f"\n--- Network Statistics ---")
    stats = network.get_network_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test privacy features
    print(f"\n--- Privacy Features ---")
    profile = network.get_user_profile(2, 1)  # Bob viewing Alice
    print(f"Bob viewing Alice's profile: {profile}")

def demonstrate_social_network_concepts():
    """Demonstrate social network concepts and algorithms"""
    print("\n=== Social Network Concepts ===")
    
    print("Key Concepts:")
    print("• Graph Representation: Users as nodes, friendships as edges")
    print("• Bidirectional Relationships: Friendships are mutual")
    print("• Privacy Controls: Access levels and visibility settings")
    print("• Network Effects: Mutual friends, friend suggestions")
    
    print("\nCommon Algorithms:")
    print("• BFS: Shortest path, friends of friends")
    print("• Connected Components: Network clusters")
    print("• Degree Centrality: Popular users (high friend count)")
    print("• Clustering Coefficient: How interconnected friend groups are")
    
    print("\nSystem Design Considerations:")
    print("• Scalability: Handle millions of users and relationships")
    print("• Privacy: Respect user privacy settings")
    print("• Performance: Fast friend suggestions and searches")
    print("• Consistency: Maintain data integrity across operations")
    
    print("\nReal-World Applications:")
    print("• Social Media Platforms: Facebook, LinkedIn, Twitter")
    print("• Professional Networks: Career connections and recommendations")
    print("• Gaming Networks: Friend lists and multiplayer connections")
    print("• Dating Apps: Mutual connections and compatibility")

def analyze_social_network_complexity():
    """Analyze complexity of social network operations"""
    print("\n=== Social Network Complexity Analysis ===")
    
    print("Operation Complexities:")
    
    print("\n1. **Basic Operations:**")
    print("   • Add User: O(1)")
    print("   • Send Friend Request: O(1)")
    print("   • Accept Request: O(1)")
    print("   • Block User: O(1)")
    
    print("\n2. **Search Operations:**")
    print("   • Find Mutual Friends: O(min(F1, F2))")
    print("   • Shortest Path: O(V + E) - BFS")
    print("   • Friends of Friends: O(V + E)")
    print("   • Friend Suggestions: O(V + E)")
    
    print("\n3. **Analytics Operations:**")
    print("   • Network Statistics: O(V + E)")
    print("   • Connected Components: O(V + E)")
    print("   • Clustering Coefficient: O(V * F^2)")
    
    print("\n4. **Scalability Considerations:**")
    print("   • V = number of users (millions to billions)")
    print("   • E = number of friendships (billions to trillions)")
    print("   • F = average friends per user (hundreds)")
    print("   • Need distributed systems for large scale")
    
    print("\nOptimization Strategies:")
    print("• Caching: Cache friend lists and suggestions")
    print("• Indexing: Database indexes on user relationships")
    print("• Sharding: Distribute users across multiple servers")
    print("• Denormalization: Store computed values for fast access")

if __name__ == "__main__":
    test_social_network_basics()
    demonstrate_social_network_concepts()
    analyze_social_network_complexity()

"""
Social Network Basics - Key Insights:

1. **Graph Modeling:**
   - Users as vertices, friendships as edges
   - Bidirectional relationships for mutual connections
   - Additional metadata for privacy and preferences
   - Efficient data structures for fast operations

2. **Core Operations:**
   - User management and authentication
   - Friend request workflow and acceptance
   - Privacy controls and access management
   - Blocking and content filtering

3. **Network Analysis:**
   - Mutual friends and connection discovery
   - Shortest paths between users
   - Friend suggestions based on network structure
   - Network statistics and health metrics

4. **System Design Principles:**
   - Scalability for millions of users
   - Privacy and security considerations
   - Performance optimization for real-time operations
   - Data consistency and integrity

5. **Real-World Applications:**
   - Social media platforms and networking
   - Professional networking and career development
   - Gaming and entertainment platforms
   - Dating and relationship applications

Social networks demonstrate the power of graph algorithms
in creating meaningful connections and communities at scale.
"""
