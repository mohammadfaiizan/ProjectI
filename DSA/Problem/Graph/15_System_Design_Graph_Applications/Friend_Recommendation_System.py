"""
Friend Recommendation System - Advanced System Design
Difficulty: Medium

This file implements a comprehensive friend recommendation system that combines
multiple algorithms and data sources to suggest meaningful connections in
social networks. Used by platforms like Facebook, LinkedIn, and Twitter.

Key Concepts:
1. Collaborative Filtering
2. Graph-based Recommendations
3. Content-based Filtering
4. Hybrid Recommendation Systems
5. Machine Learning Integration
6. Real-time Recommendation Updates
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import heapq
import math
import random

class User:
    """Enhanced user representation for recommendations"""
    
    def __init__(self, user_id: int, profile: Dict):
        self.user_id = user_id
        self.profile = profile
        self.friends = set()
        self.interests = set(profile.get('interests', []))
        self.location = profile.get('location', '')
        self.age = profile.get('age', 0)
        self.education = profile.get('education', '')
        self.workplace = profile.get('workplace', '')
        self.mutual_friends_cache = {}
        self.recommendation_scores = {}

class FriendRecommendationSystem:
    """Advanced friend recommendation system"""
    
    def __init__(self):
        self.users = {}
        self.friendship_graph = defaultdict(set)
        self.interest_graph = defaultdict(set)  # interest -> users
        self.location_graph = defaultdict(set)  # location -> users
        self.workplace_graph = defaultdict(set)  # workplace -> users
        self.education_graph = defaultdict(set)  # education -> users
        
        # Recommendation weights
        self.weights = {
            'mutual_friends': 0.4,
            'common_interests': 0.2,
            'location_proximity': 0.15,
            'workplace': 0.1,
            'education': 0.1,
            'age_similarity': 0.05
        }
    
    def add_user(self, user_id: int, profile: Dict):
        """Add user to the system"""
        user = User(user_id, profile)
        self.users[user_id] = user
        
        # Update auxiliary graphs
        for interest in user.interests:
            self.interest_graph[interest].add(user_id)
        
        if user.location:
            self.location_graph[user.location].add(user_id)
        
        if user.workplace:
            self.workplace_graph[user.workplace].add(user_id)
        
        if user.education:
            self.education_graph[user.education].add(user_id)
    
    def add_friendship(self, user1_id: int, user2_id: int):
        """Add friendship between two users"""
        if user1_id in self.users and user2_id in self.users:
            self.users[user1_id].friends.add(user2_id)
            self.users[user2_id].friends.add(user1_id)
            self.friendship_graph[user1_id].add(user2_id)
            self.friendship_graph[user2_id].add(user1_id)
            
            # Clear recommendation caches
            self.users[user1_id].recommendation_scores.clear()
            self.users[user2_id].recommendation_scores.clear()
    
    def mutual_friends_score(self, user1_id: int, user2_id: int) -> float:
        """
        Approach 1: Mutual Friends Recommendation
        
        Score based on number of mutual friends.
        
        Time: O(min(F1, F2)), Space: O(1)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return 0.0
        
        friends1 = self.users[user1_id].friends
        friends2 = self.users[user2_id].friends
        
        mutual_count = len(friends1.intersection(friends2))
        
        # Normalize by geometric mean of friend counts
        if len(friends1) > 0 and len(friends2) > 0:
            normalization = math.sqrt(len(friends1) * len(friends2))
            return mutual_count / normalization
        
        return mutual_count * 0.1  # Small score for users with few friends
    
    def common_interests_score(self, user1_id: int, user2_id: int) -> float:
        """
        Approach 2: Common Interests Recommendation
        
        Score based on shared interests and hobbies.
        
        Time: O(min(I1, I2)), Space: O(1)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return 0.0
        
        interests1 = self.users[user1_id].interests
        interests2 = self.users[user2_id].interests
        
        if not interests1 or not interests2:
            return 0.0
        
        common_count = len(interests1.intersection(interests2))
        
        # Jaccard similarity
        union_count = len(interests1.union(interests2))
        return common_count / union_count if union_count > 0 else 0.0
    
    def location_proximity_score(self, user1_id: int, user2_id: int) -> float:
        """
        Approach 3: Location-based Recommendation
        
        Score based on geographical proximity.
        
        Time: O(1), Space: O(1)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return 0.0
        
        location1 = self.users[user1_id].location
        location2 = self.users[user2_id].location
        
        if not location1 or not location2:
            return 0.0
        
        # Exact match gets full score
        if location1 == location2:
            return 1.0
        
        # Could implement more sophisticated distance calculation
        # For now, partial matches get partial score
        if any(word in location2.lower() for word in location1.lower().split()):
            return 0.5
        
        return 0.0
    
    def workplace_score(self, user1_id: int, user2_id: int) -> float:
        """
        Approach 4: Workplace-based Recommendation
        
        Score based on shared workplace or industry.
        
        Time: O(1), Space: O(1)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return 0.0
        
        workplace1 = self.users[user1_id].workplace
        workplace2 = self.users[user2_id].workplace
        
        if not workplace1 or not workplace2:
            return 0.0
        
        return 1.0 if workplace1 == workplace2 else 0.0
    
    def education_score(self, user1_id: int, user2_id: int) -> float:
        """
        Approach 5: Education-based Recommendation
        
        Score based on shared educational background.
        
        Time: O(1), Space: O(1)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return 0.0
        
        education1 = self.users[user1_id].education
        education2 = self.users[user2_id].education
        
        if not education1 or not education2:
            return 0.0
        
        return 1.0 if education1 == education2 else 0.0
    
    def age_similarity_score(self, user1_id: int, user2_id: int) -> float:
        """
        Approach 6: Age Similarity Recommendation
        
        Score based on age proximity.
        
        Time: O(1), Space: O(1)
        """
        if user1_id not in self.users or user2_id not in self.users:
            return 0.0
        
        age1 = self.users[user1_id].age
        age2 = self.users[user2_id].age
        
        if age1 <= 0 or age2 <= 0:
            return 0.0
        
        age_diff = abs(age1 - age2)
        
        # Gaussian-like decay with age difference
        return math.exp(-age_diff / 10.0)
    
    def collaborative_filtering_score(self, user_id: int, candidate_id: int) -> float:
        """
        Approach 7: Collaborative Filtering
        
        Score based on similar users' friendship patterns.
        
        Time: O(F * F_avg), Space: O(F)
        """
        if user_id not in self.users or candidate_id not in self.users:
            return 0.0
        
        user_friends = self.users[user_id].friends
        candidate_friends = self.users[candidate_id].friends
        
        # Find users similar to current user
        similar_users = []
        
        for other_user_id in self.users:
            if other_user_id != user_id and other_user_id != candidate_id:
                other_friends = self.users[other_user_id].friends
                
                # Calculate similarity based on mutual friends
                if user_friends and other_friends:
                    similarity = len(user_friends.intersection(other_friends)) / len(user_friends.union(other_friends))
                    if similarity > 0.1:  # Threshold for similarity
                        similar_users.append((other_user_id, similarity))
        
        # Score based on how many similar users are friends with candidate
        total_score = 0.0
        total_weight = 0.0
        
        for similar_user_id, similarity in similar_users:
            if candidate_id in self.users[similar_user_id].friends:
                total_score += similarity
            total_weight += similarity
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def calculate_composite_score(self, user_id: int, candidate_id: int) -> float:
        """
        Approach 8: Hybrid Recommendation Score
        
        Combine multiple signals with weighted scoring.
        
        Time: O(F * F_avg), Space: O(1)
        """
        if user_id == candidate_id or candidate_id in self.users[user_id].friends:
            return 0.0
        
        scores = {
            'mutual_friends': self.mutual_friends_score(user_id, candidate_id),
            'common_interests': self.common_interests_score(user_id, candidate_id),
            'location_proximity': self.location_proximity_score(user_id, candidate_id),
            'workplace': self.workplace_score(user_id, candidate_id),
            'education': self.education_score(user_id, candidate_id),
            'age_similarity': self.age_similarity_score(user_id, candidate_id)
        }
        
        # Weighted combination
        composite_score = sum(self.weights[signal] * score 
                            for signal, score in scores.items())
        
        # Add collaborative filtering bonus
        collab_score = self.collaborative_filtering_score(user_id, candidate_id)
        composite_score += 0.1 * collab_score
        
        return composite_score
    
    def get_friend_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Approach 9: Generate Friend Recommendations
        
        Get top friend recommendations with explanations.
        
        Time: O(V * F * F_avg), Space: O(V)
        """
        if user_id not in self.users:
            return []
        
        user_friends = self.users[user_id].friends
        recommendations = []
        
        # Score all potential candidates
        for candidate_id in self.users:
            if candidate_id != user_id and candidate_id not in user_friends:
                score = self.calculate_composite_score(user_id, candidate_id)
                
                if score > 0.01:  # Minimum threshold
                    # Generate explanation
                    explanation = self._generate_explanation(user_id, candidate_id)
                    recommendations.append((candidate_id, score, explanation))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def get_recommendations_by_category(self, user_id: int) -> Dict[str, List[Tuple[int, float]]]:
        """
        Approach 10: Categorized Recommendations
        
        Group recommendations by different categories.
        
        Time: O(V), Space: O(V)
        """
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        categories = {
            'mutual_friends': [],
            'same_workplace': [],
            'same_education': [],
            'same_location': [],
            'similar_interests': []
        }
        
        for candidate_id, candidate in self.users.items():
            if candidate_id == user_id or candidate_id in user.friends:
                continue
            
            # Mutual friends
            mutual_score = self.mutual_friends_score(user_id, candidate_id)
            if mutual_score > 0.1:
                categories['mutual_friends'].append((candidate_id, mutual_score))
            
            # Same workplace
            if user.workplace and user.workplace == candidate.workplace:
                categories['same_workplace'].append((candidate_id, 1.0))
            
            # Same education
            if user.education and user.education == candidate.education:
                categories['same_education'].append((candidate_id, 1.0))
            
            # Same location
            if user.location and user.location == candidate.location:
                categories['same_location'].append((candidate_id, 1.0))
            
            # Similar interests
            interest_score = self.common_interests_score(user_id, candidate_id)
            if interest_score > 0.2:
                categories['similar_interests'].append((candidate_id, interest_score))
        
        # Sort each category
        for category in categories:
            categories[category].sort(key=lambda x: x[1], reverse=True)
            categories[category] = categories[category][:5]  # Top 5 per category
        
        return categories
    
    def _generate_explanation(self, user_id: int, candidate_id: int) -> Dict:
        """Generate explanation for why this user is recommended"""
        explanation = {
            'reasons': [],
            'mutual_friends_count': 0,
            'common_interests': [],
            'shared_attributes': []
        }
        
        user = self.users[user_id]
        candidate = self.users[candidate_id]
        
        # Mutual friends
        mutual_friends = user.friends.intersection(candidate.friends)
        if mutual_friends:
            explanation['mutual_friends_count'] = len(mutual_friends)
            explanation['reasons'].append(f"{len(mutual_friends)} mutual friends")
        
        # Common interests
        common_interests = user.interests.intersection(candidate.interests)
        if common_interests:
            explanation['common_interests'] = list(common_interests)
            explanation['reasons'].append(f"Shared interests: {', '.join(list(common_interests)[:3])}")
        
        # Shared attributes
        if user.workplace and user.workplace == candidate.workplace:
            explanation['shared_attributes'].append('workplace')
            explanation['reasons'].append(f"Works at {user.workplace}")
        
        if user.education and user.education == candidate.education:
            explanation['shared_attributes'].append('education')
            explanation['reasons'].append(f"Studied at {user.education}")
        
        if user.location and user.location == candidate.location:
            explanation['shared_attributes'].append('location')
            explanation['reasons'].append(f"Lives in {user.location}")
        
        return explanation
    
    def update_recommendation_weights(self, feedback: Dict[str, float]):
        """Update recommendation weights based on user feedback"""
        for signal, adjustment in feedback.items():
            if signal in self.weights:
                self.weights[signal] = max(0.0, min(1.0, self.weights[signal] + adjustment))
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for signal in self.weights:
                self.weights[signal] /= total_weight

def test_friend_recommendation_system():
    """Test friend recommendation system"""
    print("=== Testing Friend Recommendation System ===")
    
    # Create recommendation system
    system = FriendRecommendationSystem()
    
    # Add users with profiles
    users_data = [
        (1, {'name': 'Alice', 'interests': ['music', 'travel', 'photography'], 
             'location': 'San Francisco', 'workplace': 'Google', 'education': 'Stanford', 'age': 28}),
        (2, {'name': 'Bob', 'interests': ['music', 'sports', 'technology'], 
             'location': 'San Francisco', 'workplace': 'Facebook', 'education': 'MIT', 'age': 30}),
        (3, {'name': 'Charlie', 'interests': ['travel', 'photography', 'art'], 
             'location': 'New York', 'workplace': 'Google', 'education': 'Stanford', 'age': 27}),
        (4, {'name': 'Diana', 'interests': ['sports', 'fitness', 'cooking'], 
             'location': 'San Francisco', 'workplace': 'Apple', 'education': 'Berkeley', 'age': 29}),
        (5, {'name': 'Eve', 'interests': ['technology', 'art', 'music'], 
             'location': 'Seattle', 'workplace': 'Microsoft', 'education': 'MIT', 'age': 31})
    ]
    
    for user_id, profile in users_data:
        system.add_user(user_id, profile)
        print(f"Added user: {profile['name']}")
    
    # Add some friendships
    friendships = [(1, 2), (2, 4), (3, 5)]
    for user1, user2 in friendships:
        system.add_friendship(user1, user2)
        print(f"Added friendship: {users_data[user1-1][1]['name']} <-> {users_data[user2-1][1]['name']}")
    
    # Test recommendations for Alice (user 1)
    print(f"\n--- Friend Recommendations for Alice ---")
    recommendations = system.get_friend_recommendations(1, 5)
    
    for candidate_id, score, explanation in recommendations:
        candidate_name = users_data[candidate_id-1][1]['name']
        print(f"{candidate_name}: {score:.3f}")
        print(f"  Reasons: {', '.join(explanation['reasons'])}")
        if explanation['mutual_friends_count'] > 0:
            print(f"  Mutual friends: {explanation['mutual_friends_count']}")
        print()
    
    # Test categorized recommendations
    print(f"--- Categorized Recommendations for Alice ---")
    categorized = system.get_recommendations_by_category(1)
    
    for category, recs in categorized.items():
        if recs:
            print(f"{category.replace('_', ' ').title()}:")
            for candidate_id, score in recs[:3]:
                candidate_name = users_data[candidate_id-1][1]['name']
                print(f"  {candidate_name}: {score:.3f}")
            print()

def demonstrate_recommendation_concepts():
    """Demonstrate friend recommendation concepts"""
    print("\n=== Friend Recommendation Concepts ===")
    
    print("Core Algorithms:")
    print("• Collaborative Filtering: Users with similar friends")
    print("• Content-based Filtering: Users with similar profiles")
    print("• Graph-based Methods: Network structure analysis")
    print("• Hybrid Approaches: Combine multiple signals")
    
    print("\nRecommendation Signals:")
    print("• Mutual Friends: Strongest signal for social connections")
    print("• Common Interests: Shared hobbies and activities")
    print("• Location Proximity: Geographic closeness")
    print("• Workplace/Education: Professional and academic connections")
    print("• Demographics: Age, gender, and other attributes")
    
    print("\nSystem Design Challenges:")
    print("• Scalability: Handle millions of users and relationships")
    print("• Real-time Updates: Incorporate new friendships immediately")
    print("• Privacy: Respect user privacy settings")
    print("• Diversity: Avoid echo chambers and filter bubbles")
    print("• Cold Start: Recommend for new users with limited data")
    
    print("\nEvaluation Metrics:")
    print("• Precision: Fraction of recommendations that are accepted")
    print("• Recall: Fraction of potential friends that are recommended")
    print("• Diversity: Variety in recommended user types")
    print("• Novelty: Recommending unexpected but relevant connections")

if __name__ == "__main__":
    test_friend_recommendation_system()
    demonstrate_recommendation_concepts()

"""
Friend Recommendation System demonstrates the complexity of building
personalized recommendation systems that balance multiple signals
while maintaining user privacy and system scalability.
"""
