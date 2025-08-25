"""
355. Design Twitter - Multiple Approaches
Difficulty: Medium

Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and see the 10 most recent tweets in the user's news feed.

Implement the Twitter class:
- Twitter() Initializes your twitter object.
- void postTweet(int userId, int tweetId) Composes a new tweet with ID tweetId by the user userId.
- List<Integer> getNewsFeed(int userId) Retrieves the 10 most recent tweet IDs in the user's news feed.
- void follow(int followerId, int followeeId) The user with ID followerId started following the user with ID followeeId.
- void unfollow(int followerId, int followeeId) The user with ID followerId started unfollowing the user with ID followeeId.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque
import heapq
import time

class TwitterBasic:
    """
    Approach 1: Basic Implementation with Linear Search
    
    Simple implementation using lists and linear operations.
    
    Time Complexity:
    - postTweet: O(1)
    - follow/unfollow: O(1)
    - getNewsFeed: O(n * m) where n is tweets, m is followed users
    
    Space Complexity: O(users + tweets + follows)
    """
    
    def __init__(self):
        self.tweets = []  # (userId, tweetId, timestamp)
        self.follows = defaultdict(set)  # userId -> set of followees
        self.timestamp = 0
    
    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets.append((userId, tweetId, self.timestamp))
        self.timestamp += 1
    
    def getNewsFeed(self, userId: int) -> List[int]:
        # Get all users to include in feed (self + followees)
        relevant_users = {userId}
        relevant_users.update(self.follows[userId])
        
        # Find relevant tweets in reverse chronological order
        feed = []
        for user_id, tweet_id, timestamp in reversed(self.tweets):
            if user_id in relevant_users:
                feed.append(tweet_id)
                if len(feed) >= 10:
                    break
        
        return feed
    
    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.follows[followerId].add(followeeId)
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.follows[followerId].discard(followeeId)

class TwitterOptimized:
    """
    Approach 2: Optimized with Per-User Tweet Storage
    
    Store tweets per user and use merge-sort approach for news feed.
    
    Time Complexity:
    - postTweet: O(1)
    - follow/unfollow: O(1)
    - getNewsFeed: O(f * log(f) + 10) where f is number of followed users
    
    Space Complexity: O(users + tweets + follows)
    """
    
    def __init__(self):
        self.user_tweets = defaultdict(deque)  # userId -> deque of (tweetId, timestamp)
        self.follows = defaultdict(set)
        self.timestamp = 0
    
    def postTweet(self, userId: int, tweetId: int) -> None:
        self.user_tweets[userId].appendleft((tweetId, self.timestamp))
        self.timestamp += 1
        
        # Keep only recent tweets per user to save memory
        if len(self.user_tweets[userId]) > 10:
            self.user_tweets[userId].pop()
    
    def getNewsFeed(self, userId: int) -> List[int]:
        # Use heap to merge recent tweets from all relevant users
        heap = []
        
        # Add user's own tweets
        for i, (tweet_id, timestamp) in enumerate(self.user_tweets[userId]):
            if i >= 10:  # Only consider recent tweets
                break
            heapq.heappush(heap, (-timestamp, tweet_id, userId, i))
        
        # Add tweets from followed users
        for followee_id in self.follows[userId]:
            for i, (tweet_id, timestamp) in enumerate(self.user_tweets[followee_id]):
                if i >= 10:
                    break
                heapq.heappush(heap, (-timestamp, tweet_id, followee_id, i))
        
        # Extract top 10 tweets
        feed = []
        processed_users = set()
        
        while heap and len(feed) < 10:
            neg_timestamp, tweet_id, user_id, index = heapq.heappop(heap)
            feed.append(tweet_id)
            
            # Add next tweet from same user if available
            if (user_id, index) not in processed_users:
                processed_users.add((user_id, index))
                if index + 1 < len(self.user_tweets[user_id]) and index + 1 < 10:
                    next_tweet_id, next_timestamp = self.user_tweets[user_id][index + 1]
                    heapq.heappush(heap, (-next_timestamp, next_tweet_id, user_id, index + 1))
        
        return feed
    
    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.follows[followerId].add(followeeId)
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.follows[followerId].discard(followeeId)

class TwitterAdvanced:
    """
    Approach 3: Advanced with Features and Analytics
    
    Enhanced Twitter with user management, analytics, and additional features.
    
    Time Complexity:
    - postTweet: O(1)
    - follow/unfollow: O(1)
    - getNewsFeed: O(f * log(10))
    
    Space Complexity: O(users + tweets + follows + analytics)
    """
    
    def __init__(self):
        # Core data structures
        self.user_tweets = defaultdict(deque)
        self.follows = defaultdict(set)
        self.followers = defaultdict(set)  # Track followers for analytics
        self.timestamp = 0
        
        # Analytics
        self.user_stats = defaultdict(lambda: {
            'tweets_count': 0,
            'followers_count': 0,
            'following_count': 0,
            'feed_requests': 0
        })
        
        # Features
        self.tweet_metadata = {}  # tweetId -> metadata
        self.user_activity = defaultdict(list)  # Recent activity log
    
    def postTweet(self, userId: int, tweetId: int) -> None:
        current_time = time.time()
        
        # Store tweet
        self.user_tweets[userId].appendleft((tweetId, self.timestamp, current_time))
        self.timestamp += 1
        
        # Update statistics
        self.user_stats[userId]['tweets_count'] += 1
        
        # Store metadata
        self.tweet_metadata[tweetId] = {
            'author': userId,
            'timestamp': self.timestamp - 1,
            'created_at': current_time
        }
        
        # Log activity
        self.user_activity[userId].append(('post', tweetId, current_time))
        
        # Limit storage
        if len(self.user_tweets[userId]) > 50:
            self.user_tweets[userId].pop()
        
        if len(self.user_activity[userId]) > 100:
            self.user_activity[userId] = self.user_activity[userId][-50:]
    
    def getNewsFeed(self, userId: int) -> List[int]:
        self.user_stats[userId]['feed_requests'] += 1
        
        # Collect recent tweets from self and followees
        all_tweets = []
        
        # Own tweets
        for tweet_id, timestamp, created_at in list(self.user_tweets[userId])[:10]:
            all_tweets.append((timestamp, tweet_id))
        
        # Followees' tweets
        for followee_id in self.follows[userId]:
            for tweet_id, timestamp, created_at in list(self.user_tweets[followee_id])[:10]:
                all_tweets.append((timestamp, tweet_id))
        
        # Sort by timestamp and return top 10
        all_tweets.sort(reverse=True)
        return [tweet_id for _, tweet_id in all_tweets[:10]]
    
    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId and followeeId not in self.follows[followerId]:
            self.follows[followerId].add(followeeId)
            self.followers[followeeId].add(followerId)
            
            # Update statistics
            self.user_stats[followerId]['following_count'] += 1
            self.user_stats[followeeId]['followers_count'] += 1
            
            # Log activity
            current_time = time.time()
            self.user_activity[followerId].append(('follow', followeeId, current_time))
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.follows[followerId]:
            self.follows[followerId].remove(followeeId)
            self.followers[followeeId].discard(followerId)
            
            # Update statistics
            self.user_stats[followerId]['following_count'] -= 1
            self.user_stats[followeeId]['followers_count'] -= 1
            
            # Log activity
            current_time = time.time()
            self.user_activity[followerId].append(('unfollow', followeeId, current_time))
    
    def getUserStats(self, userId: int) -> dict:
        """Get user statistics"""
        return self.user_stats[userId].copy()
    
    def getTweetInfo(self, tweetId: int) -> dict:
        """Get tweet metadata"""
        return self.tweet_metadata.get(tweetId, {})
    
    def getFollowers(self, userId: int) -> List[int]:
        """Get list of followers"""
        return list(self.followers[userId])
    
    def getFollowing(self, userId: int) -> List[int]:
        """Get list of users being followed"""
        return list(self.follows[userId])
    
    def getUserActivity(self, userId: int, limit: int = 10) -> List[tuple]:
        """Get recent user activity"""
        return self.user_activity[userId][-limit:]

class TwitterMemoryOptimized:
    """
    Approach 4: Memory-Optimized for Large Scale
    
    Optimized for memory usage with limited tweet storage per user.
    
    Time Complexity:
    - postTweet: O(1)
    - follow/unfollow: O(1)
    - getNewsFeed: O(f * 10) where f is followed users
    
    Space Complexity: O(users * 10 + follows)
    """
    
    def __init__(self, max_tweets_per_user: int = 10):
        self.max_tweets_per_user = max_tweets_per_user
        self.user_tweets = defaultdict(lambda: deque(maxlen=max_tweets_per_user))
        self.follows = defaultdict(set)
        self.timestamp = 0
    
    def postTweet(self, userId: int, tweetId: int) -> None:
        self.user_tweets[userId].appendleft((tweetId, self.timestamp))
        self.timestamp += 1
    
    def getNewsFeed(self, userId: int) -> List[int]:
        # Collect all relevant tweets
        candidates = []
        
        # Add own tweets
        for tweet_id, timestamp in self.user_tweets[userId]:
            candidates.append((timestamp, tweet_id))
        
        # Add followees' tweets
        for followee_id in self.follows[userId]:
            for tweet_id, timestamp in self.user_tweets[followee_id]:
                candidates.append((timestamp, tweet_id))
        
        # Sort and return top 10
        candidates.sort(reverse=True)
        return [tweet_id for _, tweet_id in candidates[:10]]
    
    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.follows[followerId].add(followeeId)
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.follows[followerId].discard(followeeId)


def test_twitter_basic():
    """Test basic Twitter functionality"""
    print("=== Testing Basic Twitter Functionality ===")
    
    implementations = [
        ("Basic", TwitterBasic),
        ("Optimized", TwitterOptimized),
        ("Advanced", TwitterAdvanced),
        ("Memory Optimized", TwitterMemoryOptimized)
    ]
    
    for name, TwitterClass in implementations:
        print(f"\n{name}:")
        
        twitter = TwitterClass()
        
        # Test sequence from problem
        twitter.postTweet(1, 5)
        feed = twitter.getNewsFeed(1)
        print(f"  User 1 posts tweet 5, feed: {feed}")
        
        twitter.follow(1, 2)
        twitter.postTweet(2, 6)
        feed = twitter.getNewsFeed(1)
        print(f"  User 1 follows 2, user 2 posts 6, feed: {feed}")
        
        twitter.unfollow(1, 2)
        feed = twitter.getNewsFeed(1)
        print(f"  User 1 unfollows 2, feed: {feed}")

def test_twitter_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Twitter Edge Cases ===")
    
    twitter = TwitterAdvanced()
    
    # Test empty feed
    print("Empty feed:")
    feed = twitter.getNewsFeed(999)
    print(f"  New user feed: {feed}")
    
    # Test self-follow
    print(f"\nSelf-follow test:")
    twitter.follow(1, 1)  # Should be ignored
    twitter.postTweet(1, 100)
    feed = twitter.getNewsFeed(1)
    print(f"  User 1 self-follow and post: {feed}")
    
    # Test multiple tweets from same user
    print(f"\nMultiple tweets:")
    for i in range(15):
        twitter.postTweet(2, 200 + i)
    
    feed = twitter.getNewsFeed(2)
    print(f"  User 2 posts 15 tweets, feed shows: {len(feed)} tweets")
    
    # Test unfollow non-existing
    print(f"\nUnfollow non-existing:")
    twitter.unfollow(1, 999)  # Should not crash
    print("  Unfollow completed successfully")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    twitter = TwitterAdvanced()
    
    # Create some users and activity
    twitter.postTweet(1, 1001)
    twitter.postTweet(2, 1002)
    twitter.follow(1, 2)
    twitter.follow(3, 1)
    twitter.postTweet(1, 1003)
    
    # Test user statistics
    stats1 = twitter.getUserStats(1)
    print(f"User 1 stats: {stats1}")
    
    stats2 = twitter.getUserStats(2)
    print(f"User 2 stats: {stats2}")
    
    # Test tweet info
    tweet_info = twitter.getTweetInfo(1001)
    print(f"Tweet 1001 info: {tweet_info}")
    
    # Test followers/following
    followers = twitter.getFollowers(1)
    following = twitter.getFollowing(1)
    print(f"User 1 - Followers: {followers}, Following: {following}")
    
    # Test activity log
    activity = twitter.getUserActivity(1)
    print(f"User 1 recent activity: {[act[:2] for act in activity]}")  # Show action and target

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Social media platform
    print("Application 1: Social Media Platform")
    
    social_platform = TwitterAdvanced()
    
    # Simulate user interactions
    users = [1, 2, 3, 4, 5]
    
    # Create follow relationships
    social_platform.follow(1, 2)
    social_platform.follow(1, 3)
    social_platform.follow(2, 3)
    social_platform.follow(2, 4)
    social_platform.follow(3, 4)
    social_platform.follow(3, 5)
    
    print("  Follow network created")
    
    # Post tweets
    tweets = [
        (1, "Hello world!"),
        (2, "Good morning!"),
        (3, "Check out this cool feature"),
        (4, "Happy Friday!"),
        (5, "Weekend plans?"),
        (1, "Follow-up tweet"),
        (2, "Breaking news!")
    ]
    
    for i, (user, content) in enumerate(tweets):
        tweet_id = 2000 + i
        social_platform.postTweet(user, tweet_id)
        print(f"    User {user} posted tweet {tweet_id}: '{content}'")
    
    # Show feeds for different users
    for user in [1, 2, 3]:
        feed = social_platform.getNewsFeed(user)
        stats = social_platform.getUserStats(user)
        print(f"  User {user} feed: {feed} (following {stats['following_count']} users)")
    
    # Application 2: News aggregation service
    print(f"\nApplication 2: News Aggregation Service")
    
    news_service = TwitterOptimized()
    
    # Simulate news sources
    news_sources = {
        101: "TechNews",
        102: "SportNews", 
        103: "WorldNews",
        104: "LocalNews"
    }
    
    # User subscribes to news sources
    user_id = 500
    for source_id in news_sources:
        news_service.follow(user_id, source_id)
    
    print(f"  User {user_id} subscribed to {len(news_sources)} news sources")
    
    # News sources post updates
    news_updates = [
        (101, "Latest tech breakthrough announced"),
        (102, "Championship game results"),
        (103, "International summit concludes"),
        (101, "New smartphone released"),
        (104, "Local election results"),
        (102, "Trade deadline moves"),
        (103, "Climate conference update")
    ]
    
    for i, (source, headline) in enumerate(news_updates):
        article_id = 3000 + i
        news_service.postTweet(source, article_id)
        print(f"    {news_sources[source]} published: {headline} (ID: {article_id})")
    
    # User gets personalized news feed
    news_feed = news_service.getNewsFeed(user_id)
    print(f"  User's personalized news feed: {news_feed}")
    
    # Application 3: Corporate communication platform
    print(f"\nApplication 3: Corporate Communication Platform")
    
    corp_platform = TwitterMemoryOptimized(max_tweets_per_user=5)
    
    # Simulate departments and employees
    departments = {
        'engineering': [1001, 1002, 1003],
        'marketing': [2001, 2002],
        'sales': [3001, 3002, 3003, 3004]
    }
    
    # Set up department follows
    for dept_employees in departments.values():
        for emp1 in dept_employees:
            for emp2 in dept_employees:
                if emp1 != emp2:
                    corp_platform.follow(emp1, emp2)
    
    print("  Department communication networks established")
    
    # Cross-department follows (managers)
    managers = [1001, 2001, 3001]  # One from each department
    for i, mgr1 in enumerate(managers):
        for j, mgr2 in enumerate(managers):
            if i != j:
                corp_platform.follow(mgr1, mgr2)
    
    # Post department updates
    updates = [
        (1001, "Engineering sprint planning complete"),
        (2001, "Q4 marketing campaign launched"),
        (3001, "Sales targets exceeded!"),
        (1002, "New feature deployment successful"),
        (2002, "Customer survey results in"),
        (3002, "New client onboarded")
    ]
    
    for employee, message in updates:
        msg_id = hash(message) % 10000
        corp_platform.postTweet(employee, msg_id)
        
        # Find department
        dept = next(d for d, emps in departments.items() if employee in emps)
        print(f"    {dept.title()} (Employee {employee}): {message}")
    
    # Show feeds for managers
    for manager in managers:
        feed = corp_platform.getNewsFeed(manager)
        dept = next(d for d, emps in departments.items() if manager in emps)
        print(f"  {dept.title()} manager feed: {feed}")

def test_performance():
    """Test performance with large datasets"""
    print("\n=== Testing Performance ===")
    
    import time
    
    implementations = [
        ("Basic", TwitterBasic),
        ("Optimized", TwitterOptimized),
        ("Memory Optimized", TwitterMemoryOptimized)
    ]
    
    for name, TwitterClass in implementations:
        twitter = TwitterClass()
        
        # Setup: Create users and follows
        num_users = 100
        tweets_per_user = 50
        
        start_time = time.time()
        
        # Create follow relationships
        for i in range(1, num_users + 1):
            for j in range(max(1, i - 5), min(num_users + 1, i + 5)):
                if i != j:
                    twitter.follow(i, j)
        
        setup_time = (time.time() - start_time) * 1000
        
        # Post tweets
        start_time = time.time()
        
        for user in range(1, num_users + 1):
            for tweet in range(tweets_per_user):
                tweet_id = user * 1000 + tweet
                twitter.postTweet(user, tweet_id)
        
        post_time = (time.time() - start_time) * 1000
        
        # Test news feed generation
        start_time = time.time()
        
        for user in range(1, min(21, num_users + 1)):  # Test 20 users
            twitter.getNewsFeed(user)
        
        feed_time = (time.time() - start_time) * 1000
        
        total_tweets = num_users * tweets_per_user
        
        print(f"  {name}:")
        print(f"    Setup: {setup_time:.2f}ms")
        print(f"    Post {total_tweets} tweets: {post_time:.2f}ms")
        print(f"    Generate 20 feeds: {feed_time:.2f}ms")

def stress_test_twitter():
    """Stress test Twitter implementation"""
    print("\n=== Stress Testing Twitter ===")
    
    import time
    import random
    
    twitter = TwitterOptimized()
    
    # Large scale test
    num_users = 1000
    num_operations = 5000
    
    print(f"Stress test: {num_users} users, {num_operations} operations")
    
    start_time = time.time()
    
    operations = ['post', 'follow', 'unfollow', 'feed']
    operation_counts = {op: 0 for op in operations}
    
    for i in range(num_operations):
        operation = random.choices(operations, weights=[0.4, 0.2, 0.1, 0.3])[0]
        operation_counts[operation] += 1
        
        if operation == 'post':
            user = random.randint(1, num_users)
            tweet = random.randint(10000, 99999)
            twitter.postTweet(user, tweet)
        
        elif operation == 'follow':
            follower = random.randint(1, num_users)
            followee = random.randint(1, num_users)
            twitter.follow(follower, followee)
        
        elif operation == 'unfollow':
            follower = random.randint(1, num_users)
            followee = random.randint(1, num_users)
            twitter.unfollow(follower, followee)
        
        elif operation == 'feed':
            user = random.randint(1, num_users)
            twitter.getNewsFeed(user)
        
        # Progress update
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"    {i + 1} operations, {rate:.0f} ops/sec")
    
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Rate: {num_operations / total_time:.0f} operations/sec")
    print(f"  Operation breakdown: {operation_counts}")

def test_news_feed_quality():
    """Test news feed quality and ordering"""
    print("\n=== Testing News Feed Quality ===")
    
    twitter = TwitterAdvanced()
    
    # Create test scenario
    users = [1, 2, 3, 4]
    
    # User 1 follows users 2, 3, 4
    for user in [2, 3, 4]:
        twitter.follow(1, user)
    
    # Post tweets in specific order
    tweet_sequence = [
        (2, 2001, "Tweet from user 2 - oldest"),
        (1, 1001, "Tweet from user 1 - own tweet"),
        (3, 3001, "Tweet from user 3 - middle"),
        (4, 4001, "Tweet from user 4 - newest"),
        (2, 2002, "Tweet from user 2 - second")
    ]
    
    print("Posting tweets in sequence:")
    for user, tweet_id, content in tweet_sequence:
        twitter.postTweet(user, tweet_id)
        print(f"  User {user}: {content}")
    
    # Check feed ordering
    feed = twitter.getNewsFeed(1)
    print(f"\nUser 1's news feed: {feed}")
    print("Expected order: [2002, 4001, 3001, 1001, 2001] (newest first)")
    
    # Test with more tweets than feed limit
    print(f"\nTesting feed limit:")
    
    # Post 15 more tweets
    for i in range(15):
        twitter.postTweet(2, 3000 + i)
    
    extended_feed = twitter.getNewsFeed(1)
    print(f"Feed after 15 more tweets: {len(extended_feed)} items (max 10)")
    print(f"Most recent tweets: {extended_feed[:5]}")

def benchmark_feed_generation():
    """Benchmark feed generation with different follow counts"""
    print("\n=== Benchmarking Feed Generation ===")
    
    import time
    
    follow_counts = [1, 5, 10, 25, 50, 100]
    
    for follow_count in follow_counts:
        twitter = TwitterOptimized()
        
        # Setup: User 1 follows 'follow_count' users
        for i in range(2, 2 + follow_count):
            twitter.follow(1, i)
        
        # Each followed user posts 10 tweets
        for user in range(2, 2 + follow_count):
            for tweet in range(10):
                twitter.postTweet(user, user * 100 + tweet)
        
        # User 1 posts some tweets too
        for tweet in range(10):
            twitter.postTweet(1, 1000 + tweet)
        
        # Benchmark feed generation
        start_time = time.time()
        
        num_tests = 100
        for _ in range(num_tests):
            twitter.getNewsFeed(1)
        
        elapsed = (time.time() - start_time) * 1000
        avg_time = elapsed / num_tests
        
        total_tweets = follow_count * 10 + 10
        print(f"  {follow_count} follows ({total_tweets} tweets): {avg_time:.3f}ms per feed")

if __name__ == "__main__":
    test_twitter_basic()
    test_twitter_edge_cases()
    test_advanced_features()
    demonstrate_applications()
    test_performance()
    stress_test_twitter()
    test_news_feed_quality()
    benchmark_feed_generation()

"""
Twitter Design demonstrates key concepts:

Core Approaches:
1. Basic - Simple linear search through all tweets
2. Optimized - Per-user tweet storage with heap-based merging
3. Advanced - Enhanced with analytics, user management, and features
4. Memory Optimized - Limited storage per user for large-scale systems

Key Design Principles:
- Social graph management (follows/followers)
- Timeline generation and chronological ordering
- Efficient news feed algorithms
- User activity tracking and analytics

Performance Characteristics:
- Basic: O(n*m) feed generation, simple but inefficient
- Optimized: O(f*log(f)) feed generation, optimal for most cases
- Advanced: Additional overhead for features and analytics
- Memory Optimized: Bounded space usage per user

Real-world Applications:
- Social media platforms (Twitter, Facebook, Instagram)
- News aggregation and personalization services
- Corporate communication and collaboration tools
- Content distribution and recommendation systems
- Real-time messaging and notification systems
- Professional networking platforms

The optimized approach with per-user tweet storage and heap-based
merging provides the best balance of performance and functionality
for real-time social media applications.
"""
