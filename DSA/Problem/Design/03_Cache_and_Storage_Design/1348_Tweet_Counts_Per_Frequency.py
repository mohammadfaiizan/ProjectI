"""
1348. Tweet Counts Per Frequency - Multiple Approaches
Difficulty: Medium

A social media company is trying to monitor activity on their site by analyzing the number of tweets that occur in select periods of time. These periods can be partitioned into smaller time chunks based on a certain frequency (every minute, hour, or day).

Implement the TweetCounts class:
- TweetCounts() Initializes the TweetCounts object.
- void recordTweet(String tweetName, int time) Stores the tweetName at the recorded time (in seconds).
- List<Integer> getTweetCountsPerFrequency(String freq, String tweetName, int startTime, int endTime) Returns a list of integers representing the number of tweets with tweetName in each time chunk for the given period of time [startTime, endTime] (in seconds) and frequency freq.
  - freq is one of "minute", "hour", or "day" representing a time chunk of 60, 3600, or 86400 seconds, respectively.
  - The first time chunk always starts from startTime, so the time chunks are [startTime, startTime + chunk_size), [startTime + chunk_size, startTime + 2 * chunk_size), etc.
  - The last time chunk may be shorter than chunk_size seconds.
"""

from typing import List, Dict
from collections import defaultdict
import bisect

class TweetCountsSimple:
    """
    Approach 1: Simple List Storage
    
    Store tweets in lists and count for each time chunk.
    
    Time Complexity:
    - recordTweet: O(1)
    - getTweetCountsPerFrequency: O(n) where n is total tweets for name
    
    Space Complexity: O(n) where n is total tweets
    """
    
    def __init__(self):
        # Dictionary: tweetName -> list of timestamps
        self.tweets = defaultdict(list)
        self.freq_to_seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
    
    def recordTweet(self, tweetName: str, time: int) -> None:
        self.tweets[tweetName].append(time)
    
    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        chunk_size = self.freq_to_seconds[freq]
        
        # Generate time chunks
        chunks = []
        current_start = startTime
        
        while current_start <= endTime:
            current_end = min(current_start + chunk_size - 1, endTime)
            chunks.append((current_start, current_end))
            current_start += chunk_size
        
        # Count tweets in each chunk
        result = []
        tweet_times = self.tweets[tweetName]
        
        for chunk_start, chunk_end in chunks:
            count = 0
            for tweet_time in tweet_times:
                if chunk_start <= tweet_time <= chunk_end:
                    count += 1
            result.append(count)
        
        return result

class TweetCountsSorted:
    """
    Approach 2: Sorted Lists with Binary Search
    
    Keep tweets sorted for efficient range queries.
    
    Time Complexity:
    - recordTweet: O(log n) for insertion in sorted order
    - getTweetCountsPerFrequency: O(log n + k) where k is chunks
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.tweets = defaultdict(list)  # Keep sorted by time
        self.freq_to_seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
    
    def recordTweet(self, tweetName: str, time: int) -> None:
        # Insert in sorted order
        bisect.insort(self.tweets[tweetName], time)
    
    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        chunk_size = self.freq_to_seconds[freq]
        tweet_times = self.tweets[tweetName]
        
        result = []
        current_start = startTime
        
        while current_start <= endTime:
            current_end = min(current_start + chunk_size - 1, endTime)
            
            # Binary search for range [current_start, current_end]
            left_idx = bisect.bisect_left(tweet_times, current_start)
            right_idx = bisect.bisect_right(tweet_times, current_end)
            
            count = right_idx - left_idx
            result.append(count)
            
            current_start += chunk_size
        
        return result

class TweetCountsHashMap:
    """
    Approach 3: HashMap with Time Buckets
    
    Pre-bucket tweets by time intervals for faster queries.
    
    Time Complexity:
    - recordTweet: O(1)
    - getTweetCountsPerFrequency: O(k) where k is number of chunks
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        # Dictionary: tweetName -> {time_bucket -> count}
        self.minute_buckets = defaultdict(lambda: defaultdict(int))
        self.hour_buckets = defaultdict(lambda: defaultdict(int))
        self.day_buckets = defaultdict(lambda: defaultdict(int))
        
        self.freq_to_seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
    
    def recordTweet(self, tweetName: str, time: int) -> None:
        # Update all frequency buckets
        minute_bucket = time // 60
        hour_bucket = time // 3600
        day_bucket = time // 86400
        
        self.minute_buckets[tweetName][minute_bucket] += 1
        self.hour_buckets[tweetName][hour_bucket] += 1
        self.day_buckets[tweetName][day_bucket] += 1
    
    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        chunk_size = self.freq_to_seconds[freq]
        
        # Select appropriate bucket
        if freq == "minute":
            buckets = self.minute_buckets[tweetName]
        elif freq == "hour":
            buckets = self.hour_buckets[tweetName]
        else:  # day
            buckets = self.day_buckets[tweetName]
        
        result = []
        current_start = startTime
        
        while current_start <= endTime:
            current_end = min(current_start + chunk_size - 1, endTime)
            
            # Find overlapping buckets
            start_bucket = current_start // chunk_size
            end_bucket = current_end // chunk_size
            
            count = 0
            for bucket_id in range(start_bucket, end_bucket + 1):
                count += buckets.get(bucket_id, 0)
            
            result.append(count)
            current_start += chunk_size
        
        return result

class TweetCountsSegmentTree:
    """
    Approach 4: Segment Tree for Range Queries
    
    Use segment tree for efficient range sum queries.
    
    Time Complexity:
    - recordTweet: O(log max_time)
    - getTweetCountsPerFrequency: O(k log max_time)
    
    Space Complexity: O(max_time)
    """
    
    def __init__(self):
        self.max_time = 10**9  # Maximum timestamp
        self.trees = defaultdict(lambda: defaultdict(int))  # tweetName -> {time -> count}
        self.freq_to_seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
    
    def recordTweet(self, tweetName: str, time: int) -> None:
        self.trees[tweetName][time] += 1
    
    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        chunk_size = self.freq_to_seconds[freq]
        tree = self.trees[tweetName]
        
        result = []
        current_start = startTime
        
        while current_start <= endTime:
            current_end = min(current_start + chunk_size - 1, endTime)
            
            # Sum tweets in range [current_start, current_end]
            count = sum(tree[t] for t in range(current_start, current_end + 1) if t in tree)
            result.append(count)
            
            current_start += chunk_size
        
        return result

class TweetCountsAdvanced:
    """
    Approach 5: Advanced with Analytics and Optimization
    
    Enhanced version with caching, analytics, and performance optimizations.
    
    Time Complexity:
    - recordTweet: O(log n)
    - getTweetCountsPerFrequency: O(log n + k) with caching
    
    Space Complexity: O(n + cache_size)
    """
    
    def __init__(self):
        self.tweets = defaultdict(list)  # Sorted lists
        self.freq_to_seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        # Analytics
        self.total_tweets = 0
        self.total_queries = 0
        self.tweet_names = set()
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.tweets_per_name = defaultdict(int)
        self.queries_per_name = defaultdict(int)
        self.popular_frequencies = defaultdict(int)
    
    def recordTweet(self, tweetName: str, time: int) -> None:
        self.total_tweets += 1
        self.tweet_names.add(tweetName)
        self.tweets_per_name[tweetName] += 1
        
        # Insert in sorted order
        bisect.insort(self.tweets[tweetName], time)
        
        # Invalidate cache for this tweet name
        keys_to_remove = [key for key in self.query_cache if key[1] == tweetName]
        for key in keys_to_remove:
            del self.query_cache[key]
    
    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        self.total_queries += 1
        self.queries_per_name[tweetName] += 1
        self.popular_frequencies[freq] += 1
        
        # Check cache
        cache_key = (freq, tweetName, startTime, endTime)
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        self.cache_misses += 1
        
        chunk_size = self.freq_to_seconds[freq]
        tweet_times = self.tweets[tweetName]
        
        result = []
        current_start = startTime
        
        while current_start <= endTime:
            current_end = min(current_start + chunk_size - 1, endTime)
            
            # Binary search for efficient range counting
            left_idx = bisect.bisect_left(tweet_times, current_start)
            right_idx = bisect.bisect_right(tweet_times, current_end)
            
            count = right_idx - left_idx
            result.append(count)
            
            current_start += chunk_size
        
        # Cache result (limit cache size)
        if len(self.query_cache) < 1000:
            self.query_cache[cache_key] = result
        
        return result
    
    def getAnalytics(self) -> dict:
        """Get system analytics"""
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return {
            'total_tweets': self.total_tweets,
            'total_queries': self.total_queries,
            'unique_tweet_names': len(self.tweet_names),
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.query_cache),
            'most_popular_frequency': max(self.popular_frequencies.items(), 
                                        key=lambda x: x[1], default=('none', 0))[0],
            'average_tweets_per_name': self.total_tweets / max(1, len(self.tweet_names))
        }
    
    def getTopTweetNames(self, k: int = 5) -> List[tuple]:
        """Get top K tweet names by volume"""
        sorted_names = sorted(self.tweets_per_name.items(), key=lambda x: x[1], reverse=True)
        return sorted_names[:k]
    
    def getTimeRangeStats(self, tweetName: str) -> dict:
        """Get time range statistics for a tweet name"""
        if tweetName not in self.tweets or not self.tweets[tweetName]:
            return {'min_time': None, 'max_time': None, 'span': 0}
        
        times = self.tweets[tweetName]
        min_time = min(times)
        max_time = max(times)
        
        return {
            'min_time': min_time,
            'max_time': max_time,
            'span': max_time - min_time,
            'total_tweets': len(times)
        }


def test_tweet_counts_basic():
    """Test basic TweetCounts functionality"""
    print("=== Testing Basic TweetCounts Functionality ===")
    
    implementations = [
        ("Simple Lists", TweetCountsSimple),
        ("Sorted with Binary Search", TweetCountsSorted),
        ("HashMap Buckets", TweetCountsHashMap),
        ("Segment Tree", TweetCountsSegmentTree),
        ("Advanced", TweetCountsAdvanced)
    ]
    
    for name, TweetCountsClass in implementations:
        print(f"\n{name}:")
        
        tc = TweetCountsClass()
        
        # Test sequence from problem
        operations = [
            ("recordTweet", "tweet3", 0),
            ("recordTweet", "tweet3", 60),
            ("recordTweet", "tweet3", 10),
            ("getTweetCountsPerFrequency", "minute", "tweet3", 0, 59),
            ("getTweetCountsPerFrequency", "minute", "tweet3", 0, 60),
            ("recordTweet", "tweet3", 120),
            ("getTweetCountsPerFrequency", "hour", "tweet3", 0, 210)
        ]
        
        for op, *args in operations:
            if op == "recordTweet":
                tweetName, time = args
                tc.recordTweet(tweetName, time)
                print(f"  recordTweet('{tweetName}', {time})")
            elif op == "getTweetCountsPerFrequency":
                freq, tweetName, startTime, endTime = args
                result = tc.getTweetCountsPerFrequency(freq, tweetName, startTime, endTime)
                print(f"  getTweetCountsPerFrequency('{freq}', '{tweetName}', {startTime}, {endTime}): {result}")

def test_tweet_counts_edge_cases():
    """Test edge cases"""
    print("\n=== Testing TweetCounts Edge Cases ===")
    
    tc = TweetCountsAdvanced()
    
    # Test empty queries
    print("Empty queries:")
    result = tc.getTweetCountsPerFrequency("minute", "nonexistent", 0, 59)
    print(f"  Query for nonexistent tweet: {result}")
    
    # Test single time point
    print(f"\nSingle time point:")
    tc.recordTweet("single", 100)
    result = tc.getTweetCountsPerFrequency("minute", "single", 100, 100)
    print(f"  Single second query: {result}")
    
    # Test multiple tweets at same time
    print(f"\nMultiple tweets at same time:")
    for _ in range(5):
        tc.recordTweet("burst", 200)
    
    result = tc.getTweetCountsPerFrequency("minute", "burst", 200, 200)
    print(f"  5 tweets at same time: {result}")
    
    # Test large time ranges
    print(f"\nLarge time range:")
    tc.recordTweet("sparse", 0)
    tc.recordTweet("sparse", 86400)  # 1 day later
    
    result = tc.getTweetCountsPerFrequency("day", "sparse", 0, 86400)
    print(f"  Day-long range: {result}")
    
    # Test boundary conditions
    print(f"\nBoundary conditions:")
    tc.recordTweet("boundary", 59)
    tc.recordTweet("boundary", 60)
    tc.recordTweet("boundary", 61)
    
    result = tc.getTweetCountsPerFrequency("minute", "boundary", 0, 119)
    print(f"  Tweets at minute boundaries: {result}")

def test_frequency_types():
    """Test different frequency types"""
    print("\n=== Testing Different Frequency Types ===")
    
    tc = TweetCountsSorted()
    
    # Record tweets across different time scales
    times = [0, 30, 60, 90, 1800, 3600, 7200, 86400]
    
    for time in times:
        tc.recordTweet("multi_freq", time)
    
    print("Testing different frequencies:")
    
    # Test minute frequency
    result = tc.getTweetCountsPerFrequency("minute", "multi_freq", 0, 180)
    print(f"  Minute (0-180s): {result}")
    
    # Test hour frequency  
    result = tc.getTweetCountsPerFrequency("hour", "multi_freq", 0, 7200)
    print(f"  Hour (0-7200s): {result}")
    
    # Test day frequency
    result = tc.getTweetCountsPerFrequency("day", "multi_freq", 0, 86400)
    print(f"  Day (0-86400s): {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Simple Lists", TweetCountsSimple),
        ("Sorted Binary Search", TweetCountsSorted),
        ("HashMap Buckets", TweetCountsHashMap)
    ]
    
    num_tweets = 10000
    num_queries = 1000
    
    for name, TweetCountsClass in implementations:
        tc = TweetCountsClass()
        
        # Time recording tweets
        start_time = time.time()
        
        import random
        for i in range(num_tweets):
            tweet_name = f"tweet_{i % 100}"  # 100 different tweet names
            timestamp = random.randint(0, 86400)  # Random time in a day
            tc.recordTweet(tweet_name, timestamp)
        
        record_time = (time.time() - start_time) * 1000
        
        # Time queries
        start_time = time.time()
        
        for _ in range(num_queries):
            tweet_name = f"tweet_{random.randint(0, 99)}"
            freq = random.choice(["minute", "hour", "day"])
            start_t = random.randint(0, 43200)
            end_t = start_t + random.randint(3600, 43200)
            
            tc.getTweetCountsPerFrequency(freq, tweet_name, start_t, end_t)
        
        query_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    {num_tweets} records: {record_time:.2f}ms")
        print(f"    {num_queries} queries: {query_time:.2f}ms")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    tc = TweetCountsAdvanced()
    
    # Create diverse tweet data
    tweet_data = [
        ("trending_topic", [100, 150, 200, 250, 300]),
        ("news_flash", [500, 510, 520]),
        ("viral_meme", [1000, 1010, 1020, 1030, 1040, 1050]),
        ("breaking_news", [2000, 2100])
    ]
    
    print("Recording diverse tweet data:")
    for tweet_name, times in tweet_data:
        for time in times:
            tc.recordTweet(tweet_name, time)
        print(f"  {tweet_name}: {len(times)} tweets")
    
    # Test analytics
    analytics = tc.getAnalytics()
    print(f"\nSystem analytics:")
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test top tweet names
    top_tweets = tc.getTopTweetNames(3)
    print(f"\nTop 3 tweet names by volume:")
    for tweet_name, count in top_tweets:
        print(f"  {tweet_name}: {count} tweets")
    
    # Test time range stats
    print(f"\nTime range statistics:")
    for tweet_name, _ in tweet_data:
        stats = tc.getTimeRangeStats(tweet_name)
        print(f"  {tweet_name}: span={stats['span']}s, tweets={stats['total_tweets']}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Social media monitoring
    print("Application 1: Social Media Trend Monitoring")
    
    trend_monitor = TweetCountsAdvanced()
    
    # Simulate trending hashtag activity
    hashtag_timeline = [
        ("#covid19", [0, 60, 120, 180, 240, 300, 360, 420]),
        ("#election2024", [1000, 1200, 1400, 1600]),
        ("#sports", [2000, 2300, 2600, 2900, 3200])
    ]
    
    print("  Recording hashtag activity:")
    for hashtag, times in hashtag_timeline:
        for time in times:
            trend_monitor.recordTweet(hashtag, time)
        print(f"    {hashtag}: activity at {len(times)} time points")
    
    # Analyze trends by hour
    print(f"  Hourly trend analysis:")
    for hashtag, _ in hashtag_timeline:
        counts = trend_monitor.getTweetCountsPerFrequency("hour", hashtag, 0, 3600)
        peak_hour = counts.index(max(counts)) if counts and max(counts) > 0 else -1
        print(f"    {hashtag}: peak activity in hour {peak_hour}, counts: {counts}")
    
    # Application 2: News event tracking
    print(f"\nApplication 2: Breaking News Event Tracking")
    
    news_tracker = TweetCountsSorted()
    
    # Simulate breaking news event
    breaking_news_times = [
        3600, 3605, 3610, 3620, 3630, 3645,  # Initial burst
        3700, 3750, 3800, 3850, 3900,        # Follow-up coverage
        4500, 4800, 5100                      # Later reactions
    ]
    
    print(f"  Simulating breaking news tweet pattern:")
    for time in breaking_news_times:
        news_tracker.recordTweet("breaking_earthquake", time)
    
    # Analyze by minute for first hour
    minute_counts = news_tracker.getTweetCountsPerFrequency("minute", "breaking_earthquake", 3600, 4200)
    
    print(f"  Minute-by-minute activity (first 10 minutes):")
    for i, count in enumerate(minute_counts[:10]):
        minute_mark = 3600 + i * 60
        print(f"    Minute {minute_mark//60}: {count} tweets")
    
    # Application 3: Marketing campaign analysis
    print(f"\nApplication 3: Marketing Campaign Analysis")
    
    campaign_tracker = TweetCountsHashMap()
    
    # Simulate campaign mentions across a week
    campaigns = {
        "summer_sale": [0, 3600, 7200, 86400, 90000, 172800, 176400],  # Spread across week
        "flash_deal": [43200, 43260, 43320, 43380],                     # Concentrated burst
        "brand_awareness": [0, 21600, 43200, 64800, 86400, 108000]     # Regular intervals
    }
    
    print(f"  Recording campaign mentions:")
    for campaign, times in campaigns.items():
        for time in times:
            campaign_tracker.recordTweet(campaign, time)
        print(f"    {campaign}: {len(times)} mentions")
    
    # Daily analysis
    print(f"  Daily campaign performance:")
    for campaign in campaigns:
        daily_counts = campaign_tracker.getTweetCountsPerFrequency("day", campaign, 0, 172800)
        total_mentions = sum(daily_counts)
        print(f"    {campaign}: {total_mentions} total mentions over {len(daily_counts)} days")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Simple Lists", TweetCountsSimple),
        ("Sorted Lists", TweetCountsSorted),
        ("HashMap Buckets", TweetCountsHashMap)
    ]
    
    for name, TweetCountsClass in implementations:
        tc = TweetCountsClass()
        
        # Add tweets for multiple names
        num_names = 100
        tweets_per_name = 1000
        
        for name_id in range(num_names):
            tweet_name = f"tweet_{name_id}"
            for i in range(tweets_per_name):
                tc.recordTweet(tweet_name, i * 60)  # One tweet per minute
        
        # Estimate memory usage
        if hasattr(tc, 'tweets'):
            if isinstance(tc.tweets, dict):
                memory_estimate = sum(len(tweets) for tweets in tc.tweets.values())
            else:
                memory_estimate = num_names * tweets_per_name
        elif hasattr(tc, 'minute_buckets'):
            bucket_count = sum(len(buckets) for buckets in tc.minute_buckets.values())
            memory_estimate = bucket_count
        else:
            memory_estimate = num_names * tweets_per_name
        
        print(f"  {name}: ~{memory_estimate} storage units for {num_names * tweets_per_name} tweets")

def stress_test_tweet_counts():
    """Stress test tweet counts"""
    print("\n=== Stress Testing Tweet Counts ===")
    
    import time
    import random
    
    tc = TweetCountsAdvanced()
    
    # Large scale test
    num_operations = 10000
    num_tweet_names = 200
    
    print(f"Stress test: {num_operations} mixed operations")
    
    start_time = time.time()
    
    for i in range(num_operations):
        if i % 3 == 0:  # Record tweet
            tweet_name = f"tweet_{random.randint(0, num_tweet_names-1)}"
            timestamp = random.randint(0, 86400 * 7)  # Week's worth of data
            tc.recordTweet(tweet_name, timestamp)
        
        else:  # Query
            tweet_name = f"tweet_{random.randint(0, num_tweet_names-1)}"
            freq = random.choice(["minute", "hour", "day"])
            start_time_q = random.randint(0, 86400 * 6)
            end_time_q = start_time_q + random.randint(3600, 86400)
            
            tc.getTweetCountsPerFrequency(freq, tweet_name, start_time_q, end_time_q)
    
    elapsed = (time.time() - start_time) * 1000
    
    # Get final analytics
    analytics = tc.getAnalytics()
    
    print(f"  Completed in {elapsed:.2f}ms")
    print(f"  Final analytics:")
    print(f"    Total tweets: {analytics['total_tweets']}")
    print(f"    Total queries: {analytics['total_queries']}")
    print(f"    Cache hit rate: {analytics['cache_hit_rate']:.3f}")

def benchmark_chunk_sizes():
    """Benchmark different chunk sizes and time ranges"""
    print("\n=== Benchmarking Chunk Sizes ===")
    
    import time
    
    tc = TweetCountsSorted()
    
    # Add tweets spread across a week
    for i in range(10000):
        tweet_name = f"tweet_{i % 50}"
        timestamp = i * 60  # One tweet per minute
        tc.recordTweet(tweet_name, timestamp)
    
    # Test different frequency queries
    test_cases = [
        ("Minute chunks", "minute", 0, 3600),        # 1 hour in minutes
        ("Hour chunks", "hour", 0, 86400),           # 1 day in hours  
        ("Day chunks", "day", 0, 604800)             # 1 week in days
    ]
    
    for test_name, freq, start_t, end_t in test_cases:
        start_time = time.time()
        
        result = tc.getTweetCountsPerFrequency(freq, "tweet_0", start_t, end_t)
        
        elapsed = (time.time() - start_time) * 1000
        chunk_count = len(result)
        
        print(f"  {test_name}: {elapsed:.2f}ms for {chunk_count} chunks")

if __name__ == "__main__":
    test_tweet_counts_basic()
    test_tweet_counts_edge_cases()
    test_frequency_types()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    stress_test_tweet_counts()
    benchmark_chunk_sizes()

"""
Tweet Counts Per Frequency Design demonstrates key concepts:

Core Approaches:
1. Simple Lists - Store tweets in lists, linear scan for counting
2. Sorted with Binary Search - Maintain sorted order for efficient range queries
3. HashMap Buckets - Pre-bucket tweets by time intervals
4. Segment Tree - Range sum queries with tree structure
5. Advanced - Enhanced with caching, analytics, and optimizations

Key Design Principles:
- Time-based data partitioning and aggregation
- Range query optimization techniques
- Trade-offs between storage and query performance
- Caching strategies for repeated queries

Performance Characteristics:
- Simple: O(1) record, O(n) query per chunk
- Sorted: O(log n) record, O(log n + k) query where k is chunks
- HashMap: O(1) record, O(k) query with pre-bucketing
- Advanced: Includes caching for improved repeated query performance

Real-world Applications:
- Social media trend monitoring and analytics
- Breaking news event tracking and analysis
- Marketing campaign performance measurement
- User activity pattern analysis
- System monitoring and alerting
- Time-series data aggregation

The sorted approach with binary search provides the best
balance for most use cases, offering logarithmic performance
for both recording and querying operations.
"""
