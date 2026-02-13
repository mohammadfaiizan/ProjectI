# Medium Hashing Problems

## 1. Three Sum

**Description**: Find all unique triplets that sum to zero.

**Approach**: Sort, fix one element, two pointers or hash for remaining pair. Skip duplicates.

---

## 2. Group Anagrams

**Description**: Group strings that are anagrams of each other.

**Approach**: Use sorted string or character count tuple as hash key.

---

## 3. Longest Substring Without Repeating Characters

**Description**: Find longest substring with all unique characters.

**Approach**: Sliding window with set or dict storing char to index.

---

## 4. Subarray Sum Equals K

**Description**: Count contiguous subarrays with sum k.

**Approach**: Prefix sum hash; prefix[j]-prefix[i]=k means prefix[i]=prefix[j]-k.

---

## 5. Top K Frequent Elements

**Description**: Return k most frequent elements.

**Approach**: Counter for frequencies, then bucket sort or heap.

---

## 6. Longest Consecutive Sequence

**Description**: Longest consecutive integer sequence length.

**Approach**: Put all in set; for each potential sequence start (x-1 not in set), count forward.

---

## 7. Contiguous Array

**Description**: Longest subarray with equal 0s and 1s.

**Approach**: Treat 0 as -1; prefix sum hash for sum 0.

---

## 8. Design Underground System

**Description**: Track check-in/out and compute average time between stations.

**Approach**: Hash check-in by id; hash (start, end) to (total_time, count).

---

## 9. LRU Cache

**Description**: Cache with get/put in O(1), evict least recently used when full.

**Approach**: Hash map + doubly linked list for order.

---

## 10. Encode and Decode TinyURL

**Description**: Shorten and expand URLs.

**Approach**: Hash long URL to short code; store bidirectional mapping.

---

## 11. Insert Delete GetRandom O(1)

**Description**: Data structure with insert, remove, getRandom all O(1).

**Approach**: List + dict; on remove swap with last and pop.

---

## 12. Time Based Key-Value Store

**Description**: Store multiple versions per key; get(key, timestamp) returns value at or before timestamp.

**Approach**: Dict of key to list of (timestamp, value); binary search for floor.

---

## 13. Snapshot Array

**Description**: Array with set, snap, and get(index, snap_id).

**Approach**: Each index stores list of (snap_id, value); binary search for snap_id.

---

## 14. 4Sum

**Description**: Find all unique quadruplets with sum target.

**Approach**: Sort, fix two elements, two pointers or hash for remaining pair.

---

## 15. 4Sum II

**Description**: Four arrays; count tuples (i,j,k,l) with A[i]+B[j]+C[k]+D[l]=0.

**Approach**: Hash sums of A+B; for each sum of C+D, count -sum in hash.

---

## 16. Sort Characters by Frequency

**Description**: Sort string by character frequency descending.

**Approach**: Counter, then most_common or sort by count.

---

## 17. Find All Duplicates in an Array

**Description**: Array 1..n, some appear twice; return all duplicates.

**Approach**: Mark indices by negating; already negative means duplicate.

---

## 18. Contains Duplicate III

**Description**: Check if |nums[i]-nums[j]| <= t and |i-j| <= k.

**Approach**: Bucket hash or sliding window with sorted container.

---

## 19. Majority Element II

**Description**: Find elements appearing more than n/3 times.

**Approach**: Extended Boyer-Moore; at most two candidates.

---

## 20. Longest Harmonious Subsequence

**Description**: Longest subsequence where max-min=1.

**Approach**: Count frequencies; for each x, add count(x)+count(x+1) if both exist.

---

## 21. Brick Wall

**Description**: Least bricks to cross (vertical line).

**Approach**: Hash gap positions; max gaps across rows = least bricks.

---

## 22. Number of Boomerangs

**Description**: Triples (i,j,k) where dist(i,j)=dist(i,k).

**Approach**: For each point, count distances; each count c adds c*(c-1).

---

## 23. Find the Duplicate Number

**Description**: Array 1..n with one duplicate; find it in O(1) space.

**Approach**: Floyd cycle detection (linked list) or binary search on count.

---

## 24. Subarray Sum Divisible by K

**Description**: Count subarrays with sum divisible by k.

**Approach**: Prefix sum mod k; same remainder means divisible subarray.

---

## 25. Maximum Size Subarray Sum Equals K

**Description**: Longest subarray with sum k.

**Approach**: Prefix sum hash; store first occurrence of each prefix for longest.

---

## 26. Copy List with Random Pointer

**Description**: Deep copy linked list with random pointer.

**Approach**: Hash map old node to new node; two passes.

---

## 27. Reconstruct Original Digits from English

**Description**: Given string with jumbled digits in words, return digits in order.

**Approach**: Count unique chars; some digits have unique letters (e.g., z only in zero).

---

## 28. Design Authentication Manager

**Description**: Token with expiry; count unexpired tokens.

**Approach**: Hash token to expiry time; on count, filter by current time.

---

## 29. Insert Delete GetRandom O(1) with Duplicates

**Description**: Same as RandomizedSet but allow duplicates.

**Approach**: Dict of value to set of indices; list for storage; swap with last on remove.

---

## 30. LFU Cache

**Description**: Cache evicting least frequently used.

**Approach**: Hash key to node; freq to doubly linked list; min_freq tracker.
