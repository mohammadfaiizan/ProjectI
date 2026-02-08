"""
Problem: First Non-Repeating Character in a Stream
URL: https://www.geeksforgeeks.org/queue-based-approach-for-first-non-repeating-character-in-a-stream/

Problem Statement:
Given a stream of characters, find the first non-repeating character from the stream. You need to tell the first non-repeating character in O(1) time at any moment.
If a non-repeating character doesn't exist, return -1.

Sample Input/Output:
Input: stream = "aabc"
Output: "a -1 a a"
Explanation: a -> a (first non-repeating)
            aa -> -1 (no non-repeating)
            aab -> a (first non-repeating is a)
            aabc -> a (first non-repeating is a)
"""

from collections import deque


class Solution:
    def First_Non_Repeating_Stream_Queue(self, stream):
        """
        Find first non-repeating character using queue + frequency array.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        q = deque()
        freq = [0] * 26
        result = ""
        
        for c in stream:
            freq[ord(c) - ord('a')] += 1
            q.append(c)
            
            while q and freq[ord(q[0]) - ord('a')] > 1:
                q.popleft()
            
            if not q:
                result += "-1 "
            else:
                result += q[0] + " "
        
        return result


def Test_First_Non_Repeating_Stream():
    solution = Solution()
    
    stream1 = "aabc"
    print(f"Test 1 - Queue: {solution.First_Non_Repeating_Stream_Queue(stream1)}")
    
    stream2 = "aabcbc"
    print(f"Test 2 - Queue: {solution.First_Non_Repeating_Stream_Queue(stream2)}")
    
    stream3 = "zz"
    print(f"Test 3 - Queue: {solution.First_Non_Repeating_Stream_Queue(stream3)}")
    
    stream4 = "abcde"
    print(f"Test 4 - Queue: {solution.First_Non_Repeating_Stream_Queue(stream4)}")


if __name__ == "__main__":
    Test_First_Non_Repeating_Stream()
