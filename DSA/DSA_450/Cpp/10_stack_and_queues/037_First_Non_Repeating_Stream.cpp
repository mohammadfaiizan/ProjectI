/*
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
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string First_Non_Repeating_Stream_Queue(string stream) {
        /*
        Queue + frequency array
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        queue<char> q;
        vector<int> freq(26, 0);
        string result = "";
        
        for (char c : stream) {
            freq[c - 'a']++;
            q.push(c);
            
            while (!q.empty() && freq[q.front() - 'a'] > 1) {
                q.pop();
            }
            
            if (q.empty()) {
                result += "-1 ";
            } else {
                result += string(1, q.front()) + " ";
            }
        }
        
        return result;
    }
};

void Test_First_Non_Repeating_Stream() {
    Solution solution;
    
    string stream1 = "aabc";
    cout << "Test 1 - Queue: " << solution.First_Non_Repeating_Stream_Queue(stream1) << endl;
    
    string stream2 = "aabcbc";
    cout << "Test 2 - Queue: " << solution.First_Non_Repeating_Stream_Queue(stream2) << endl;
    
    string stream3 = "zz";
    cout << "Test 3 - Queue: " << solution.First_Non_Repeating_Stream_Queue(stream3) << endl;
    
    string stream4 = "abcde";
    cout << "Test 4 - Queue: " << solution.First_Non_Repeating_Stream_Queue(stream4) << endl;
}

int main() {
    Test_First_Non_Repeating_Stream();
    return 0;
}
