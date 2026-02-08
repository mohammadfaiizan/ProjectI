/*
Problem: Rearrange Characters So No Two Adjacent Are Same
URL: https://www.geeksforgeeks.org/rearrange-characters-string-no-two-adjacent/

Problem Statement:
Rearrange characters in a string so no two adjacent characters are same. Return if possible and the rearranged string.

Sample Input/Output:
Input: "aabb"
Output: "abab"
Explanation: Max-heap frequency based approach or count check approach.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Rearrange_Max_Heap(string s) {
        /*
        Max-heap frequency based approach
        Time Complexity: O(n log k) where k is distinct chars
        Space Complexity: O(n)
        */
        unordered_map<char, int> freq;
        for (char c : s) {
            freq[c]++;
        }
        
        priority_queue<pair<int, char>> max_heap;
        for (auto& p : freq) {
            max_heap.push({p.second, p.first});
        }
        
        string result = "";
        pair<int, char> prev = {-1, '#'};
        
        while (!max_heap.empty() || prev.first > 0) {
            if (prev.first > 0 && max_heap.empty()) {
                return "";
            }
            
            pair<int, char> curr = max_heap.top();
            max_heap.pop();
            
            result += curr.second;
            curr.first--;
            
            if (prev.first > 0) {
                max_heap.push(prev);
            }
            
            prev = curr;
        }
        
        return result;
    }
    
    string Rearrange_Count_Check(string s) {
        /*
        Count check approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        unordered_map<char, int> freq;
        int max_freq = 0;
        
        for (char c : s) {
            freq[c]++;
            max_freq = max(max_freq, freq[c]);
        }
        
        if (max_freq > (s.length() + 1) / 2) {
            return "";
        }
        
        return Rearrange_Max_Heap(s);
    }
};

void Test_Rearrange_No_Adjacent_Same() {
    Solution solution;
    
    cout << "Test 1 (Max-Heap): " << solution.Rearrange_Max_Heap("aabb") << endl;
    cout << "Test 1 (Count Check): " << solution.Rearrange_Count_Check("aabb") << endl;
    
    cout << "Test 2 (Max-Heap): " << solution.Rearrange_Max_Heap("aaabc") << endl;
    cout << "Test 2 (Count Check): " << solution.Rearrange_Count_Check("aaabc") << endl;
    
    cout << "Test 3 (Max-Heap): " << solution.Rearrange_Max_Heap("aaa") << endl;
    cout << "Test 3 (Count Check): " << solution.Rearrange_Count_Check("aaa") << endl;
}

int main() {
    Test_Rearrange_No_Adjacent_Same();
    return 0;
}
