/*
Problem: Rearrange Characters So No Two Adjacent Are Same
URL: https://www.geeksforgeeks.org/rearrange-characters-string-no-two-adjacent/

Problem Statement:
Given a string, rearrange it so no two adjacent characters are the same. Return the rearranged string or empty if impossible.

Sample Input/Output:
Input: "aaabb"
Output: "ababa"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Rearrange_Max_Heap(string s) {
        /*
        Max Heap Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        unordered_map<char, int> freq;
        for (char c : s) {
            freq[c]++;
        }
        
        priority_queue<pair<int, char>> pq;
        for (auto& p : freq) {
            pq.push({p.second, p.first});
        }
        
        string result = "";
        pair<int, char> prev = {-1, '#'};
        
        while (!pq.empty() || prev.first > 0) {
            if (pq.empty() && prev.first > 0) {
                return "";
            }
            
            pair<int, char> current = pq.top();
            pq.pop();
            result += current.second;
            current.first--;
            
            if (prev.first > 0) {
                pq.push(prev);
            }
            
            prev = current;
        }
        
        return result;
    }
    
    string Rearrange_Fill_Even_Odd(string s) {
        /*
        Fill Even-Odd Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<char, int> freq;
        int maxFreq = 0;
        char maxChar = '#';
        
        for (char c : s) {
            freq[c]++;
            if (freq[c] > maxFreq) {
                maxFreq = freq[c];
                maxChar = c;
            }
        }
        
        int n = s.length();
        if (maxFreq > (n + 1) / 2) {
            return "";
        }
        
        vector<char> result(n);
        int idx = 0;
        
        for (int i = 0; i < maxFreq; i++) {
            result[idx] = maxChar;
            idx += 2;
        }
        
        freq[maxChar] = 0;
        
        for (auto& p : freq) {
            while (p.second > 0) {
                if (idx >= n) idx = 1;
                result[idx] = p.first;
                idx += 2;
                p.second--;
            }
        }
        
        return string(result.begin(), result.end());
    }
};

bool IsValid(string s) {
    for (int i = 1; i < s.length(); i++) {
        if (s[i] == s[i-1]) return false;
    }
    return true;
}

void Test_Rearrange() {
    Solution solution;
    
    string s1 = "aaabb";
    string result1 = solution.Rearrange_Max_Heap(s1);
    cout << "Input: " << s1 << " -> Output: " << result1 << " (Valid: " << IsValid(result1) << ")" << endl;
    
    string s1b = "aaabb";
    string result1b = solution.Rearrange_Fill_Even_Odd(s1b);
    cout << "Input: " << s1b << " -> Output: " << result1b << " (Valid: " << IsValid(result1b) << ")" << endl;
    
    string s2 = "aaab";
    string result2 = solution.Rearrange_Max_Heap(s2);
    cout << "Input: " << s2 << " -> Output: " << (result2.empty() ? "empty" : result2) << endl;
    
    string s2b = "aaab";
    string result2b = solution.Rearrange_Fill_Even_Odd(s2b);
    cout << "Input: " << s2b << " -> Output: " << (result2b.empty() ? "empty" : result2b) << endl;
    
    string s3 = "aabb";
    string result3 = solution.Rearrange_Max_Heap(s3);
    cout << "Input: " << s3 << " -> Output: " << result3 << " (Valid: " << IsValid(result3) << ")" << endl;
    
    string s3b = "aabb";
    string result3b = solution.Rearrange_Fill_Even_Odd(s3b);
    cout << "Input: " << s3b << " -> Output: " << result3b << " (Valid: " << IsValid(result3b) << ")" << endl;
}

int main() {
    Test_Rearrange();
    return 0;
}
