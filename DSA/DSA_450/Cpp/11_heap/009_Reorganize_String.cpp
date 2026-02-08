/*
Problem: Reorganize String
URL: https://leetcode.com/problems/reorganize-string/

Problem Statement:
Given a string, rearrange it so no two adjacent characters are the same. Return empty string if impossible.

Sample Input/Output:
Input: "aab"
Output: "aba"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Reorganize_String_Max_Heap(string s) {
        /*
        Greedy with Max Heap
        Time Complexity: O(n log n)
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
            if (max_heap.empty() && prev.first > 0) {
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

    string Reorganize_String_Counting(string s) {
        /*
        Counting and Place at Even Indices
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<char, int> freq;
        int max_freq = 0;
        char max_char = 'a';
        
        for (char c : s) {
            freq[c]++;
            if (freq[c] > max_freq) {
                max_freq = freq[c];
                max_char = c;
            }
        }
        
        int n = s.size();
        if (max_freq > (n + 1) / 2) {
            return "";
        }
        
        string result(n, ' ');
        int idx = 0;
        
        while (freq[max_char] > 0) {
            result[idx] = max_char;
            idx += 2;
            freq[max_char]--;
        }
        
        for (auto& p : freq) {
            while (p.second > 0) {
                if (idx >= n) {
                    idx = 1;
                }
                result[idx] = p.first;
                idx += 2;
                p.second--;
            }
        }
        
        return result;
    }
};

void Test_Reorganize_String() {
    Solution solution;
    
    string s1 = "aab";
    cout << "Input: \"" << s1 << "\"" << endl;
    string res1 = solution.Reorganize_String_Max_Heap(s1);
    cout << "Max Heap Result: \"" << res1 << "\"" << endl;
    string res2 = solution.Reorganize_String_Counting(s1);
    cout << "Counting Result: \"" << res2 << "\"" << endl;
    
    string s2 = "aaab";
    cout << "\nInput: \"" << s2 << "\"" << endl;
    string res3 = solution.Reorganize_String_Max_Heap(s2);
    cout << "Max Heap Result: \"" << res3 << "\"" << endl;
    string res4 = solution.Reorganize_String_Counting(s2);
    cout << "Counting Result: \"" << res4 << "\"" << endl;
    
    string s3 = "aabbcc";
    cout << "\nInput: \"" << s3 << "\"" << endl;
    string res5 = solution.Reorganize_String_Max_Heap(s3);
    cout << "Max Heap Result: \"" << res5 << "\"" << endl;
    string res6 = solution.Reorganize_String_Counting(s3);
    cout << "Counting Result: \"" << res6 << "\"" << endl;
    
    string s4 = "vvvlo";
    cout << "\nInput: \"" << s4 << "\"" << endl;
    string res7 = solution.Reorganize_String_Max_Heap(s4);
    cout << "Max Heap Result: \"" << res7 << "\"" << endl;
    string res8 = solution.Reorganize_String_Counting(s4);
    cout << "Counting Result: \"" << res8 << "\"" << endl;
    
    string s5 = "aaabb";
    cout << "\nInput: \"" << s5 << "\"" << endl;
    string res9 = solution.Reorganize_String_Max_Heap(s5);
    cout << "Max Heap Result: \"" << res9 << "\"" << endl;
    string res10 = solution.Reorganize_String_Counting(s5);
    cout << "Counting Result: \"" << res10 << "\"" << endl;
}

int main() {
    Test_Reorganize_String();
    return 0;
}
