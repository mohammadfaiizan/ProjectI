/*
Problem: Word Wrap
URL: https://practice.geeksforgeeks.org/problems/word-wrap1646/1

Problem Statement:
Given a sequence of words and a line width, arrange the words in lines such that the total cost (penalty for extra spaces) is minimized.

Sample Input/Output:
Input: words = [3,2,2,5], line_width = 6
Output: Minimum cost arrangement
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Word_Wrap_DP(vector<int>& words, int lineWidth) {
        /*
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = words.size();
        vector<int> dp(n + 1, INT_MAX);
        dp[n] = 0;
        
        for (int i = n - 1; i >= 0; i--) {
            int currentLength = 0;
            for (int j = i; j < n; j++) {
                currentLength += words[j];
                if (j > i) currentLength++;
                
                if (currentLength > lineWidth) break;
                
                int cost = (j == n - 1) ? 0 : pow(lineWidth - currentLength, 2);
                dp[i] = min(dp[i], cost + dp[j + 1]);
            }
        }
        
        return dp[0];
    }
    
    int Word_Wrap_Greedy(vector<int>& words, int lineWidth) {
        /*
        Greedy approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = words.size();
        int cost = 0;
        int currentLength = 0;
        
        for (int i = 0; i < n; i++) {
            if (currentLength + words[i] > lineWidth) {
                if (currentLength > 0) {
                    cost += pow(lineWidth - currentLength, 2);
                }
                currentLength = words[i];
            } else {
                if (currentLength > 0) currentLength++;
                currentLength += words[i];
            }
        }
        
        if (currentLength > 0) {
            cost += pow(lineWidth - currentLength, 2);
        }
        
        return cost;
    }
};

void Test_Word_Wrap() {
    Solution solution;
    
    vector<int> words = {3, 2, 2, 5};
    int lineWidth = 6;
    
    cout << "DP: ";
    for (int w : words) cout << w << " ";
    cout << ", width=" << lineWidth << " -> " 
         << solution.Word_Wrap_DP(words, lineWidth) << endl;
    
    cout << "Greedy: ";
    for (int w : words) cout << w << " ";
    cout << ", width=" << lineWidth << " -> " 
         << solution.Word_Wrap_Greedy(words, lineWidth) << endl;
}

int main() {
    Test_Word_Wrap();
    return 0;
}
