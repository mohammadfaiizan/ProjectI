/*
Problem: Word Ladder
URL: https://leetcode.com/problems/word-ladder/

Problem Statement:
Find shortest transformation sequence length from beginWord to endWord, changing one letter at a time with each word in wordList.

Sample Input/Output:
Input: begin="hit", end="cog", wordList=["hot","dot","dog","lot","log","cog"]
Output: 5
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Word_Ladder_BFS(string beginWord, string endWord, vector<string>& wordList) {
        /*
        BFS - Try All 26 Chars at Each Position
        Time Complexity: O(M^2 * N) where M=word length, N=list size
        Space Complexity: O(N)
        */
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        
        if (wordSet.find(endWord) == wordSet.end()) {
            return 0;
        }
        
        queue<pair<string, int>> q;
        q.push({beginWord, 1});
        wordSet.erase(beginWord);
        
        while (!q.empty()) {
            auto current = q.front();
            q.pop();
            
            string word = current.first;
            int level = current.second;
            
            if (word == endWord) {
                return level;
            }
            
            for (int i = 0; i < word.length(); i++) {
                char original = word[i];
                for (char c = 'a'; c <= 'z'; c++) {
                    if (c == original) continue;
                    
                    word[i] = c;
                    if (wordSet.find(word) != wordSet.end()) {
                        q.push({word, level + 1});
                        wordSet.erase(word);
                    }
                }
                word[i] = original;
            }
        }
        
        return 0;
    }

    int Word_Ladder_Bidirectional_BFS(string beginWord, string endWord, vector<string>& wordList) {
        /*
        Bidirectional BFS
        Time Complexity: O(M^2 * N)
        Space Complexity: O(N)
        */
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        
        if (wordSet.find(endWord) == wordSet.end()) {
            return 0;
        }
        
        unordered_set<string> beginSet, endSet;
        beginSet.insert(beginWord);
        endSet.insert(endWord);
        wordSet.erase(beginWord);
        wordSet.erase(endWord);
        
        int level = 1;
        
        while (!beginSet.empty() && !endSet.empty()) {
            if (beginSet.size() > endSet.size()) {
                swap(beginSet, endSet);
            }
            
            unordered_set<string> nextSet;
            
            for (string word : beginSet) {
                for (int i = 0; i < word.length(); i++) {
                    char original = word[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == original) continue;
                        
                        word[i] = c;
                        
                        if (endSet.find(word) != endSet.end()) {
                            return level + 1;
                        }
                        
                        if (wordSet.find(word) != wordSet.end()) {
                            nextSet.insert(word);
                            wordSet.erase(word);
                        }
                    }
                    word[i] = original;
                }
            }
            
            beginSet = nextSet;
            level++;
        }
        
        return 0;
    }
};

void Test_Word_Ladder() {
    Solution solution;
    
    cout << "Test: Word Ladder" << endl;
    string beginWord = "hit";
    string endWord = "cog";
    vector<string> wordList = {"hot", "dot", "dog", "lot", "log", "cog"};
    
    int result1 = solution.Word_Ladder_BFS(beginWord, endWord, wordList);
    cout << "Shortest sequence length (BFS): " << result1 << endl;
    
    vector<string> wordList2 = {"hot", "dot", "dog", "lot", "log", "cog"};
    int result2 = solution.Word_Ladder_Bidirectional_BFS(beginWord, endWord, wordList2);
    cout << "Shortest sequence length (Bidirectional BFS): " << result2 << endl;
    
    cout << "\nTest 2: No valid transformation" << endl;
    string beginWord2 = "hit";
    string endWord2 = "cog";
    vector<string> wordList3 = {"hot", "dot", "dog", "lot", "log"};
    
    int result3 = solution.Word_Ladder_BFS(beginWord2, endWord2, wordList3);
    cout << "Shortest sequence length: " << result3 << endl;
}

int main() {
    Test_Word_Ladder();
    return 0;
}
