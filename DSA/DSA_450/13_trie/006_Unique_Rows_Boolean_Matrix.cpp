/*
Problem: Print Unique Rows in a Given Boolean Matrix
URL: https://practice.geeksforgeeks.org/problems/unique-rows-in-boolean-matrix/1

Problem Statement:
Given a boolean matrix, print all unique rows.

Sample Input/Output:
Input: matrix with duplicate and unique rows
Output: unique rows only
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[2];
    bool isEndOfRow;
    
    TrieNode() {
        children[0] = nullptr;
        children[1] = nullptr;
        isEndOfRow = false;
    }
};

class Solution {
public:
    vector<vector<int>> Unique_Rows_Trie(vector<vector<int>>& matrix) {
        /*
        Unique_Rows_Trie (insert each row into trie, print only new insertions, O(R*C))
        Time Complexity: O(R*C) where R is rows, C is columns
        Space Complexity: O(R*C)
        */
        TrieNode* root = new TrieNode();
        vector<vector<int>> result;
        
        for (vector<int> row : matrix) {
            TrieNode* node = root;
            bool isNew = false;
            
            for (int val : row) {
                if (!node->children[val]) {
                    node->children[val] = new TrieNode();
                    isNew = true;
                }
                node = node->children[val];
            }
            
            if (!node->isEndOfRow) {
                node->isEndOfRow = true;
                result.push_back(row);
            }
        }
        
        return result;
    }
    
    vector<vector<int>> Unique_Rows_Set(vector<vector<int>>& matrix) {
        /*
        Unique_Rows_Set (use set of strings/vectors, O(R*C*log R))
        Time Complexity: O(R*C*log R) where R is rows, C is columns
        Space Complexity: O(R*C)
        */
        set<vector<int>> seen;
        vector<vector<int>> result;
        
        for (vector<int> row : matrix) {
            if (seen.find(row) == seen.end()) {
                seen.insert(row);
                result.push_back(row);
            }
        }
        
        return result;
    }
    
    vector<vector<int>> Unique_Rows_Set_String(vector<vector<int>>& matrix) {
        /*
        Unique_Rows_Set_String (convert rows to strings, use set, O(R*C*log R))
        Time Complexity: O(R*C*log R)
        Space Complexity: O(R*C)
        */
        set<string> seen;
        vector<vector<int>> result;
        
        for (vector<int> row : matrix) {
            string rowStr = "";
            for (int val : row) {
                rowStr += to_string(val);
            }
            
            if (seen.find(rowStr) == seen.end()) {
                seen.insert(rowStr);
                result.push_back(row);
            }
        }
        
        return result;
    }
    
    vector<vector<int>> Unique_Rows_Map(vector<vector<int>>& matrix) {
        /*
        Unique_Rows_Map (use map to track first occurrence, O(R*C))
        Time Complexity: O(R*C)
        Space Complexity: O(R*C)
        */
        unordered_map<string, bool> seen;
        vector<vector<int>> result;
        
        for (vector<int> row : matrix) {
            string rowStr = "";
            for (int val : row) {
                rowStr += to_string(val);
            }
            
            if (seen.find(rowStr) == seen.end()) {
                seen[rowStr] = true;
                result.push_back(row);
            }
        }
        
        return result;
    }
};

void Test_Unique_Rows_Boolean_Matrix() {
    Solution solution;
    
    cout << "=== Test Case 1 ===" << endl;
    vector<vector<int>> matrix1 = {
        {1, 0, 0, 1},
        {0, 1, 1, 0},
        {1, 0, 0, 1},
        {0, 0, 1, 1}
    };
    cout << "Input Matrix:" << endl;
    for (auto row : matrix1) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> result1 = solution.Unique_Rows_Trie(matrix1);
    cout << "Unique Rows (Trie):" << endl;
    for (auto row : result1) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 2 ===" << endl;
    vector<vector<int>> matrix2 = {
        {1, 1, 0},
        {0, 0, 1},
        {1, 1, 0},
        {0, 0, 1},
        {1, 0, 1}
    };
    cout << "Input Matrix:" << endl;
    for (auto row : matrix2) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> result2 = solution.Unique_Rows_Trie(matrix2);
    cout << "Unique Rows (Trie):" << endl;
    for (auto row : result2) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 3 (All unique) ===" << endl;
    vector<vector<int>> matrix3 = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    cout << "Input Matrix:" << endl;
    for (auto row : matrix3) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> result3 = solution.Unique_Rows_Trie(matrix3);
    cout << "Unique Rows (Trie):" << endl;
    for (auto row : result3) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 4 (All same) ===" << endl;
    vector<vector<int>> matrix4 = {
        {1, 0, 1},
        {1, 0, 1},
        {1, 0, 1}
    };
    cout << "Input Matrix:" << endl;
    for (auto row : matrix4) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> result4 = solution.Unique_Rows_Trie(matrix4);
    cout << "Unique Rows (Trie):" << endl;
    for (auto row : result4) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 5 ===" << endl;
    vector<vector<int>> matrix5 = {
        {0},
        {1},
        {0},
        {1},
        {0}
    };
    cout << "Input Matrix:" << endl;
    for (auto row : matrix5) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> result5 = solution.Unique_Rows_Trie(matrix5);
    cout << "Unique Rows (Trie):" << endl;
    for (auto row : result5) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Comparison Test ===" << endl;
    vector<vector<int>> matrix6 = {
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {1, 1, 0, 0}
    };
    
    vector<vector<int>> trieResult = solution.Unique_Rows_Trie(matrix6);
    vector<vector<int>> setResult = solution.Unique_Rows_Set(matrix6);
    vector<vector<int>> mapResult = solution.Unique_Rows_Map(matrix6);
    
    cout << "Trie method count: " << trieResult.size() << endl;
    cout << "Set method count: " << setResult.size() << endl;
    cout << "Map method count: " << mapResult.size() << endl;
}

int main() {
    Test_Unique_Rows_Boolean_Matrix();
    return 0;
}
