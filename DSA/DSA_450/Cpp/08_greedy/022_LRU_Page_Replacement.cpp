/*
Problem: LRU Page Replacement
URL: https://practice.geeksforgeeks.org/problems/page-faults-in-lru5603/1

Problem Statement:
Count page faults using LRU page replacement algorithm given page references and capacity.

Sample Input/Output:
Input: pages[] = {5, 0, 1, 3, 2, 4, 1, 0, 5}, capacity = 4
Output: 8
Explanation: Page faults occur when pages are not in memory and capacity is full.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LRU_Page_Replacement_HashSet_HashMap(vector<int>& pages, int capacity) {
        /*
        HashSet + HashMap index tracking greedy approach
        Time Complexity: O(n * capacity)
        Space Complexity: O(capacity)
        */
        unordered_set<int> memory;
        unordered_map<int, int> last_used;
        int page_faults = 0;
        
        for (int i = 0; i < pages.size(); i++) {
            int page = pages[i];
            
            if (memory.find(page) != memory.end()) {
                last_used[page] = i;
                continue;
            }
            
            page_faults++;
            
            if (memory.size() < capacity) {
                memory.insert(page);
                last_used[page] = i;
            } else {
                int lru_page = -1;
                int min_time = INT_MAX;
                
                for (int p : memory) {
                    if (last_used[p] < min_time) {
                        min_time = last_used[p];
                        lru_page = p;
                    }
                }
                
                memory.erase(lru_page);
                memory.insert(page);
                last_used[page] = i;
            }
        }
        
        return page_faults;
    }
};

void Test_LRU_Page_Replacement() {
    Solution solution;
    
    vector<int> pages1 = {5, 0, 1, 3, 2, 4, 1, 0, 5};
    cout << "Test 1: " << solution.LRU_Page_Replacement_HashSet_HashMap(pages1, 4) << endl;
    
    vector<int> pages2 = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2};
    cout << "Test 2: " << solution.LRU_Page_Replacement_HashSet_HashMap(pages2, 4) << endl;
    
    vector<int> pages3 = {1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5};
    cout << "Test 3: " << solution.LRU_Page_Replacement_HashSet_HashMap(pages3, 3) << endl;
}

int main() {
    Test_LRU_Page_Replacement();
    return 0;
}
