/*
Problem: LRU Cache Implementation
URL: https://practice.geeksforgeeks.org/problems/lru-cache/1

Problem Statement:
Implement LRU Cache with get and put operations in O(1) time.

Sample Input/Output:
Input: capacity=2, put(1,1), put(2,2), get(1), put(3,3), get(2)
Output: 1, -1
*/

#include <bits/stdc++.h>
using namespace std;

class LRUCache {
private:
    int capacity;
    list<pair<int, int>> cache;
    unordered_map<int, list<pair<int, int>>::iterator> map;

public:
    LRUCache(int cap) : capacity(cap) {}

    int Get(int key) {
        if (map.find(key) == map.end()) {
            return -1;
        }
        auto it = map[key];
        int value = it->second;
        cache.erase(it);
        cache.push_front({key, value});
        map[key] = cache.begin();
        return value;
    }

    void Put(int key, int value) {
        if (map.find(key) != map.end()) {
            cache.erase(map[key]);
        } else if (cache.size() >= capacity) {
            auto last = cache.back();
            map.erase(last.first);
            cache.pop_back();
        }
        cache.push_front({key, value});
        map[key] = cache.begin();
    }
};

class Solution {
public:
    void Test_LRU_Cache() {
        LRUCache cache(2);
        
        cache.Put(1, 1);
        cache.Put(2, 2);
        cout << "Get(1): " << cache.Get(1) << endl;
        
        cache.Put(3, 3);
        cout << "Get(2): " << cache.Get(2) << endl;
        cout << "Get(3): " << cache.Get(3) << endl;
        cout << "Get(1): " << cache.Get(1) << endl;
        
        cache.Put(4, 4);
        cout << "Get(1): " << cache.Get(1) << endl;
        cout << "Get(3): " << cache.Get(3) << endl;
        cout << "Get(4): " << cache.Get(4) << endl;
    }
};

void Test_LRU_Cache() {
    Solution solution;
    solution.Test_LRU_Cache();
}

int main() {
    Test_LRU_Cache();
    return 0;
}
