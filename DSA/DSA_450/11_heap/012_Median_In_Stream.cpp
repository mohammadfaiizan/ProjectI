/*
Problem: Find Median in a Stream of Integers
URL: https://practice.geeksforgeeks.org/problems/find-median-in-a-stream-1587115620/1

Problem Statement:
Design a data structure that supports adding integers and finding median efficiently.

Sample Input/Output:
Input: stream [5,15,1,3,2,8,7,9]
Output: medians after each insertion
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
private:
    priority_queue<int> maxHeap;
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
public:
    void Insert_Median(int num) {
        /*
        Two Heaps Approach
        Time Complexity: O(log n)
        Space Complexity: O(n)
        */
        if (maxHeap.empty() || num <= maxHeap.top()) {
            maxHeap.push(num);
        } else {
            minHeap.push(num);
        }
        
        if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        } else if (minHeap.size() > maxHeap.size() + 1) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double Find_Median() {
        /*
        Find Median
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        if (maxHeap.size() == minHeap.size()) {
            return (maxHeap.top() + minHeap.top()) / 2.0;
        } else if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        } else {
            return minHeap.top();
        }
    }
};

void Test_Median_Stream() {
    Solution solution;
    
    vector<int> stream1 = {5, 15, 1, 3, 2, 8, 7, 9};
    cout << "Stream 1: ";
    for (int num : stream1) {
        solution.Insert_Median(num);
        cout << num << " -> Median: " << solution.Find_Median() << " ";
    }
    cout << endl;
    
    Solution solution2;
    vector<int> stream2 = {1, 2, 3, 4, 5};
    cout << "Stream 2: ";
    for (int num : stream2) {
        solution2.Insert_Median(num);
        cout << num << " -> Median: " << solution2.Find_Median() << " ";
    }
    cout << endl;
    
    Solution solution3;
    vector<int> stream3 = {10, 20, 30};
    cout << "Stream 3: ";
    for (int num : stream3) {
        solution3.Insert_Median(num);
        cout << num << " -> Median: " << solution3.Find_Median() << " ";
    }
    cout << endl;
}

int main() {
    Test_Median_Stream();
    return 0;
}
