/*
Problem: Intersection Point in Y Shaped Linked Lists
URL: https://practice.geeksforgeeks.org/problems/intersection-point-in-y-shapped-linked-lists/1

Problem Statement:
There are two singly linked lists of size N and M in a system. But, due to some programming error the end node of one of the linked list got linked into the second list, forming an inverted Y shaped list. Write a program to get the point where two linked lists intersect.

Sample Input/Output:
Input: LinkList1 = 3->6->9->common, LinkList2 = 10->common, common = 15->30->NULL
Output: 15
Explanation: The Y shaped list ends after 15.
*/

#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int data;
    ListNode* next;
    ListNode(int x) : data(x), next(NULL) {}
};

ListNode* Create_List(vector<int>& arr) {
    if (arr.empty()) return NULL;
    ListNode* head = new ListNode(arr[0]);
    ListNode* curr = head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new ListNode(arr[i]);
        curr = curr->next;
    }
    return head;
}

void Print_List(ListNode* head) {
    while (head) {
        cout << head->data;
        if (head->next) cout << "->";
        head = head->next;
    }
    cout << endl;
}

class Solution {
public:
    int Intersect_Point_Hashing(ListNode* head1, ListNode* head2) {
        /*
        Hashing approach
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        */
        unordered_set<ListNode*> visited;
        while (head1) {
            visited.insert(head1);
            head1 = head1->next;
        }
        while (head2) {
            if (visited.find(head2) != visited.end()) {
                return head2->data;
            }
            head2 = head2->next;
        }
        return -1;
    }
    
    int Intersect_Point_Difference(ListNode* head1, ListNode* head2) {
        /*
        Difference of node counts
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        */
        int len1 = 0, len2 = 0;
        ListNode* temp1 = head1, *temp2 = head2;
        
        while (temp1) {
            len1++;
            temp1 = temp1->next;
        }
        while (temp2) {
            len2++;
            temp2 = temp2->next;
        }
        
        int diff = abs(len1 - len2);
        if (len1 > len2) {
            while (diff--) head1 = head1->next;
        } else {
            while (diff--) head2 = head2->next;
        }
        
        while (head1 && head2) {
            if (head1 == head2) {
                return head1->data;
            }
            head1 = head1->next;
            head2 = head2->next;
        }
        return -1;
    }
    
    int Intersect_Point_Two_Pointer(ListNode* head1, ListNode* head2) {
        /*
        Two pointer technique
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        */
        ListNode* ptr1 = head1;
        ListNode* ptr2 = head2;
        
        while (ptr1 != ptr2) {
            ptr1 = ptr1 ? ptr1->next : head2;
            ptr2 = ptr2 ? ptr2->next : head1;
        }
        
        return ptr1 ? ptr1->data : -1;
    }
};

void Test_Intersection_Point_Y_Shape() {
    Solution solution;
    
    ListNode* common = new ListNode(15);
    common->next = new ListNode(30);
    
    ListNode* head1 = new ListNode(3);
    head1->next = new ListNode(6);
    head1->next->next = new ListNode(9);
    head1->next->next->next = common;
    
    ListNode* head2 = new ListNode(10);
    head2->next = common;
    
    int result1 = solution.Intersect_Point_Hashing(head1, head2);
    cout << "Test 1 - Hashing: " << result1 << endl;
    
    int result2 = solution.Intersect_Point_Difference(head1, head2);
    cout << "Test 1 - Difference: " << result2 << endl;
    
    int result3 = solution.Intersect_Point_Two_Pointer(head1, head2);
    cout << "Test 1 - Two Pointer: " << result3 << endl;
}

int main() {
    Test_Intersection_Point_Y_Shape();
    return 0;
}
