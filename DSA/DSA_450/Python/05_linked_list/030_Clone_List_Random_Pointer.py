"""
Problem: Clone a Linked List with Next and Random Pointer
URL: https://practice.geeksforgeeks.org/problems/clone-a-linked-list-with-next-and-random-pointer/1

Problem Statement:
Given a linked list where each node has a next pointer and a random pointer pointing to any node in the list or NULL. Clone this linked list.

Sample Input/Output:
Input: 1 -> 2 -> 3 -> 4 -> NULL
       |    |    |    |
       NULL 1    3    2
Output: 1 -> 2 -> 3 -> 4 -> NULL
        |    |    |    |
        NULL 1    3    2
Explanation: Create a deep copy with same structure and random pointers.
"""

class RandomNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.random = None

def Create_Random_List(arr, random_indices):
    if not arr:
        return None
    nodes = []
    head = RandomNode(arr[0])
    current = head
    nodes.append(head)
    for i in range(1, len(arr)):
        current.next = RandomNode(arr[i])
        current = current.next
        nodes.append(current)
    for i in range(len(random_indices)):
        if random_indices[i] != -1:
            nodes[i].random = nodes[random_indices[i]]
    return head

def List_To_Array(head):
    result = []
    while head:
        result.append(head.data)
        head = head.next
    return result

def Print_Random_List(head):
    current = head
    result = []
    while current:
        random_val = current.random.data if current.random else "NULL"
        result.append(f"{current.data} (random: {random_val})")
        current = current.next
    print(" -> ".join(result))

class Solution:
    def Clone_Hash_Map(self, head):
        """
        Use hash map to store original to clone mapping
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if head is None:
            return None
        map_dict = {}
        current = head
        while current:
            map_dict[current] = RandomNode(current.data)
            current = current.next
        current = head
        while current:
            map_dict[current].next = map_dict.get(current.next) if current.next else None
            map_dict[current].random = map_dict.get(current.random) if current.random else None
            current = current.next
        return map_dict[head]

    def Clone_Interleaving_Nodes(self, head):
        """
        Interleave original and clone nodes, then separate
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if head is None:
            return None
        current = head
        while current:
            clone = RandomNode(current.data)
            clone.next = current.next
            current.next = clone
            current = clone.next
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next
        current = head
        clone_head = head.next
        clone_current = clone_head
        while current:
            current.next = current.next.next
            if clone_current.next:
                clone_current.next = clone_current.next.next
            current = current.next
            clone_current = clone_current.next
        return clone_head

def Test_Clone_List_Random_Pointer():
    solution = Solution()
    
    test1_arr = [1, 2, 3, 4]
    test1_random = [-1, 0, 2, 1]
    list1 = Create_Random_List(test1_arr, test1_random)
    clone1 = solution.Clone_Hash_Map(list1)
    print("Test 1 Hash Map Clone: ", end="")
    Print_Random_List(clone1)
    
    test2_arr = [1, 3, 5]
    test2_random = [2, -1, 0]
    list2 = Create_Random_List(test2_arr, test2_random)
    clone2 = solution.Clone_Interleaving_Nodes(list2)
    print("Test 2 Interleaving Clone: ", end="")
    Print_Random_List(clone2)
    
    test3_arr = [7]
    test3_random = [-1]
    list3 = Create_Random_List(test3_arr, test3_random)
    clone3 = solution.Clone_Hash_Map(list3)
    print("Test 3 Single Node: ", end="")
    Print_Random_List(clone3)

if __name__ == "__main__":
    Test_Clone_List_Random_Pointer()
