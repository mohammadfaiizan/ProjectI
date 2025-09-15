"""
Problem: Linked List Cycle II
URL: https://leetcode.com/problems/linked-list-cycle-ii/

Problem Statement:
Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.
There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. 
Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. 
Note that pos is not passed as a parameter.

Sample Input/Output:
Input: head = [3,2,0,-4], pos = 1
Output: node with value 2
Explanation: There is a cycle in the linked list, where tail connects to the second node.

Input: head = [1,2], pos = 0
Output: node with value 1
Explanation: There is a cycle in the linked list, where tail connects to the first node.

Input: head = [1], pos = -1
Output: null
Explanation: There is no cycle in the linked list.
"""

from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def Detect_Cycle_Hash_Set(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Hash Set - Store visited nodes and return first duplicate
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        visited = set()
        current = head
        
        while current:
            if current in visited:
                return current
            visited.add(current)
            current = current.next
        
        return None
    
    def Detect_Cycle_Floyd_Optimal(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Floyd's Algorithm Optimal - Mathematical cycle detection
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return None
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                start = head
                
                while start != slow:
                    start = start.next
                    slow = slow.next
                
                return start
        
        return None
    
    def Detect_Cycle_Distance_Calculation(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Distance Calculation - Calculate cycle length then find start
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None
        
        def Get_Cycle_Length() -> int:
            slow = fast = head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    length = 1
                    current = slow
                    
                    while current.next != slow:
                        current = current.next
                        length += 1
                    
                    return length
            
            return 0
        
        cycle_length = Get_Cycle_Length()
        
        if cycle_length == 0:
            return None
        
        first = head
        second = head
        
        for _ in range(cycle_length):
            second = second.next
        
        while first != second:
            first = first.next
            second = second.next
        
        return first
    
    def Detect_Cycle_Two_Pass(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pass - First detect cycle, then find start
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return None
        
        def Has_Cycle() -> bool:
            slow = fast = head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    return True
            
            return False
        
        if not Has_Cycle():
            return None
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        
        slow = head
        
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def Detect_Cycle_Mark_Nodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Mark Nodes - Mark visited nodes with special value
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None
        
        MARKER = "VISITED"
        current = head
        
        while current:
            if hasattr(current, 'marker') and current.marker == MARKER:
                return current
            
            current.marker = MARKER
            current = current.next
        
        return None
    
    def Detect_Cycle_Reverse_Engineering(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse Engineering - Use mathematical properties
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return None
        
        slow = fast = head
        meeting_point = None
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                meeting_point = slow
                break
        
        if not meeting_point:
            return None
        
        def Count_Steps_To_Meeting(start: ListNode) -> int:
            steps = 0
            current = start
            
            while current != meeting_point:
                current = current.next
                steps += 1
            
            return steps
        
        steps_from_head = Count_Steps_To_Meeting(head)
        
        current = head
        for _ in range(steps_from_head):
            current = current.next
        
        return current

def Create_Cyclic_List(values, pos):
    if not values:
        return None
    
    nodes = []
    for val in values:
        nodes.append(ListNode(val))
    
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    if pos >= 0 and pos < len(nodes):
        nodes[-1].next = nodes[pos]
        return nodes[0], nodes[pos]
    
    return nodes[0], None

def Test_Detect_Cycle():
    solution = Solution()
    
    test_cases = [
        ([3,2,0,-4], 1, 2),
        ([1,2], 0, 1),
        ([1], -1, None),
        ([1,2,3,4,5], 2, 3),
        ([1,2,3], -1, None)
    ]
    
    methods = [
        ("Hash Set", solution.Detect_Cycle_Hash_Set),
        ("Floyd Optimal", solution.Detect_Cycle_Floyd_Optimal),
        ("Distance Calculation", solution.Detect_Cycle_Distance_Calculation),
        ("Two Pass", solution.Detect_Cycle_Two_Pass)
    ]
    
    for values, pos, expected_val in test_cases:
        print(f"Values: {values}, Pos: {pos}")
        print(f"Expected Value: {expected_val}")
        
        for method_name, method in methods:
            head, expected_node = Create_Cyclic_List(values, pos)
            try:
                result = method(head)
                result_val = result.val if result else None
                print(f"{method_name}: {result_val}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Detect_Cycle()
