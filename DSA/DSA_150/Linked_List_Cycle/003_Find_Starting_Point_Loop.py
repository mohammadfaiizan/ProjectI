"""
Problem: Find the Starting Point of the Loop
URL: https://www.geeksforgeeks.org/find-the-starting-point-of-the-loop-in-a-linked-list/

Problem Statement:
Given a linked list with a loop, find the starting node of the loop.

Sample Input/Output:
Input: 1 -> 2 -> 3 -> 4 -> 5 -> 2 (cycle back to node 2)
Output: 2
Explanation: The loop starts at node with value 2.

Input: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
Output: NULL
Explanation: There is no loop in the linked list.
"""

from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def Detect_Cycle_Hash_Set(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Hash Set - Store visited nodes
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
        Floyd's Algorithm Optimal - Mathematical approach
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
                slow = head
                
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                
                return slow
        
        return None
    
    def Detect_Cycle_Count_Method(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Count Method - Count nodes then find start
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None
        
        def Get_Loop_Length() -> int:
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
        
        loop_length = Get_Loop_Length()
        
        if loop_length == 0:
            return None
        
        first = head
        second = head
        
        for _ in range(loop_length):
            second = second.next
        
        while first != second:
            first = first.next
            second = second.next
        
        return first
    
    def Detect_Cycle_Distance_Calculation(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Distance Calculation - Calculate distances to intersection
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
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
        
        def Count_Distance_To_Start(node: ListNode) -> int:
            distance = 0
            current = head
            
            while current != node:
                current = current.next
                distance += 1
            
            return distance
        
        current = head
        
        while True:
            temp_slow = meeting_point
            temp_fast = meeting_point
            found = False
            
            while temp_fast and temp_fast.next:
                temp_slow = temp_slow.next
                temp_fast = temp_fast.next.next
                
                if temp_slow == current:
                    found = True
                    break
            
            if found:
                return current
            
            current = current.next
    
    def Detect_Cycle_Mark_Nodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Mark Nodes - Mark visited nodes
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
    
    def Detect_Cycle_Two_Pointer_Variant(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pointer Variant - Alternative two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return None
        
        def Find_Meeting_Point() -> Optional[ListNode]:
            slow = fast = head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    return slow
            
            return None
        
        meeting_point = Find_Meeting_Point()
        
        if not meeting_point:
            return None
        
        ptr1 = head
        ptr2 = meeting_point
        
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        
        return ptr1

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
        ([1,2,3,4,5], 1, 2),
        ([1,2,3,4,5], 0, 1),
        ([1,2,3,4,5], -1, None),
        ([1,2], 0, 1),
        ([1], -1, None),
        ([3,2,0,-4], 1, 2)
    ]
    
    methods = [
        ("Hash Set", solution.Detect_Cycle_Hash_Set),
        ("Floyd Optimal", solution.Detect_Cycle_Floyd_Optimal),
        ("Count Method", solution.Detect_Cycle_Count_Method),
        ("Two Pointer Variant", solution.Detect_Cycle_Two_Pointer_Variant)
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
