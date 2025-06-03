"""
Linked List Problems - Classic Interview Problems and Solutions
This module implements solutions to the most common linked list problems asked in interviews.
"""

from typing import Optional, List, Dict, Set

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class FlattenNode:
    """Node class for flatten problems"""
    def __init__(self, val: int = 0, next: Optional['FlattenNode'] = None, 
                 down: Optional['FlattenNode'] = None):
        self.val = val
        self.next = next
        self.down = down
    
    def __repr__(self):
        return f"FlattenNode({self.val})"

class LinkedListProblems:
    """
    Implementation of classic linked list interview problems
    """
    
    def __init__(self):
        """Initialize problems solver"""
        pass
    
    # ==================== REMOVE DUPLICATES ====================
    
    def remove_duplicates_sorted(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Remove duplicates from sorted linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of sorted linked list
        
        Returns:
            Head of list with duplicates removed
        """
        if not head:
            return head
        
        current = head
        
        while current.next:
            if current.val == current.next.val:
                current.next = current.next.next
            else:
                current = current.next
        
        return head
    
    def remove_duplicates_sorted_all(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Remove all nodes that have duplicates from sorted list
        
        Args:
            head: Head of sorted linked list
        
        Returns:
            Head of list with all duplicates removed
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        while current:
            # Check if current node has duplicates
            if current.next and current.val == current.next.val:
                # Skip all nodes with same value
                val = current.val
                while current and current.val == val:
                    current = current.next
                prev.next = current
            else:
                prev = current
                current = current.next
        
        return dummy.next
    
    def remove_duplicates_unsorted(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Remove duplicates from unsorted linked list
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of list with duplicates removed
        """
        if not head:
            return head
        
        seen = set()
        current = head
        seen.add(current.val)
        
        while current.next:
            if current.next.val in seen:
                current.next = current.next.next
            else:
                seen.add(current.next.val)
                current = current.next
        
        return head
    
    def remove_duplicates_unsorted_no_extra_space(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Remove duplicates from unsorted list without extra space
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of list with duplicates removed
        """
        if not head:
            return head
        
        current = head
        
        while current:
            # Remove all future nodes with same value
            runner = current
            while runner.next:
                if runner.next.val == current.val:
                    runner.next = runner.next.next
                else:
                    runner = runner.next
            current = current.next
        
        return head
    
    # ==================== REMOVE NTH NODE FROM END ====================
    
    def remove_nth_from_end(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Remove nth node from end of list
        
        Time Complexity: O(length)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            n: Position from end to remove (1-indexed)
        
        Returns:
            Head of modified list
        """
        dummy = ListNode(0)
        dummy.next = head
        fast = slow = dummy
        
        # Move fast pointer n+1 steps ahead
        for _ in range(n + 1):
            if fast:
                fast = fast.next
            else:
                return head  # n is larger than list length
        
        # Move both pointers until fast reaches end
        while fast:
            fast = fast.next
            slow = slow.next
        
        # Remove nth node from end
        slow.next = slow.next.next
        
        return dummy.next
    
    def remove_nth_from_end_two_pass(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Remove nth node from end using two-pass approach
        
        Args:
            head: Head of linked list
            n: Position from end to remove (1-indexed)
        
        Returns:
            Head of modified list
        """
        # First pass: get length
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # Handle edge case: remove head
        if n == length:
            return head.next
        
        # Second pass: find node to remove
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        
        for _ in range(length - n):
            current = current.next
        
        current.next = current.next.next
        
        return dummy.next
    
    def remove_multiple_nth_from_end(self, head: Optional[ListNode], positions: List[int]) -> Optional[ListNode]:
        """
        Remove multiple nodes from end
        
        Args:
            head: Head of linked list
            positions: List of positions from end to remove
        
        Returns:
            Head of modified list
        """
        # Sort positions in descending order to remove from right to left
        positions.sort(reverse=True)
        
        for pos in positions:
            head = self.remove_nth_from_end(head, pos)
        
        return head
    
    # ==================== PARTITION LIST ====================
    
    def partition_list(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        """
        Partition list around value x (like QuickSort pivot)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            x: Partition value
        
        Returns:
            Head of partitioned list
        """
        # Create two separate lists
        less_dummy = ListNode(0)
        greater_dummy = ListNode(0)
        
        less = less_dummy
        greater = greater_dummy
        current = head
        
        while current:
            if current.val < x:
                less.next = current
                less = less.next
            else:
                greater.next = current
                greater = greater.next
            current = current.next
        
        # Connect the two lists
        greater.next = None  # Important: terminate the greater list
        less.next = greater_dummy.next
        
        return less_dummy.next
    
    def partition_list_stable(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        """
        Stable partition maintaining relative order
        
        Args:
            head: Head of linked list
            x: Partition value
        
        Returns:
            Head of partitioned list
        """
        if not head:
            return head
        
        # Keep original order within partitions
        before_head = before = ListNode(0)
        after_head = after = ListNode(0)
        
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        after.next = None
        before.next = after_head.next
        
        return before_head.next
    
    def three_way_partition(self, head: Optional[ListNode], low: int, high: int) -> Optional[ListNode]:
        """
        Three-way partition: < low, >= low and <= high, > high
        
        Args:
            head: Head of linked list
            low: Lower bound
            high: Upper bound
        
        Returns:
            Head of partitioned list
        """
        small_head = small = ListNode(0)
        medium_head = medium = ListNode(0)
        large_head = large = ListNode(0)
        
        current = head
        
        while current:
            if current.val < low:
                small.next = current
                small = small.next
            elif current.val <= high:
                medium.next = current
                medium = medium.next
            else:
                large.next = current
                large = large.next
            current = current.next
        
        # Connect all three parts
        large.next = None
        medium.next = large_head.next
        small.next = medium_head.next
        
        return small_head.next
    
    # ==================== ADD TWO NUMBERS ====================
    
    def add_two_numbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Add two numbers represented as linked lists (LeetCode 2)
        Numbers are stored in reverse order
        
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        
        Args:
            l1: First number (reverse order)
            l2: Second number (reverse order)
        
        Returns:
            Sum as linked list (reverse order)
        """
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            current.next = ListNode(digit)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy.next
    
    def add_two_numbers_forward(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Add two numbers stored in forward order
        
        Args:
            l1: First number (forward order)
            l2: Second number (forward order)
        
        Returns:
            Sum as linked list (forward order)
        """
        # Use stack to process from right to left
        stack1, stack2 = [], []
        
        # Push all digits to stacks
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        
        while l2:
            stack2.append(l2.val)
            l2 = l2.next
        
        # Process addition
        carry = 0
        result = None
        
        while stack1 or stack2 or carry:
            val1 = stack1.pop() if stack1 else 0
            val2 = stack2.pop() if stack2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            # Build result in reverse order
            new_node = ListNode(digit)
            new_node.next = result
            result = new_node
        
        return result
    
    def multiply_two_numbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Multiply two numbers represented as linked lists
        
        Args:
            l1: First number
            l2: Second number
        
        Returns:
            Product as linked list
        """
        # Convert to integers
        num1 = self._list_to_number(l1)
        num2 = self._list_to_number(l2)
        
        # Multiply
        product = num1 * num2
        
        # Convert back to list
        return self._number_to_list(product)
    
    # ==================== INTERSECTION OF TWO LISTS ====================
    
    def get_intersection_node(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection point of two linked lists
        
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        
        Args:
            headA: Head of first list
            headB: Head of second list
        
        Returns:
            Intersection node or None
        """
        if not headA or not headB:
            return None
        
        # Get lengths
        lenA = self._get_length(headA)
        lenB = self._get_length(headB)
        
        # Align starting points
        while lenA > lenB:
            headA = headA.next
            lenA -= 1
        
        while lenB > lenA:
            headB = headB.next
            lenB -= 1
        
        # Find intersection
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    def get_intersection_node_elegant(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Elegant intersection solution using two pointers
        
        Args:
            headA: Head of first list
            headB: Head of second list
        
        Returns:
            Intersection node or None
        """
        if not headA or not headB:
            return None
        
        ptrA = headA
        ptrB = headB
        
        while ptrA != ptrB:
            ptrA = headB if ptrA is None else ptrA.next
            ptrB = headA if ptrB is None else ptrB.next
        
        return ptrA
    
    def get_intersection_with_cycles(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection when lists may have cycles
        
        Args:
            headA: Head of first list
            headB: Head of second list
        
        Returns:
            Intersection node or None
        """
        # First detect if either list has cycle
        cycleA = self._detect_cycle(headA)
        cycleB = self._detect_cycle(headB)
        
        if not cycleA and not cycleB:
            # Both lists are acyclic
            return self.get_intersection_node(headA, headB)
        elif cycleA and cycleB:
            # Both lists have cycles
            # Check if cycles are the same
            current = cycleA.next
            while current != cycleA:
                if current == cycleB:
                    # Same cycle, find intersection before cycle
                    return self._get_intersection_before_cycle(headA, headB, cycleA)
                current = current.next
            return None
        else:
            # One has cycle, one doesn't - no intersection
            return None
    
    # ==================== FLATTEN LINKED LIST ====================
    
    def flatten_nested_list(self, head: Optional[FlattenNode]) -> Optional[FlattenNode]:
        """
        Flatten a nested linked list where each node may have a down pointer
        
        Time Complexity: O(n)
        Space Complexity: O(d) where d is maximum depth
        
        Args:
            head: Head of nested list
        
        Returns:
            Head of flattened list
        """
        if not head:
            return None
        
        stack = []
        current = head
        
        while current:
            if current.down:
                # If there's a next node, save it for later
                if current.next:
                    stack.append(current.next)
                
                # Connect down node as next
                current.next = current.down
                current.down = None
            
            # If no next node and stack has nodes
            if not current.next and stack:
                current.next = stack.pop()
            
            current = current.next
        
        return head
    
    def flatten_matrix_list(self, head: Optional[FlattenNode]) -> List[int]:
        """
        Flatten a matrix-like linked list structure
        
        Args:
            head: Head of matrix-like structure
        
        Returns:
            List of values in flattened order
        """
        result = []
        
        def dfs(node):
            if not node:
                return
            
            result.append(node.val)
            
            # Visit down first (depth-first)
            if node.down:
                dfs(node.down)
            
            # Then visit next
            if node.next:
                dfs(node.next)
        
        dfs(head)
        return result
    
    def flatten_sorted_merge(self, head: Optional[FlattenNode]) -> Optional[FlattenNode]:
        """
        Flatten nested list where each level is sorted, maintain sorted order
        
        Args:
            head: Head of nested sorted list
        
        Returns:
            Head of flattened sorted list
        """
        if not head:
            return None
        
        def merge_two_sorted(a, b):
            if not a:
                return b
            if not b:
                return a
            
            if a.val <= b.val:
                a.down = merge_two_sorted(a.down, b)
                return a
            else:
                b.down = merge_two_sorted(a, b.down)
                return b
        
        def flatten_recursive(node):
            if not node or not node.next:
                return node
            
            # Recursively flatten next part
            node.next = flatten_recursive(node.next)
            
            # Merge current column with flattened part
            node = merge_two_sorted(node, node.next)
            
            return node
        
        return flatten_recursive(head)
    
    # ==================== ADVANCED PROBLEMS ====================
    
    def merge_in_between(self, list1: Optional[ListNode], a: int, b: int, 
                        list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge list2 between positions a and b in list1
        
        Args:
            list1: First list
            a: Start position
            b: End position
            list2: List to merge in
        
        Returns:
            Head of merged list
        """
        # Find nodes at positions a-1 and b+1
        current = list1
        
        # Move to position a-1
        for _ in range(a - 1):
            current = current.next
        
        # Save node at position a-1
        before_a = current
        
        # Move to position b+1
        for _ in range(b - a + 2):
            current = current.next
        
        # Save node at position b+1
        after_b = current
        
        # Connect list2
        before_a.next = list2
        
        # Find end of list2
        while list2.next:
            list2 = list2.next
        
        # Connect to after_b
        list2.next = after_b
        
        return list1
    
    def reverse_nodes_in_k_group_with_remainder(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Reverse nodes in k-group, leave remainder as is
        
        Args:
            head: Head of linked list
            k: Group size
        
        Returns:
            Head of modified list
        """
        # Count total nodes
        count = 0
        current = head
        while current:
            count += 1
            current = current.next
        
        def reverse_k_nodes(node, k):
            prev = None
            current = node
            
            for _ in range(k):
                if not current:
                    break
                next_temp = current.next
                current.next = prev
                prev = current
                current = next_temp
            
            return prev, current
        
        dummy = ListNode(0)
        dummy.next = head
        prev_group_end = dummy
        
        while count >= k:
            group_start = prev_group_end.next
            
            # Reverse k nodes
            new_group_start, next_group = reverse_k_nodes(group_start, k)
            
            # Connect with previous group
            prev_group_end.next = new_group_start
            group_start.next = next_group
            
            # Update for next iteration
            prev_group_end = group_start
            count -= k
        
        return dummy.next
    
    def split_list_to_parts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        """
        Split linked list into k consecutive parts
        
        Args:
            head: Head of linked list
            k: Number of parts
        
        Returns:
            List of k parts
        """
        # Get total length
        length = self._get_length(head)
        
        # Calculate part sizes
        part_size = length // k
        extra_nodes = length % k
        
        result = []
        current = head
        
        for i in range(k):
            part_head = current
            
            # Calculate current part size
            current_part_size = part_size + (1 if i < extra_nodes else 0)
            
            # Move to end of current part
            for _ in range(current_part_size - 1):
                if current:
                    current = current.next
            
            # Break the connection
            if current:
                next_part = current.next
                current.next = None
                current = next_part
            
            result.append(part_head)
        
        return result
    
    # ==================== UTILITY METHODS ====================
    
    def _get_length(self, head: Optional[ListNode]) -> int:
        """Get length of linked list"""
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        return length
    
    def _detect_cycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Detect cycle and return cycle start node"""
        if not head or not head.next:
            return None
        
        slow = fast = head
        
        # Phase 1: Detect cycle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return None
        
        # Phase 2: Find cycle start
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def _get_intersection_before_cycle(self, headA: Optional[ListNode], 
                                     headB: Optional[ListNode], 
                                     cycle_start: ListNode) -> Optional[ListNode]:
        """Get intersection before cycle starts"""
        # This is a simplified implementation
        # In practice, you'd need more complex logic
        return self.get_intersection_node(headA, headB)
    
    def _list_to_number(self, head: Optional[ListNode]) -> int:
        """Convert linked list to number"""
        result = 0
        current = head
        
        while current:
            result = result * 10 + current.val
            current = current.next
        
        return result
    
    def _number_to_list(self, num: int) -> Optional[ListNode]:
        """Convert number to linked list"""
        if num == 0:
            return ListNode(0)
        
        dummy = ListNode(0)
        current = dummy
        
        while num > 0:
            digit = num % 10
            current.next = ListNode(digit)
            current = current.next
            num //= 10
        
        return dummy.next
    
    def create_list_from_array(self, arr: List[int]) -> Optional[ListNode]:
        """Create linked list from array"""
        if not arr:
            return None
        
        head = ListNode(arr[0])
        current = head
        
        for val in arr[1:]:
            current.next = ListNode(val)
            current = current.next
        
        return head
    
    def list_to_array(self, head: Optional[ListNode]) -> List[int]:
        """Convert linked list to array"""
        result = []
        current = head
        
        while current:
            result.append(current.val)
            current = current.next
        
        return result
    
    def create_flatten_list(self, matrix: List[List[int]]) -> Optional[FlattenNode]:
        """Create nested list from matrix"""
        if not matrix or not matrix[0]:
            return None
        
        # Create first row
        head = FlattenNode(matrix[0][0])
        current_row = head
        
        # Build first row
        for j in range(1, len(matrix[0])):
            current_row.next = FlattenNode(matrix[0][j])
            current_row = current_row.next
        
        # Build subsequent rows
        for i in range(1, len(matrix)):
            current_row = head
            for j in range(len(matrix[i])):
                # Move to correct position in first row
                for _ in range(j):
                    current_row = current_row.next
                
                # Add down connection
                if j == 0:
                    current_row.down = FlattenNode(matrix[i][j])
                    current_down = current_row.down
                else:
                    current_down.next = FlattenNode(matrix[i][j])
                    current_down = current_down.next
        
        return head

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Linked List Problems Demo ===\n")
    
    llp = LinkedListProblems()
    
    # Example 1: Remove Duplicates
    print("1. Remove Duplicates:")
    
    # Sorted list with duplicates
    sorted_dup = llp.create_list_from_array([1, 1, 2, 3, 3, 4, 5, 5])
    print(f"Sorted with duplicates: {llp.list_to_array(sorted_dup)}")
    
    result1 = llp.remove_duplicates_sorted(sorted_dup)
    print(f"Remove duplicates (sorted): {llp.list_to_array(result1)}")
    
    # Remove all duplicates
    sorted_dup2 = llp.create_list_from_array([1, 2, 3, 3, 4, 4, 5])
    result2 = llp.remove_duplicates_sorted_all(sorted_dup2)
    print(f"Remove all duplicates: {llp.list_to_array(result2)}")
    
    # Unsorted list
    unsorted_dup = llp.create_list_from_array([4, 2, 1, 3, 2, 4])
    print(f"Unsorted with duplicates: {llp.list_to_array(unsorted_dup)}")
    
    result3 = llp.remove_duplicates_unsorted(unsorted_dup)
    print(f"Remove duplicates (unsorted): {llp.list_to_array(result3)}")
    print()
    
    # Example 2: Remove Nth from End
    print("2. Remove Nth from End:")
    
    test_list = llp.create_list_from_array([1, 2, 3, 4, 5])
    print(f"Original list: {llp.list_to_array(test_list)}")
    
    for n in [1, 2, 5]:
        test_copy = llp.create_list_from_array([1, 2, 3, 4, 5])
        result = llp.remove_nth_from_end(test_copy, n)
        print(f"Remove {n}th from end: {llp.list_to_array(result)}")
    
    # Remove multiple
    multi_test = llp.create_list_from_array([1, 2, 3, 4, 5, 6])
    multi_result = llp.remove_multiple_nth_from_end(multi_test, [1, 3])
    print(f"Remove 1st and 3rd from end: {llp.list_to_array(multi_result)}")
    print()
    
    # Example 3: Partition List
    print("3. Partition List:")
    
    partition_list = llp.create_list_from_array([1, 4, 3, 2, 5, 2])
    print(f"Original list: {llp.list_to_array(partition_list)}")
    
    partitioned = llp.partition_list(partition_list, 3)
    print(f"Partition around 3: {llp.list_to_array(partitioned)}")
    
    # Three-way partition
    three_way_list = llp.create_list_from_array([5, 1, 8, 3, 7, 2, 9, 4])
    three_way_result = llp.three_way_partition(three_way_list, 3, 7)
    print(f"Three-way partition (3, 7): {llp.list_to_array(three_way_result)}")
    print()
    
    # Example 4: Add Two Numbers
    print("4. Add Two Numbers:")
    
    # 342 + 465 = 807 (reverse order)
    num1 = llp.create_list_from_array([2, 4, 3])  # 342
    num2 = llp.create_list_from_array([5, 6, 4])  # 465
    
    print(f"Number 1 (reverse): {llp.list_to_array(num1)} -> 342")
    print(f"Number 2 (reverse): {llp.list_to_array(num2)} -> 465")
    
    sum_result = llp.add_two_numbers(num1, num2)
    print(f"Sum (reverse): {llp.list_to_array(sum_result)} -> 807")
    
    # Forward order
    num3 = llp.create_list_from_array([3, 4, 2])  # 342
    num4 = llp.create_list_from_array([4, 6, 5])  # 465
    
    forward_sum = llp.add_two_numbers_forward(num3, num4)
    print(f"Sum (forward): {llp.list_to_array(forward_sum)} -> 807")
    
    # Multiplication
    mul_result = llp.multiply_two_numbers(
        llp.create_list_from_array([2, 3]),  # 23
        llp.create_list_from_array([4, 5])   # 45
    )
    print(f"23 × 45 = {llp._list_to_number(mul_result)}")
    print()
    
    # Example 5: Intersection of Two Lists
    print("5. Intersection of Two Lists:")
    
    # Create intersecting lists
    # List A: 4 -> 1 -> 8 -> 4 -> 5
    # List B: 5 -> 6 -> 1 -> 8 -> 4 -> 5
    #                      ^
    #                  intersection
    
    common = llp.create_list_from_array([8, 4, 5])
    listA = llp.create_list_from_array([4, 1])
    listB = llp.create_list_from_array([5, 6, 1])
    
    # Connect to common part
    currentA = listA
    while currentA.next:
        currentA = currentA.next
    currentA.next = common
    
    currentB = listB
    while currentB.next:
        currentB = currentB.next
    currentB.next = common
    
    print(f"List A: {llp.list_to_array(listA)}")
    print(f"List B: {llp.list_to_array(listB)}")
    
    intersection = llp.get_intersection_node(listA, listB)
    intersection_elegant = llp.get_intersection_node_elegant(listA, listB)
    
    print(f"Intersection value: {intersection.val if intersection else 'None'}")
    print(f"Elegant method: {intersection_elegant.val if intersection_elegant else 'None'}")
    print()
    
    # Example 6: Advanced Problems
    print("6. Advanced Problems:")
    
    # Split list to parts
    split_list = llp.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Original list: {llp.list_to_array(split_list)}")
    
    parts = llp.split_list_to_parts(split_list, 3)
    print(f"Split into 3 parts:")
    for i, part in enumerate(parts):
        print(f"  Part {i + 1}: {llp.list_to_array(part)}")
    
    # Merge in between
    merge_list1 = llp.create_list_from_array([0, 1, 2, 3, 4, 5, 6])
    merge_list2 = llp.create_list_from_array([1000, 1001, 1002])
    
    print(f"\nMerge between positions:")
    print(f"List 1: {llp.list_to_array(merge_list1)}")
    print(f"List 2: {llp.list_to_array(merge_list2)}")
    
    merged_between = llp.merge_in_between(merge_list1, 2, 5, merge_list2)
    print(f"Merge list2 between positions 2-5: {llp.list_to_array(merged_between)}")
    print()
    
    # Example 7: Performance Analysis
    print("7. Performance Analysis:")
    
    # Large list operations
    large_list = llp.create_list_from_array(list(range(1, 1001)))
    print(f"Testing on list with {llp._get_length(large_list)} elements")
    
    # Remove duplicates performance
    dup_large = llp.create_list_from_array([i % 100 for i in range(1000)])
    print(f"List with duplicates length: {llp._get_length(dup_large)}")
    
    dup_removed = llp.remove_duplicates_unsorted(dup_large)
    print(f"After removing duplicates: {llp._get_length(dup_removed)}")
    
    # Partition performance
    random_list = llp.create_list_from_array([i % 50 for i in range(500, 0, -1)])
    partitioned_large = llp.partition_list(random_list, 25)
    print(f"Partitioned large list length: {llp._get_length(partitioned_large)}")
    
    # Add large numbers
    large_num1 = llp.create_list_from_array([9] * 100)  # 999...999 (100 nines)
    large_num2 = llp.create_list_from_array([1] + [0] * 100)  # 100...000 (1 followed by 100 zeros)
    
    large_sum = llp.add_two_numbers(large_num1, large_num2)
    print(f"Sum of large numbers length: {llp._get_length(large_sum)}")
    
    print("\n=== Demo Complete ===") 