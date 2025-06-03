"""
Linked List Reversal Techniques - Comprehensive Reversal Algorithms
This module implements all types of linked list reversal techniques with detailed explanations.
"""

from typing import Optional, List, Tuple
from collections import deque

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class LinkedListReversal:
    """
    Comprehensive implementation of linked list reversal techniques
    """
    
    def __init__(self):
        """Initialize reversal solver"""
        pass
    
    # ==================== BASIC REVERSAL TECHNIQUES ====================
    
    def reverse_iterative(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse entire linked list iteratively
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            New head of reversed list
        """
        prev = None
        current = head
        
        while current:
            next_temp = current.next  # Store next node
            current.next = prev       # Reverse the link
            prev = current           # Move prev forward
            current = next_temp      # Move current forward
        
        return prev
    
    def reverse_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse entire linked list recursively
        
        Time Complexity: O(n)
        Space Complexity: O(n) - due to recursion stack
        
        Args:
            head: Head of linked list
        
        Returns:
            New head of reversed list
        """
        # Base case
        if not head or not head.next:
            return head
        
        # Recursively reverse the rest of the list
        new_head = self.reverse_recursive(head.next)
        
        # Reverse the current connection
        head.next.next = head
        head.next = None
        
        return new_head
    
    def reverse_with_stack(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse linked list using stack
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of linked list
        
        Returns:
            New head of reversed list
        """
        if not head:
            return None
        
        # Push all nodes to stack
        stack = []
        current = head
        
        while current:
            stack.append(current)
            current = current.next
        
        # Pop nodes and rebuild connections
        new_head = stack.pop()
        current = new_head
        
        while stack:
            current.next = stack.pop()
            current = current.next
        
        current.next = None
        return new_head
    
    def reverse_step_by_step(self, head: Optional[ListNode]) -> Tuple[Optional[ListNode], List[List[int]]]:
        """
        Reverse list and return step-by-step visualization
        
        Args:
            head: Head of linked list
        
        Returns:
            Tuple of (new_head, steps)
        """
        steps = []
        
        def list_to_array(node):
            arr = []
            while node:
                arr.append(node.val)
                node = node.next
            return arr
        
        # Initial state
        steps.append(list_to_array(head))
        
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
            
            # Record step
            steps.append(list_to_array(prev))
        
        return prev, steps
    
    # ==================== REVERSE BETWEEN POSITIONS ====================
    
    def reverse_between(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        """
        Reverse nodes between positions left and right (1-indexed)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            left: Start position (1-indexed)
            right: End position (1-indexed)
        
        Returns:
            Head of modified list
        """
        if not head or left == right:
            return head
        
        # Create dummy node for easier handling
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # Move to position before left
        for _ in range(left - 1):
            prev = prev.next
        
        # Start of reversal
        current = prev.next
        
        # Reverse the sublist
        for _ in range(right - left):
            next_temp = current.next
            current.next = next_temp.next
            next_temp.next = prev.next
            prev.next = next_temp
        
        return dummy.next
    
    def reverse_between_recursive(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        """
        Reverse between positions using recursion
        
        Args:
            head: Head of linked list
            left: Start position (1-indexed)
            right: End position (1-indexed)
        
        Returns:
            Head of modified list
        """
        def reverse_n(node, n):
            """Reverse first n nodes"""
            if n == 1:
                return node
            
            new_head = reverse_n(node.next, n - 1)
            successor = node.next.next
            node.next.next = node
            node.next = successor
            
            return new_head
        
        if left == 1:
            return reverse_n(head, right)
        
        head.next = self.reverse_between_recursive(head.next, left - 1, right - 1)
        return head
    
    def reverse_sublist_with_positions(self, head: Optional[ListNode], positions: List[Tuple[int, int]]) -> Optional[ListNode]:
        """
        Reverse multiple sublists given list of (start, end) positions
        
        Args:
            head: Head of linked list
            positions: List of (start, end) tuples (1-indexed)
        
        Returns:
            Head of modified list
        """
        # Sort positions by start position
        positions.sort()
        
        current_head = head
        
        for left, right in positions:
            current_head = self.reverse_between(current_head, left, right)
        
        return current_head
    
    # ==================== REVERSE IN K-GROUPS ====================
    
    def reverse_k_group(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Reverse nodes in groups of k (LeetCode 25)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            k: Group size
        
        Returns:
            Head of modified list
        """
        if not head or k == 1:
            return head
        
        # Check if we have k nodes to reverse
        def has_k_nodes(node, k):
            count = 0
            while node and count < k:
                node = node.next
                count += 1
            return count == k
        
        if not has_k_nodes(head, k):
            return head
        
        # Reverse first k nodes
        prev = None
        current = head
        
        for _ in range(k):
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        # Recursively reverse remaining groups
        head.next = self.reverse_k_group(current, k)
        
        return prev
    
    def reverse_k_group_iterative(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Reverse in k-groups iteratively
        
        Args:
            head: Head of linked list
            k: Group size
        
        Returns:
            Head of modified list
        """
        if not head or k == 1:
            return head
        
        # Count total nodes
        count = 0
        current = head
        while current:
            count += 1
            current = current.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev_group_end = dummy
        
        while count >= k:
            group_start = prev_group_end.next
            group_end = group_start
            
            # Find end of current group
            for _ in range(k - 1):
                group_end = group_end.next
            
            next_group_start = group_end.next
            
            # Reverse current group
            self._reverse_segment(group_start, group_end)
            
            # Connect with previous group
            prev_group_end.next = group_end
            group_start.next = next_group_start
            
            # Update for next iteration
            prev_group_end = group_start
            count -= k
        
        return dummy.next
    
    def _reverse_segment(self, start: ListNode, end: ListNode) -> None:
        """Helper method to reverse segment between start and end nodes"""
        prev = None
        current = start
        
        while prev != end:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
    
    def reverse_alternate_k_groups(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Reverse alternate k-groups (reverse 1st, skip 2nd, reverse 3rd, etc.)
        
        Args:
            head: Head of linked list
            k: Group size
        
        Returns:
            Head of modified list
        """
        if not head or k == 1:
            return head
        
        # Count nodes in current group
        count = 0
        current = head
        while current and count < k:
            current = current.next
            count += 1
        
        # If we have k nodes, reverse them
        if count == k:
            current = self.reverse_k_group(head, k)
            
            # Skip next k nodes
            for _ in range(k):
                if current:
                    current = current.next
            
            # Recursively process remaining
            if current:
                current.next = self.reverse_alternate_k_groups(current.next, k)
        
        return head
    
    def reverse_k_group_with_remainder(self, head: Optional[ListNode], k: int, reverse_remainder: bool = False) -> Optional[ListNode]:
        """
        Reverse in k-groups with option to reverse remainder
        
        Args:
            head: Head of linked list
            k: Group size
            reverse_remainder: Whether to reverse remaining nodes if less than k
        
        Returns:
            Head of modified list
        """
        if not head or k == 1:
            return head
        
        # Count total nodes
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # Process complete groups
        while length >= k:
            # Reverse k nodes
            group_prev = prev
            current = prev.next
            
            for _ in range(k):
                next_temp = current.next
                current.next = group_prev.next
                group_prev.next = current
                current = next_temp
            
            # Move prev to end of reversed group
            for _ in range(k):
                prev = prev.next
            
            prev.next = current
            length -= k
        
        # Handle remainder
        if reverse_remainder and length > 0:
            prev.next = self.reverse_iterative(current)
        
        return dummy.next
    
    # ==================== ADVANCED REVERSAL TECHNIQUES ====================
    
    def reverse_in_pairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse nodes in pairs (swap every two adjacent nodes)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        if not head or not head.next:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while prev.next and prev.next.next:
            first = prev.next
            second = prev.next.next
            
            # Swap
            prev.next = second
            first.next = second.next
            second.next = first
            
            # Move prev to end of swapped pair
            prev = first
        
        return dummy.next
    
    def reverse_in_pairs_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse in pairs recursively
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        if not head or not head.next:
            return head
        
        # Save the second node
        second = head.next
        
        # Recursively reverse the rest
        head.next = self.reverse_in_pairs_recursive(second.next)
        
        # Reverse current pair
        second.next = head
        
        return second
    
    def reverse_nodes_at_even_positions(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse only nodes at even positions (1-indexed)
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        if not head or not head.next:
            return head
        
        # Extract even positioned nodes
        even_head = None
        odd_current = head
        position = 1
        
        while odd_current and odd_current.next:
            even_node = odd_current.next
            odd_current.next = even_node.next
            
            # Add to even list (reverse order)
            even_node.next = even_head
            even_head = even_node
            
            if odd_current.next:
                odd_current = odd_current.next
            else:
                break
            position += 2
        
        # Merge odd and even lists
        if not even_head:
            return head
        
        # Find insertion points and merge
        result = ListNode(0)
        current = result
        odd_current = head
        even_current = even_head
        position = 1
        
        while odd_current or even_current:
            if position % 2 == 1:  # Odd position
                if odd_current:
                    current.next = odd_current
                    odd_current = odd_current.next
            else:  # Even position
                if even_current:
                    current.next = even_current
                    even_current = even_current.next
            
            current = current.next
            position += 1
        
        return result.next
    
    def reverse_zigzag(self, head: Optional[ListNode], pattern: str) -> Optional[ListNode]:
        """
        Reverse in zigzag pattern based on given pattern
        
        Args:
            head: Head of linked list
            pattern: String of 'R' (reverse) and 'S' (skip) operations
        
        Returns:
            Head of modified list
        """
        if not head or not pattern:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        pattern_index = 0
        
        while current:
            if pattern[pattern_index % len(pattern)] == 'R':
                # Reverse next two nodes if available
                if current and current.next:
                    first = current
                    second = current.next
                    
                    prev.next = second
                    first.next = second.next
                    second.next = first
                    
                    prev = first
                    current = first.next
                else:
                    prev = current
                    current = current.next if current else None
            else:  # Skip
                prev = current
                current = current.next if current else None
            
            pattern_index += 1
        
        return dummy.next
    
    def reverse_spiral(self, head: Optional[ListNode], levels: int) -> Optional[ListNode]:
        """
        Reverse in spiral pattern (reverse 1, skip 1, reverse 2, skip 2, etc.)
        
        Args:
            head: Head of linked list
            levels: Number of spiral levels
        
        Returns:
            Head of modified list
        """
        if not head or levels <= 0:
            return head
        
        current = head
        level = 1
        
        while current and level <= levels:
            # Reverse 'level' number of nodes
            if level % 2 == 1:  # Odd levels - reverse
                group_end = current
                for _ in range(level - 1):
                    if group_end and group_end.next:
                        group_end = group_end.next
                    else:
                        break
                
                if group_end:
                    next_group = group_end.next
                    group_end.next = None
                    
                    # Reverse current group
                    reversed_head = self.reverse_iterative(current)
                    
                    # Find end of reversed group
                    reversed_end = reversed_head
                    while reversed_end.next:
                        reversed_end = reversed_end.next
                    
                    reversed_end.next = next_group
                    current = next_group
            else:  # Even levels - skip
                for _ in range(level):
                    if current:
                        current = current.next
            
            level += 1
        
        return head
    
    # ==================== UTILITY METHODS ====================
    
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
    
    def print_list_structure(self, head: Optional[ListNode], title: str = "") -> None:
        """Print visual representation of list"""
        if title:
            print(f"\n{title}:")
        
        if not head:
            print("Empty List")
            return
        
        values = self.list_to_array(head)
        print(" -> ".join(map(str, values)) + " -> NULL")
    
    def compare_lists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> bool:
        """Compare two linked lists for equality"""
        while list1 and list2:
            if list1.val != list2.val:
                return False
            list1 = list1.next
            list2 = list2.next
        
        return list1 is None and list2 is None
    
    def get_list_length(self, head: Optional[ListNode]) -> int:
        """Get length of linked list"""
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        return length

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Linked List Reversal Techniques Demo ===\n")
    
    reverser = LinkedListReversal()
    
    # Example 1: Basic Reversal Techniques
    print("1. Basic Reversal Techniques:")
    
    original_list = reverser.create_list_from_array([1, 2, 3, 4, 5])
    reverser.print_list_structure(original_list, "Original List")
    
    # Iterative reversal
    iter_list = reverser.create_list_from_array([1, 2, 3, 4, 5])
    iter_reversed = reverser.reverse_iterative(iter_list)
    reverser.print_list_structure(iter_reversed, "Iterative Reversal")
    
    # Recursive reversal
    rec_list = reverser.create_list_from_array([1, 2, 3, 4, 5])
    rec_reversed = reverser.reverse_recursive(rec_list)
    reverser.print_list_structure(rec_reversed, "Recursive Reversal")
    
    # Stack-based reversal
    stack_list = reverser.create_list_from_array([1, 2, 3, 4, 5])
    stack_reversed = reverser.reverse_with_stack(stack_list)
    reverser.print_list_structure(stack_reversed, "Stack-based Reversal")
    
    # Step-by-step reversal
    step_list = reverser.create_list_from_array([1, 2, 3, 4])
    step_reversed, steps = reverser.reverse_step_by_step(step_list)
    print("\nStep-by-step Reversal Process:")
    for i, step in enumerate(steps):
        print(f"  Step {i}: {step}")
    print()
    
    # Example 2: Reverse Between Positions
    print("2. Reverse Between Positions:")
    
    between_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7])
    reverser.print_list_structure(between_list, "Original List")
    
    # Reverse between positions 2 and 5
    between_reversed = reverser.reverse_between(between_list, 2, 5)
    reverser.print_list_structure(between_reversed, "Reversed between positions 2-5")
    
    # Multiple position reversals
    multi_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    positions = [(2, 4), (6, 8)]
    multi_reversed = reverser.reverse_sublist_with_positions(multi_list, positions)
    reverser.print_list_structure(multi_reversed, f"Reversed at positions {positions}")
    print()
    
    # Example 3: Reverse in K-Groups
    print("3. Reverse in K-Groups:")
    
    k_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    reverser.print_list_structure(k_list, "Original List")
    
    # Reverse in groups of 3
    k_reversed = reverser.reverse_k_group(k_list, 3)
    reverser.print_list_structure(k_reversed, "Reversed in groups of 3")
    
    # Reverse alternate groups
    alt_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    alt_reversed = reverser.reverse_alternate_k_groups(alt_list, 3)
    reverser.print_list_structure(alt_reversed, "Reverse alternate groups of 3")
    
    # Reverse with remainder
    rem_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7])
    rem_reversed = reverser.reverse_k_group_with_remainder(rem_list, 3, reverse_remainder=True)
    reverser.print_list_structure(rem_reversed, "Reverse groups of 3 with remainder")
    print()
    
    # Example 4: Advanced Reversal Techniques
    print("4. Advanced Reversal Techniques:")
    
    # Reverse in pairs
    pair_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6])
    reverser.print_list_structure(pair_list, "Original List")
    
    pair_reversed = reverser.reverse_in_pairs(pair_list)
    reverser.print_list_structure(pair_reversed, "Reversed in pairs")
    
    # Reverse even positions
    even_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    even_reversed = reverser.reverse_nodes_at_even_positions(even_list)
    reverser.print_list_structure(even_reversed, "Reversed even position nodes")
    
    # Zigzag reversal
    zigzag_list = reverser.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    zigzag_reversed = reverser.reverse_zigzag(zigzag_list, "RSRS")
    reverser.print_list_structure(zigzag_reversed, "Zigzag reversal (RSRS pattern)")
    print()
    
    # Example 5: Performance Comparison
    print("5. Performance Comparison:")
    
    # Test different methods on same list
    test_array = list(range(1, 11))
    print(f"Test array: {test_array}")
    
    methods = [
        ("Iterative", reverser.reverse_iterative),
        ("Recursive", reverser.reverse_recursive),
        ("Stack-based", reverser.reverse_with_stack),
    ]
    
    for method_name, method in methods:
        test_list = reverser.create_list_from_array(test_array)
        reversed_list = method(test_list)
        result_array = reverser.list_to_array(reversed_list)
        print(f"  {method_name}: {result_array}")
    
    # Test k-group methods
    print("\nK-group reversal methods (k=3):")
    k_methods = [
        ("Recursive K-group", lambda x: reverser.reverse_k_group(x, 3)),
        ("Iterative K-group", lambda x: reverser.reverse_k_group_iterative(x, 3)),
        ("K-group with remainder", lambda x: reverser.reverse_k_group_with_remainder(x, 3, True)),
    ]
    
    for method_name, method in k_methods:
        test_list = reverser.create_list_from_array(test_array)
        reversed_list = method(test_list)
        result_array = reverser.list_to_array(reversed_list)
        print(f"  {method_name}: {result_array}")
    print()
    
    # Example 6: Large List Performance
    print("6. Large List Performance Test:")
    
    # Create large list
    large_array = list(range(1, 1001))
    large_list = reverser.create_list_from_array(large_array)
    
    print(f"Testing reversal on list of {len(large_array)} elements...")
    
    # Test iterative reversal
    large_reversed = reverser.reverse_iterative(large_list)
    reversed_array = reverser.list_to_array(large_reversed)
    
    print(f"Original first 10: {large_array[:10]}")
    print(f"Reversed first 10: {reversed_array[:10]}")
    print(f"Original last 10: {large_array[-10:]}")
    print(f"Reversed last 10: {reversed_array[-10:]}")
    
    # Verify reversal correctness
    is_correct = reversed_array == large_array[::-1]
    print(f"Reversal correctness: {'✓' if is_correct else '✗'}")
    
    # Test k-group on large list
    large_k_list = reverser.create_list_from_array(list(range(1, 101)))
    large_k_reversed = reverser.reverse_k_group(large_k_list, 5)
    large_k_array = reverser.list_to_array(large_k_reversed)
    
    print(f"K-group (k=5) on 100 elements:")
    print(f"  First 20 elements: {large_k_array[:20]}")
    
    print("\n=== Demo Complete ===") 