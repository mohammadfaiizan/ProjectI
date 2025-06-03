"""
Advanced Linked List Techniques - Complex Operations and Algorithms
This module implements advanced linked list manipulation techniques for interview preparation.
"""

from typing import Optional, Dict, List
import copy

class ListNode:
    """Basic node class for singly linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class RandomListNode:
    """Node class for linked list with random pointers"""
    def __init__(self, val: int = 0, next: Optional['RandomListNode'] = None, 
                 random: Optional['RandomListNode'] = None):
        self.val = val
        self.next = next
        self.random = random
    
    def __repr__(self):
        return f"RandomListNode({self.val})"

class MultiLevelNode:
    """Node class for multi-level linked list"""
    def __init__(self, val: int = 0, prev: Optional['MultiLevelNode'] = None,
                 next: Optional['MultiLevelNode'] = None, child: Optional['MultiLevelNode'] = None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
    
    def __repr__(self):
        return f"MultiLevelNode({self.val})"

class AdvancedLinkedListTechniques:
    """
    Implementation of advanced linked list manipulation techniques
    """
    
    def __init__(self):
        """Initialize advanced techniques class"""
        pass
    
    # ==================== SKIP M DELETE N ====================
    
    def skip_m_delete_n(self, head: Optional[ListNode], m: int, n: int) -> Optional[ListNode]:
        """
        Skip M nodes, then delete N nodes, repeat until end of list
        
        Time Complexity: O(length)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            m: Number of nodes to skip
            n: Number of nodes to delete
        
        Returns:
            Head of modified list
        """
        if not head or m < 0 or n < 0:
            return head
        
        if m == 0:
            # Delete all nodes
            return None
        
        current = head
        
        while current:
            # Skip M nodes
            for _ in range(m - 1):
                if current:
                    current = current.next
                else:
                    return head  # End of list reached
            
            if not current:
                break
            
            # Delete N nodes
            node_to_delete = current.next
            for _ in range(n):
                if node_to_delete:
                    temp = node_to_delete
                    node_to_delete = node_to_delete.next
                    # Optional: explicitly delete temp
                else:
                    break
            
            # Connect the remaining part
            current.next = node_to_delete
            current = node_to_delete
        
        return head
    
    def skip_m_delete_n_advanced(self, head: Optional[ListNode], pattern: List[tuple]) -> Optional[ListNode]:
        """
        Advanced version with different skip/delete patterns
        
        Args:
            head: Head of linked list
            pattern: List of (skip, delete) tuples
        
        Returns:
            Head of modified list
        """
        if not head or not pattern:
            return head
        
        current = head
        pattern_index = 0
        
        while current and pattern_index < len(pattern):
            m, n = pattern[pattern_index]
            
            # Skip M nodes
            for _ in range(m - 1):
                if current:
                    current = current.next
                else:
                    return head
            
            if not current:
                break
            
            # Delete N nodes
            node_to_delete = current.next
            for _ in range(n):
                if node_to_delete:
                    node_to_delete = node_to_delete.next
                else:
                    break
            
            current.next = node_to_delete
            current = node_to_delete
            
            # Move to next pattern (cycle if at end)
            pattern_index = (pattern_index + 1) % len(pattern)
        
        return head
    
    def skip_m_delete_n_recursive(self, head: Optional[ListNode], m: int, n: int) -> Optional[ListNode]:
        """
        Recursive implementation of skip M delete N
        
        Args:
            head: Head of linked list
            m: Number of nodes to skip
            n: Number of nodes to delete
        
        Returns:
            Head of modified list
        """
        if not head:
            return None
        
        if m == 1:
            # Skip current node, delete next N nodes
            current = head
            for _ in range(n):
                if current.next:
                    current.next = current.next.next
                else:
                    break
            
            # Recursively process remaining list
            if current.next:
                current.next = self.skip_m_delete_n_recursive(current.next, m, n)
            
            return head
        else:
            # Skip current node, continue with m-1
            head.next = self.skip_m_delete_n_recursive(head.next, m - 1, n)
            return head
    
    # ==================== FLATTEN MULTILEVEL LINKED LIST ====================
    
    def flatten_multilevel_list(self, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]:
        """
        Flatten a multilevel doubly linked list
        
        Time Complexity: O(n)
        Space Complexity: O(d) where d is maximum depth
        
        Args:
            head: Head of multilevel list
        
        Returns:
            Head of flattened list
        """
        if not head:
            return None
        
        stack = []
        current = head
        
        while current:
            if current.child:
                # If there's a next node, save it for later
                if current.next:
                    stack.append(current.next)
                
                # Connect child as next
                current.next = current.child
                current.child.prev = current
                
                # Clear child pointer
                current.child = None
            
            # If no next node and stack has nodes
            if not current.next and stack:
                next_node = stack.pop()
                current.next = next_node
                next_node.prev = current
            
            current = current.next
        
        return head
    
    def flatten_multilevel_recursive(self, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]:
        """
        Recursive approach to flatten multilevel list
        
        Args:
            head: Head of multilevel list
        
        Returns:
            Head of flattened list and tail of flattened portion
        """
        def flatten_helper(node):
            """Helper function that returns head and tail of flattened portion"""
            if not node:
                return None, None
            
            head = node
            tail = node
            
            while node:
                next_node = node.next
                
                if node.child:
                    # Flatten child recursively
                    child_head, child_tail = flatten_helper(node.child)
                    
                    # Connect child
                    node.next = child_head
                    child_head.prev = node
                    node.child = None
                    
                    # Update tail to child tail
                    tail = child_tail
                    
                    # Connect remaining part
                    if next_node:
                        tail.next = next_node
                        next_node.prev = tail
                
                if next_node:
                    node = next_node
                    tail = node
                else:
                    break
            
            return head, tail
        
        flattened_head, _ = flatten_helper(head)
        return flattened_head
    
    def flatten_multilevel_iterative_dfs(self, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]:
        """
        Iterative DFS approach using explicit stack
        
        Args:
            head: Head of multilevel list
        
        Returns:
            Head of flattened list
        """
        if not head:
            return None
        
        stack = [head]
        prev = None
        
        while stack:
            current = stack.pop()
            
            if prev:
                prev.next = current
                current.prev = prev
            
            # Push next and child to stack (next first, then child for DFS order)
            if current.next:
                stack.append(current.next)
            
            if current.child:
                stack.append(current.child)
                current.child = None  # Clear child pointer
            
            prev = current
        
        return head
    
    # ==================== COPY LIST WITH RANDOM POINTERS ====================
    
    def copy_random_list(self, head: Optional[RandomListNode]) -> Optional[RandomListNode]:
        """
        Deep copy linked list with random pointers
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of original list
        
        Returns:
            Head of copied list
        """
        if not head:
            return None
        
        # Phase 1: Create copy nodes and interweave them
        current = head
        while current:
            copy_node = RandomListNode(current.val)
            copy_node.next = current.next
            current.next = copy_node
            current = copy_node.next
        
        # Phase 2: Set random pointers for copy nodes
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next
        
        # Phase 3: Separate the two lists
        dummy = RandomListNode(0)
        copy_prev = dummy
        current = head
        
        while current:
            copy_node = current.next
            
            # Restore original list
            current.next = copy_node.next
            
            # Build copy list
            copy_prev.next = copy_node
            copy_prev = copy_node
            
            current = current.next
        
        return dummy.next
    
    def copy_random_list_hashmap(self, head: Optional[RandomListNode]) -> Optional[RandomListNode]:
        """
        Copy using hashmap approach
        
        Args:
            head: Head of original list
        
        Returns:
            Head of copied list
        """
        if not head:
            return None
        
        # Create mapping from original to copy nodes
        node_map = {}
        
        # First pass: create all copy nodes
        current = head
        while current:
            node_map[current] = RandomListNode(current.val)
            current = current.next
        
        # Second pass: set next and random pointers
        current = head
        while current:
            copy_node = node_map[current]
            
            if current.next:
                copy_node.next = node_map[current.next]
            
            if current.random:
                copy_node.random = node_map[current.random]
            
            current = current.next
        
        return node_map[head]
    
    def copy_random_list_recursive(self, head: Optional[RandomListNode]) -> Optional[RandomListNode]:
        """
        Recursive approach with memoization
        
        Args:
            head: Head of original list
        
        Returns:
            Head of copied list
        """
        def copy_helper(node, memo):
            if not node:
                return None
            
            if node in memo:
                return memo[node]
            
            # Create copy node
            copy_node = RandomListNode(node.val)
            memo[node] = copy_node
            
            # Recursively set pointers
            copy_node.next = copy_helper(node.next, memo)
            copy_node.random = copy_helper(node.random, memo)
            
            return copy_node
        
        return copy_helper(head, {})
    
    # ==================== ROTATE LINKED LIST ====================
    
    def rotate_right(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Rotate linked list to the right by k positions
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            k: Number of positions to rotate right
        
        Returns:
            Head of rotated list
        """
        if not head or not head.next or k == 0:
            return head
        
        # Get length and make circular
        length = 1
        tail = head
        while tail.next:
            tail = tail.next
            length += 1
        
        tail.next = head  # Make circular
        
        # Find new tail (length - k % length - 1 steps from head)
        k = k % length
        steps_to_new_tail = length - k
        
        new_tail = head
        for _ in range(steps_to_new_tail - 1):
            new_tail = new_tail.next
        
        new_head = new_tail.next
        new_tail.next = None  # Break the circle
        
        return new_head
    
    def rotate_left(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Rotate linked list to the left by k positions
        
        Args:
            head: Head of linked list
            k: Number of positions to rotate left
        
        Returns:
            Head of rotated list
        """
        if not head or not head.next or k == 0:
            return head
        
        # Get length
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # Normalize k
        k = k % length
        if k == 0:
            return head
        
        # Find new tail (k-1 steps from head)
        new_tail = head
        for _ in range(k - 1):
            new_tail = new_tail.next
        
        new_head = new_tail.next
        new_tail.next = None
        
        # Find end of new list and connect
        current = new_head
        while current.next:
            current = current.next
        
        current.next = head
        
        return new_head
    
    def rotate_by_groups(self, head: Optional[ListNode], k: int, group_size: int) -> Optional[ListNode]:
        """
        Rotate each group of nodes by k positions
        
        Args:
            head: Head of linked list
            k: Number of positions to rotate each group
            group_size: Size of each group
        
        Returns:
            Head of modified list
        """
        if not head or k == 0 or group_size <= 1:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        prev_group_end = dummy
        
        while True:
            # Check if we have enough nodes for a group
            group_start = prev_group_end.next
            if not group_start:
                break
            
            # Find group end
            group_end = group_start
            for _ in range(group_size - 1):
                if group_end.next:
                    group_end = group_end.next
                else:
                    # Not enough nodes for a complete group
                    return dummy.next
            
            next_group_start = group_end.next
            group_end.next = None  # Temporarily disconnect
            
            # Convert group to array for easy rotation
            group_nodes = []
            current = group_start
            while current:
                group_nodes.append(current)
                current = current.next
            
            # Rotate array
            k_normalized = k % len(group_nodes)
            if k_normalized > 0:
                rotated_nodes = group_nodes[-k_normalized:] + group_nodes[:-k_normalized]
            else:
                rotated_nodes = group_nodes
            
            # Reconnect rotated group
            for i in range(len(rotated_nodes)):
                if i < len(rotated_nodes) - 1:
                    rotated_nodes[i].next = rotated_nodes[i + 1]
                else:
                    rotated_nodes[i].next = None
            
            # Connect with previous group
            prev_group_end.next = rotated_nodes[0]
            rotated_nodes[-1].next = next_group_start
            
            # Update for next iteration
            prev_group_end = rotated_nodes[-1]
        
        return dummy.next
    
    # ==================== SWAP NODES IN PAIRS ====================
    
    def swap_pairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Swap every two adjacent nodes
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while prev.next and prev.next.next:
            # Identify nodes to swap
            first = prev.next
            second = prev.next.next
            
            # Swap
            prev.next = second
            first.next = second.next
            second.next = first
            
            # Move prev to end of swapped pair
            prev = first
        
        return dummy.next
    
    def swap_pairs_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Recursive approach to swap pairs
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        if not head or not head.next:
            return head
        
        # Save second node
        second = head.next
        
        # Recursively swap the rest
        head.next = self.swap_pairs_recursive(second.next)
        
        # Swap first two nodes
        second.next = head
        
        return second
    
    def swap_k_nodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Swap every k consecutive nodes
        
        Args:
            head: Head of linked list
            k: Number of nodes to swap at a time
        
        Returns:
            Head of modified list
        """
        if not head or k <= 1:
            return head
        
        # Count nodes
        count = 0
        current = head
        while current:
            count += 1
            current = current.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while count >= k:
            # Collect k nodes
            nodes = []
            current = prev.next
            
            for _ in range(k):
                nodes.append(current)
                current = current.next
            
            # Reverse the k nodes
            nodes.reverse()
            
            # Reconnect
            prev.next = nodes[0]
            for i in range(len(nodes) - 1):
                nodes[i].next = nodes[i + 1]
            nodes[-1].next = current
            
            # Update for next iteration
            prev = nodes[-1]
            count -= k
        
        return dummy.next
    
    def swap_alternate_pairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Swap alternate pairs: swap (1,2), skip (3,4), swap (5,6), etc.
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        swap_next = True
        
        while prev.next and prev.next.next:
            if swap_next:
                # Swap the pair
                first = prev.next
                second = prev.next.next
                
                prev.next = second
                first.next = second.next
                second.next = first
                
                prev = first
            else:
                # Skip the pair
                prev = prev.next.next
            
            swap_next = not swap_next
        
        return dummy.next
    
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
    
    def create_random_list(self, values: List[int], random_indices: List[Optional[int]]) -> Optional[RandomListNode]:
        """Create linked list with random pointers"""
        if not values:
            return None
        
        # Create nodes
        nodes = [RandomListNode(val) for val in values]
        
        # Set next pointers
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i + 1]
        
        # Set random pointers
        for i, random_idx in enumerate(random_indices):
            if random_idx is not None and 0 <= random_idx < len(nodes):
                nodes[i].random = nodes[random_idx]
        
        return nodes[0]
    
    def print_random_list(self, head: Optional[RandomListNode]) -> List[str]:
        """Print linked list with random pointers"""
        result = []
        current = head
        node_map = {}
        
        # First pass: assign indices
        temp = head
        index = 0
        while temp:
            node_map[temp] = index
            temp = temp.next
            index += 1
        
        # Second pass: create string representation
        while current:
            random_idx = node_map.get(current.random, None) if current.random else None
            result.append(f"Node({current.val}, random -> {random_idx})")
            current = current.next
        
        return result
    
    def create_multilevel_list(self, structure: Dict) -> Optional[MultiLevelNode]:
        """Create multilevel list from structure definition"""
        if not structure:
            return None
        
        # This is a simplified implementation
        # In practice, you'd need more complex logic to handle arbitrary structures
        nodes = {}
        
        # Create all nodes first
        for node_id, node_data in structure.items():
            nodes[node_id] = MultiLevelNode(node_data['val'])
        
        # Set up connections
        for node_id, node_data in structure.items():
            node = nodes[node_id]
            
            if 'next' in node_data and node_data['next']:
                node.next = nodes[node_data['next']]
            
            if 'prev' in node_data and node_data['prev']:
                node.prev = nodes[node_data['prev']]
            
            if 'child' in node_data and node_data['child']:
                node.child = nodes[node_data['child']]
        
        # Return the head (assuming it's the first node)
        return nodes[list(structure.keys())[0]]

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Advanced Linked List Techniques Demo ===\n")
    
    alt = AdvancedLinkedListTechniques()
    
    # Example 1: Skip M Delete N
    print("1. Skip M Delete N:")
    
    # Create test list: 1->2->3->4->5->6->7->8->9->10
    test_list = alt.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Original list: {alt.list_to_array(test_list)}")
    
    # Skip 2, Delete 3
    result1 = alt.skip_m_delete_n(test_list, 2, 3)
    print(f"Skip 2, Delete 3: {alt.list_to_array(result1)}")
    
    # Advanced pattern: [(2,1), (1,2)] - skip 2 delete 1, then skip 1 delete 2, repeat
    test_list2 = alt.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result2 = alt.skip_m_delete_n_advanced(test_list2, [(2, 1), (1, 2)])
    print(f"Pattern [(2,1), (1,2)]: {alt.list_to_array(result2)}")
    
    # Recursive approach
    test_list3 = alt.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    result3 = alt.skip_m_delete_n_recursive(test_list3, 3, 2)
    print(f"Recursive Skip 3, Delete 2: {alt.list_to_array(result3)}")
    print()
    
    # Example 2: Copy Random List
    print("2. Copy List with Random Pointers:")
    
    # Create random list: [1,2,3] with random pointers
    random_list = alt.create_random_list([1, 2, 3], [2, 0, None])
    print("Original random list:")
    for line in alt.print_random_list(random_list):
        print(f"  {line}")
    
    # Copy the list
    copied_list = alt.copy_random_list(random_list)
    print("Copied random list:")
    for line in alt.print_random_list(copied_list):
        print(f"  {line}")
    
    # Test with hashmap approach
    random_list2 = alt.create_random_list([7, 13, 11, 10, 1], [None, 0, 4, 2, 0])
    copied_list2 = alt.copy_random_list_hashmap(random_list2)
    print(f"Complex example - Original values: {[node.val for node in [random_list2] if node]}")
    print(f"Complex example - Copied values: {[node.val for node in [copied_list2] if node]}")
    print()
    
    # Example 3: Rotate Linked List
    print("3. Rotate Linked List:")
    
    # Create test list: 1->2->3->4->5
    rotate_list = alt.create_list_from_array([1, 2, 3, 4, 5])
    print(f"Original list: {alt.list_to_array(rotate_list)}")
    
    # Rotate right by 2
    rotated_right = alt.rotate_right(rotate_list, 2)
    print(f"Rotate right by 2: {alt.list_to_array(rotated_right)}")
    
    # Rotate left by 2
    rotate_list2 = alt.create_list_from_array([1, 2, 3, 4, 5])
    rotated_left = alt.rotate_left(rotate_list2, 2)
    print(f"Rotate left by 2: {alt.list_to_array(rotated_left)}")
    
    # Rotate by groups
    rotate_list3 = alt.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    rotated_groups = alt.rotate_by_groups(rotate_list3, 1, 3)
    print(f"Rotate groups of 3 by 1: {alt.list_to_array(rotated_groups)}")
    print()
    
    # Example 4: Swap Nodes in Pairs
    print("4. Swap Nodes in Pairs:")
    
    # Swap pairs
    swap_list = alt.create_list_from_array([1, 2, 3, 4, 5, 6])
    print(f"Original list: {alt.list_to_array(swap_list)}")
    
    swapped_pairs = alt.swap_pairs(swap_list)
    print(f"Swap pairs: {alt.list_to_array(swapped_pairs)}")
    
    # Swap k nodes
    swap_list2 = alt.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    swapped_k = alt.swap_k_nodes(swap_list2, 3)
    print(f"Swap groups of 3: {alt.list_to_array(swapped_k)}")
    
    # Swap alternate pairs
    swap_list3 = alt.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    swapped_alternate = alt.swap_alternate_pairs(swap_list3)
    print(f"Swap alternate pairs: {alt.list_to_array(swapped_alternate)}")
    
    # Recursive swap pairs
    swap_list4 = alt.create_list_from_array([1, 2, 3, 4, 5])
    swapped_recursive = alt.swap_pairs_recursive(swap_list4)
    print(f"Recursive swap pairs: {alt.list_to_array(swapped_recursive)}")
    print()
    
    # Example 5: Performance Analysis
    print("5. Performance Analysis:")
    
    # Large list operations
    large_list = alt.create_list_from_array(list(range(1, 1001)))
    print(f"Testing on list with {len(alt.list_to_array(large_list))} elements")
    
    # Skip delete on large list
    skip_delete_large = alt.skip_m_delete_n(large_list, 5, 2)
    remaining_elements = len(alt.list_to_array(skip_delete_large))
    print(f"After skip 5 delete 2: {remaining_elements} elements remaining")
    
    # Rotation on large list
    large_list2 = alt.create_list_from_array(list(range(1, 501)))
    rotated_large = alt.rotate_right(large_list2, 100)
    rotated_sample = alt.list_to_array(rotated_large)[:10]
    print(f"Large list rotate right 100 - first 10 elements: {rotated_sample}")
    
    # Swap operations on large list
    large_list3 = alt.create_list_from_array(list(range(1, 201)))
    swapped_large = alt.swap_pairs(large_list3)
    swap_sample = alt.list_to_array(swapped_large)[:10]
    print(f"Large list swap pairs - first 10 elements: {swap_sample}")
    
    # Complex random list copy
    complex_values = list(range(100))
    complex_random_indices = [i % 100 for i in range(0, 100, 7)]  # Every 7th index
    complex_random_list = alt.create_random_list(complex_values, complex_random_indices)
    
    # Time different copy methods (conceptually)
    copied_interweave = alt.copy_random_list(complex_random_list)
    copied_hashmap = alt.copy_random_list_hashmap(complex_random_list)
    copied_recursive = alt.copy_random_list_recursive(complex_random_list)
    
    print(f"Complex random list copied successfully with all 3 methods")
    print(f"Original first 5 values: {[complex_random_list.val] + [complex_random_list.next.val if complex_random_list.next else None][:4]}")
    
    print("\n=== Demo Complete ===") 