"""
23. Merge k Sorted Lists - Multiple Approaches
Difficulty: Hard

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.
"""

from typing import List, Optional
import heapq

# Definition for singly-linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __lt__(self, other):
        return self.val < other.val

class MergeKSortedLists:
    """Multiple approaches to merge k sorted lists"""
    
    def mergeKLists_min_heap(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 1: Min Heap (Optimal)
        
        Use min heap to track smallest elements from all lists.
        
        Time: O(n log k), Space: O(k) where n is total nodes, k is number of lists
        """
        if not lists:
            return None
        
        heap = []
        
        # Initialize heap with first node from each non-empty list
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, (head.val, i, head))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, list_idx, node = heapq.heappop(heap)
            
            # Add node to result
            current.next = node
            current = current.next
            
            # Add next node from same list if exists
            if node.next:
                heapq.heappush(heap, (node.next.val, list_idx, node.next))
        
        return dummy.next
    
    def mergeKLists_divide_conquer(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 2: Divide and Conquer (Optimal)
        
        Merge lists pairwise using divide and conquer.
        
        Time: O(n log k), Space: O(log k)
        """
        if not lists:
            return None
        
        def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            """Merge two sorted lists"""
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            # Append remaining nodes
            current.next = l1 or l2
            
            return dummy.next
        
        def divide_conquer(start: int, end: int) -> Optional[ListNode]:
            """Divide and conquer merge"""
            if start == end:
                return lists[start]
            
            if start > end:
                return None
            
            mid = (start + end) // 2
            left = divide_conquer(start, mid)
            right = divide_conquer(mid + 1, end)
            
            return merge_two_lists(left, right)
        
        return divide_conquer(0, len(lists) - 1)
    
    def mergeKLists_brute_force(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 3: Brute Force
        
        Collect all values, sort, and create new list.
        
        Time: O(n log n), Space: O(n)
        """
        values = []
        
        # Collect all values
        for head in lists:
            current = head
            while current:
                values.append(current.val)
                current = current.next
        
        # Sort values
        values.sort()
        
        # Create new sorted list
        dummy = ListNode(0)
        current = dummy
        
        for val in values:
            current.next = ListNode(val)
            current = current.next
        
        return dummy.next
    
    def mergeKLists_sequential_merge(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 4: Sequential Merge
        
        Merge lists one by one sequentially.
        
        Time: O(n * k), Space: O(1)
        """
        if not lists:
            return None
        
        def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            """Merge two sorted lists"""
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 or l2
            return dummy.next
        
        result = lists[0]
        
        for i in range(1, len(lists)):
            result = merge_two_lists(result, lists[i])
        
        return result
    
    def mergeKLists_priority_queue_nodes(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 5: Priority Queue with Node Objects
        
        Use priority queue directly with node objects.
        
        Time: O(n log k), Space: O(k)
        """
        if not lists:
            return None
        
        heap = []
        
        # Add first node from each list
        for head in lists:
            if head:
                heapq.heappush(heap, head)
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            node = heapq.heappop(heap)
            
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, node.next)
        
        return dummy.next


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create linked list from values"""
    if not values:
        return None
    
    dummy = ListNode(0)
    current = dummy
    
    for val in values:
        current.next = ListNode(val)
        current = current.next
    
    return dummy.next


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Helper function to convert linked list to list"""
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result


def test_merge_k_sorted_lists():
    """Test merge k sorted lists algorithms"""
    solver = MergeKSortedLists()
    
    test_cases = [
        ([[1,4,5],[1,3,4],[2,6]], [1,1,2,3,4,4,5,6], "Example 1"),
        ([], [], "Empty input"),
        ([[]], [], "Single empty list"),
        ([[1]], [1], "Single list with one element"),
        ([[1,2,3],[4,5,6]], [1,2,3,4,5,6], "Two non-overlapping lists"),
        ([[1,3,5],[2,4,6]], [1,2,3,4,5,6], "Two interleaved lists"),
        ([[1],[2],[3]], [1,2,3], "Three single-element lists"),
    ]
    
    algorithms = [
        ("Min Heap", solver.mergeKLists_min_heap),
        ("Divide Conquer", solver.mergeKLists_divide_conquer),
        ("Brute Force", solver.mergeKLists_brute_force),
        ("Sequential Merge", solver.mergeKLists_sequential_merge),
        ("Priority Queue Nodes", solver.mergeKLists_priority_queue_nodes),
    ]
    
    print("=== Testing Merge K Sorted Lists ===")
    
    for list_values, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input lists: {list_values}")
        print(f"Expected: {expected}")
        
        # Create linked lists
        lists = [create_linked_list(values) for values in list_values]
        
        for alg_name, alg_func in algorithms:
            try:
                # Create fresh copies for each algorithm
                lists_copy = [create_linked_list(values) for values in list_values]
                result_head = alg_func(lists_copy)
                result = linked_list_to_list(result_head)
                
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_min_heap_approach():
    """Demonstrate min heap approach step by step"""
    print("\n=== Min Heap Approach Step-by-Step Demo ===")
    
    # Create test lists: [1,4,5], [1,3,4], [2,6]
    list1 = create_linked_list([1, 4, 5])
    list2 = create_linked_list([1, 3, 4])
    list3 = create_linked_list([2, 6])
    
    lists = [list1, list2, list3]
    
    print("Input lists:")
    for i, head in enumerate(lists):
        values = linked_list_to_list(head)
        print(f"  List {i+1}: {values}")
    
    print(f"\nMerging using min heap:")
    
    heap = []
    
    # Initialize heap
    print(f"Initialize heap with first node from each list:")
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
            print(f"  Added: value={head.val} from list {i+1}")
    
    print(f"Initial heap: {[(val, list_idx) for val, list_idx, _ in heap]}")
    
    result = []
    step = 1
    
    while heap:
        print(f"\nStep {step}:")
        
        val, list_idx, node = heapq.heappop(heap)
        result.append(val)
        
        print(f"  Popped: value={val} from list {list_idx+1}")
        print(f"  Added to result: {result}")
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
            print(f"  Added next: value={node.next.val} from list {list_idx+1}")
        
        if heap:
            heap_values = [(val, list_idx) for val, list_idx, _ in heap]
            print(f"  Heap: {heap_values}")
        
        step += 1
    
    print(f"\nFinal merged result: {result}")


def demonstrate_divide_conquer_approach():
    """Demonstrate divide and conquer approach"""
    print("\n=== Divide and Conquer Approach Demonstration ===")
    
    list_values = [[1,4,5], [1,3,4], [2,6], [3,7]]
    
    print(f"Input lists: {list_values}")
    print(f"Divide and conquer merging:")
    
    def visualize_merge_process(lists_info, level=0):
        indent = "  " * level
        
        if len(lists_info) == 1:
            print(f"{indent}Base case: {lists_info[0]}")
            return lists_info[0]
        
        mid = len(lists_info) // 2
        left_lists = lists_info[:mid]
        right_lists = lists_info[mid:]
        
        print(f"{indent}Divide: {lists_info} -> {left_lists} | {right_lists}")
        
        left_result = visualize_merge_process(left_lists, level + 1)
        right_result = visualize_merge_process(right_lists, level + 1)
        
        # Simulate merge
        merged = sorted(left_result + right_result)
        print(f"{indent}Merge: {left_result} + {right_result} -> {merged}")
        
        return merged
    
    result = visualize_merge_process(list_values)
    print(f"\nFinal result: {result}")


def visualize_merge_process():
    """Visualize the merge process"""
    print("\n=== Merge Process Visualization ===")
    
    lists_data = [
        [1, 4, 5],
        [1, 3, 4],
        [2, 6]
    ]
    
    print("Lists to merge:")
    for i, data in enumerate(lists_data):
        print(f"  List {i+1}: {' -> '.join(map(str, data))}")
    
    print(f"\nMerge process using min heap:")
    
    # Simulate heap-based merge
    pointers = [0] * len(lists_data)  # Pointers for each list
    result = []
    
    while any(pointers[i] < len(lists_data[i]) for i in range(len(lists_data))):
        # Find minimum among current elements
        min_val = float('inf')
        min_list = -1
        
        for i in range(len(lists_data)):
            if pointers[i] < len(lists_data[i]):
                if lists_data[i][pointers[i]] < min_val:
                    min_val = lists_data[i][pointers[i]]
                    min_list = i
        
        # Add minimum to result
        result.append(min_val)
        pointers[min_list] += 1
        
        # Show current state
        current_elements = []
        for i in range(len(lists_data)):
            if pointers[i] < len(lists_data[i]):
                current_elements.append(f"L{i+1}:{lists_data[i][pointers[i]]}")
            else:
                current_elements.append(f"L{i+1}:END")
        
        print(f"  Choose {min_val} from List {min_list+1}")
        print(f"    Current: {' | '.join(current_elements)}")
        print(f"    Result: {result}")
    
    print(f"\nFinal merged list: {result}")


if __name__ == "__main__":
    test_merge_k_sorted_lists()
    demonstrate_min_heap_approach()
    demonstrate_divide_conquer_approach()
    visualize_merge_process()

"""
Merge k Sorted Lists demonstrates heap applications for merging multiple
sorted sequences, including divide-and-conquer optimization and multiple
approaches for efficient list merging algorithms.
"""
