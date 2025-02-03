class Heap:
    def __init__(self, is_min_heap=True):
        """
        Initialize a Heap.
        :param is_min_heap: True for Min-Heap, False for Max-Heap.
        Time Complexity: O(1)
        """
        self.heap = []
        self.is_min_heap = is_min_heap

    def _compare(self, parent, child):
        """
        Compare two elements based on heap type.
        Time Complexity: O(1)
        """
        return parent > child if self.is_min_heap else parent < child

    def insert(self, value):
        """
        Insert a value into the heap.
        :param value: Value to insert.
        Time Complexity: O(log n)
        """
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def _sift_up(self, index):
        """
        Sift up the element at the given index to maintain heap property.
        :param index: Index to sift up.
        Time Complexity: O(log n)
        """
        parent = (index - 1) // 2
        while index > 0 and self._compare(self.heap[parent], self.heap[index]):
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent
            parent = (index - 1) // 2

    def extract_top(self):
        """
        Remove and return the top element of the heap.
        Time Complexity: O(log n)
        """
        if not self.heap:
            return None
        top = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return top

    def _sift_down(self, index):
        """
        Sift down the element at the given index to maintain heap property.
        :param index: Index to sift down.
        Time Complexity: O(log n)
        """
        size = len(self.heap)
        while index < size:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest_or_largest = index

            if left < size and self._compare(self.heap[smallest_or_largest], self.heap[left]):
                smallest_or_largest = left

            if right < size and self._compare(self.heap[smallest_or_largest], self.heap[right]):
                smallest_or_largest = right

            if smallest_or_largest == index:
                break

            self.heap[index], self.heap[smallest_or_largest] = self.heap[smallest_or_largest], self.heap[index]
            index = smallest_or_largest

    def build_heap(self, elements):
        """
        Build a heap from a list of elements.
        :param elements: List of elements.
        Time Complexity: O(n)
        """
        self.heap = elements[:]
        for i in range((len(self.heap) - 2) // 2, -1, -1):
            self._sift_down(i)

    def heap_sort(self):
        """
        Perform heap sort and return the sorted elements.
        Time Complexity: O(n log n)
        """
        original_heap = self.heap[:]
        sorted_elements = []
        while self.heap:
            sorted_elements.append(self.extract_top())
        self.heap = original_heap
        return sorted_elements

    def peek(self):
        """
        Get the top element of the heap without removing it.
        Time Complexity: O(1)
        """
        return self.heap[0] if self.heap else None

    def print_heap(self):
        """
        Print the current state of the heap.
        Time Complexity: O(n)
        """
        print(self.heap)


# Main function to demonstrate the heap functionality
def main():
    # Create a Min-Heap
    print("\n--- Min-Heap Example ---")
    min_heap = Heap(is_min_heap=True)
    elements = [3, 1, 6, 5, 2, 4]
    print("Building heap with elements:", elements)
    min_heap.build_heap(elements)
    min_heap.print_heap()

    print("Insert 0 into Min-Heap")
    min_heap.insert(0)
    min_heap.print_heap()

    print("Extract Min from Min-Heap:", min_heap.extract_top())
    min_heap.print_heap()

    print("Heap Sort:", min_heap.heap_sort())

    # Create a Max-Heap
    print("\n--- Max-Heap Example ---")
    max_heap = Heap(is_min_heap=False)
    elements = [3, 1, 6, 5, 2, 4]
    print("Building heap with elements:", elements)
    max_heap.build_heap(elements)
    max_heap.print_heap()

    print("Insert 7 into Max-Heap")
    max_heap.insert(7)
    max_heap.print_heap()

    print("Extract Max from Max-Heap:", max_heap.extract_top())
    max_heap.print_heap()

    print("Heap Sort:", max_heap.heap_sort())

if __name__ == "__main__":
    main()
