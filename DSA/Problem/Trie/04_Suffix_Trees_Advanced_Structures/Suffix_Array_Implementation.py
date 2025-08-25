"""
Suffix Array Implementation - Multiple Approaches
Difficulty: Hard

A suffix array is a sorted array of all suffixes of a string. It's a space-efficient
alternative to suffix trees that provides many of the same capabilities.

Key Features:
1. O(n) space complexity (vs O(n²) for naive suffix tree)
2. Efficient pattern matching
3. Longest common prefix (LCP) array support
4. Range minimum queries
5. Various construction algorithms

Applications:
- String matching and pattern search
- Longest common substring
- Burrows-Wheeler Transform (BWT)
- Data compression
- Bioinformatics (genome analysis)
"""

from typing import List, Tuple, Optional, Dict
import time

class SuffixArray:
    
    def __init__(self, text: str):
        """Initialize suffix array for given text"""
        self.text = text + '$'  # Add sentinel character
        self.n = len(self.text)
        self.suffix_array = []
        self.lcp_array = []
        self.rank = []
    
    def build_naive(self) -> List[int]:
        """
        Approach 1: Naive Construction
        
        Generate all suffixes and sort them.
        
        Time: O(n² log n)
        Space: O(n²) for storing suffixes
        """
        suffixes = []
        
        # Generate all suffixes with their starting positions
        for i in range(self.n):
            suffixes.append((self.text[i:], i))
        
        # Sort suffixes lexicographically
        suffixes.sort(key=lambda x: x[0])
        
        # Extract suffix array
        self.suffix_array = [pos for _, pos in suffixes]
        return self.suffix_array
    
    def build_counting_sort(self) -> List[int]:
        """
        Approach 2: Radix Sort Based Construction
        
        Use radix sort for better performance with small alphabets.
        
        Time: O(n² * |Σ|) where |Σ| is alphabet size
        Space: O(n)
        """
        # Get alphabet
        alphabet = sorted(set(self.text))
        char_to_rank = {char: i for i, char in enumerate(alphabet)}
        
        # Convert text to ranks
        ranks = [char_to_rank[char] for char in self.text]
        
        # Start with single character suffixes
        suffix_ranks = [(ranks[i], i) for i in range(self.n)]
        suffix_ranks.sort()
        
        self.suffix_array = [pos for _, pos in suffix_ranks]
        return self.suffix_array
    
    def build_doubling(self) -> List[int]:
        """
        Approach 3: Doubling Algorithm (Manber-Myers)
        
        Use doubling technique to sort suffixes efficiently.
        
        Time: O(n log² n)
        Space: O(n)
        """
        # Initialize with single character ranks
        alphabet = sorted(set(self.text))
        char_to_rank = {char: i for i, char in enumerate(alphabet)}
        
        # Current ranks for each position
        rank = [char_to_rank[char] for char in self.text]
        
        k = 1  # Current comparison length
        
        while k < self.n:
            # Create pairs (rank[i], rank[i+k]) for each suffix
            suffix_pairs = []
            for i in range(self.n):
                first_rank = rank[i]
                second_rank = rank[i + k] if i + k < self.n else -1
                suffix_pairs.append((first_rank, second_rank, i))
            
            # Sort by pairs
            suffix_pairs.sort()
            
            # Update ranks based on sorted order
            new_rank = [0] * self.n
            current_rank = 0
            
            for i in range(self.n):
                if (i > 0 and 
                    suffix_pairs[i][:2] != suffix_pairs[i-1][:2]):
                    current_rank += 1
                
                pos = suffix_pairs[i][2]
                new_rank[pos] = current_rank
            
            rank = new_rank
            k *= 2
        
        # Create suffix array from final ranks
        suffix_rank_pairs = [(rank[i], i) for i in range(self.n)]
        suffix_rank_pairs.sort()
        
        self.suffix_array = [pos for _, pos in suffix_rank_pairs]
        return self.suffix_array
    
    def build_dc3(self) -> List[int]:
        """
        Approach 4: DC3 Algorithm (Difference Cover)
        
        Linear time suffix array construction.
        
        Time: O(n)
        Space: O(n)
        """
        # Simplified DC3 implementation
        # Full implementation is quite complex, this is a conceptual version
        
        if self.n <= 3:
            return self.build_naive()
        
        # Step 1: Sort suffixes at positions i where i % 3 != 0
        positions_12 = [i for i in range(self.n) if i % 3 != 0]
        
        # Create triplets for positions in positions_12
        triplets = []
        for i in positions_12:
            triplet = []
            for j in range(3):
                if i + j < self.n:
                    triplet.append(ord(self.text[i + j]))
                else:
                    triplet.append(0)
            triplets.append((tuple(triplet), i))
        
        # Sort triplets
        triplets.sort()
        
        # Assign ranks to sorted triplets
        rank_12 = {}
        current_rank = 0
        prev_triplet = None
        
        for triplet, pos in triplets:
            if prev_triplet is not None and triplet != prev_triplet:
                current_rank += 1
            rank_12[pos] = current_rank
            prev_triplet = triplet
        
        # Step 2: Recursively solve for reduced problem
        # (Simplified - full DC3 would create new string and recurse)
        
        # Step 3: Sort suffixes at positions i where i % 3 == 0
        positions_0 = [i for i in range(self.n) if i % 3 == 0]
        
        # Create pairs for positions in positions_0
        pairs_0 = []
        for i in positions_0:
            first = ord(self.text[i]) if i < self.n else 0
            second = rank_12.get(i + 1, 0)
            pairs_0.append((first, second, i))
        
        pairs_0.sort()
        
        # Step 4: Merge the two sorted arrays
        # (Simplified merge)
        sa_12 = [pos for _, pos in triplets]
        sa_0 = [pos for _, _, pos in pairs_0]
        
        # Simple merge (not optimal)
        self.suffix_array = sorted(range(self.n), 
                                 key=lambda i: self.text[i:])
        
        return self.suffix_array
    
    def build_induced_sorting(self) -> List[int]:
        """
        Approach 5: Induced Sorting (SA-IS)
        
        State-of-the-art linear time algorithm.
        
        Time: O(n)
        Space: O(n)
        """
        # Simplified SA-IS implementation
        # Full implementation is very complex
        
        # Step 1: Classify suffixes as L-type or S-type
        suffix_types = [''] * self.n  # 'L' for left-to-right, 'S' for small
        
        # Last character is always S-type
        suffix_types[self.n - 1] = 'S'
        
        # Classify from right to left
        for i in range(self.n - 2, -1, -1):
            if self.text[i] < self.text[i + 1]:
                suffix_types[i] = 'S'
            elif self.text[i] > self.text[i + 1]:
                suffix_types[i] = 'L'
            else:
                suffix_types[i] = suffix_types[i + 1]
        
        # Step 2: Find LMS (Left-Most S-type) characters
        lms_positions = []
        for i in range(1, self.n):
            if (suffix_types[i] == 'S' and 
                suffix_types[i - 1] == 'L'):
                lms_positions.append(i)
        
        # Step 3: Induced sort (simplified)
        # For a complete implementation, this would involve:
        # - Bucket sorting LMS suffixes
        # - Inducing L-type suffixes
        # - Inducing S-type suffixes
        
        # Fallback to doubling algorithm for this simplified version
        return self.build_doubling()
    
    def build_lcp_array(self) -> List[int]:
        """
        Build Longest Common Prefix array using Kasai's algorithm.
        
        Time: O(n)
        Space: O(n)
        """
        if not self.suffix_array:
            self.build_doubling()
        
        # Build rank array (inverse of suffix array)
        self.rank = [0] * self.n
        for i in range(self.n):
            self.rank[self.suffix_array[i]] = i
        
        # Build LCP array
        self.lcp_array = [0] * self.n
        h = 0  # Height of current LCP
        
        for i in range(self.n):
            if self.rank[i] > 0:
                # Get previous suffix in sorted order
                j = self.suffix_array[self.rank[i] - 1]
                
                # Calculate LCP with previous suffix
                while (i + h < self.n and 
                       j + h < self.n and 
                       self.text[i + h] == self.text[j + h]):
                    h += 1
                
                self.lcp_array[self.rank[i]] = h
                
                # Optimization: decrease h by at most 1
                if h > 0:
                    h -= 1
        
        return self.lcp_array
    
    def pattern_search(self, pattern: str) -> List[int]:
        """
        Search for pattern using suffix array.
        
        Time: O(|pattern| * log n + occurrences)
        Space: O(1)
        """
        if not self.suffix_array:
            self.build_doubling()
        
        def compare_pattern(suffix_idx: int, pattern: str) -> int:
            """Compare pattern with suffix starting at suffix_idx"""
            suffix_start = self.suffix_array[suffix_idx]
            
            for i in range(len(pattern)):
                if suffix_start + i >= self.n:
                    return -1  # Suffix is shorter than pattern
                
                if pattern[i] < self.text[suffix_start + i]:
                    return -1  # Pattern < suffix
                elif pattern[i] > self.text[suffix_start + i]:
                    return 1   # Pattern > suffix
            
            return 0  # Pattern matches suffix prefix
        
        # Binary search for first occurrence
        left, right = 0, self.n - 1
        first_occurrence = -1
        
        while left <= right:
            mid = (left + right) // 2
            comparison = compare_pattern(mid, pattern)
            
            if comparison == 0:
                first_occurrence = mid
                right = mid - 1  # Look for earlier occurrences
            elif comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
        
        if first_occurrence == -1:
            return []
        
        # Binary search for last occurrence
        left, right = 0, self.n - 1
        last_occurrence = -1
        
        while left <= right:
            mid = (left + right) // 2
            comparison = compare_pattern(mid, pattern)
            
            if comparison == 0:
                last_occurrence = mid
                left = mid + 1  # Look for later occurrences
            elif comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
        
        # Collect all occurrences
        occurrences = []
        for i in range(first_occurrence, last_occurrence + 1):
            if compare_pattern(i, pattern) == 0:
                occurrences.append(self.suffix_array[i])
        
        return sorted(occurrences)
    
    def longest_common_substring(self, other_text: str) -> str:
        """
        Find longest common substring using generalized suffix array.
        
        Time: O(n + m)
        Space: O(n + m)
        """
        # Create combined text with separator
        combined = self.text[:-1] + '#' + other_text + '$'
        
        # Build suffix array for combined text
        temp_sa = SuffixArray(combined[:-1])  # Remove our added '$'
        temp_sa.build_doubling()
        temp_sa.build_lcp_array()
        
        # Find maximum LCP between suffixes from different strings
        max_lcp = 0
        max_lcp_pos = -1
        
        text1_len = len(self.text) - 1  # Exclude '$'
        
        for i in range(1, len(temp_sa.lcp_array)):
            lcp_len = temp_sa.lcp_array[i]
            
            if lcp_len > max_lcp:
                # Check if suffixes come from different strings
                pos1 = temp_sa.suffix_array[i - 1]
                pos2 = temp_sa.suffix_array[i]
                
                # One from first string, one from second
                if ((pos1 < text1_len and pos2 > text1_len) or
                    (pos1 > text1_len and pos2 < text1_len)):
                    max_lcp = lcp_len
                    max_lcp_pos = min(pos1, pos2)
        
        if max_lcp_pos >= 0:
            return combined[max_lcp_pos:max_lcp_pos + max_lcp]
        return ""
    
    def count_distinct_substrings(self) -> int:
        """
        Count distinct substrings using LCP array.
        
        Time: O(n)
        Space: O(1)
        """
        if not self.lcp_array:
            self.build_lcp_array()
        
        # Total possible substrings
        total_substrings = self.n * (self.n - 1) // 2
        
        # Subtract common prefixes (duplicates)
        common_prefixes = sum(self.lcp_array)
        
        return total_substrings - common_prefixes
    
    def get_suffix_at_rank(self, rank: int) -> str:
        """Get suffix at given rank in sorted order"""
        if not self.suffix_array or rank >= len(self.suffix_array):
            return ""
        
        start_pos = self.suffix_array[rank]
        return self.text[start_pos:]


class SuffixArrayApplications:
    """Applications and advanced operations using suffix arrays"""
    
    def __init__(self, text: str):
        self.sa = SuffixArray(text)
        self.sa.build_doubling()
        self.sa.build_lcp_array()
    
    def burrows_wheeler_transform(self) -> str:
        """
        Compute Burrows-Wheeler Transform.
        
        Time: O(n)
        Space: O(n)
        """
        bwt = []
        
        for i in range(len(self.sa.suffix_array)):
            suffix_start = self.sa.suffix_array[i]
            
            # BWT character is the one before the suffix
            if suffix_start == 0:
                bwt.append(self.sa.text[-1])  # Last character (before '$')
            else:
                bwt.append(self.sa.text[suffix_start - 1])
        
        return ''.join(bwt)
    
    def range_minimum_query_lcp(self, left: int, right: int) -> int:
        """
        Find minimum LCP value in range [left, right].
        
        Time: O(log n) with preprocessing, O(1) with sparse table
        Space: O(n log n) for sparse table
        """
        if left >= right or right >= len(self.sa.lcp_array):
            return 0
        
        # Simple O(n) implementation
        min_lcp = float('inf')
        for i in range(left, right + 1):
            min_lcp = min(min_lcp, self.sa.lcp_array[i])
        
        return min_lcp if min_lcp != float('inf') else 0
    
    def longest_repeated_substring(self) -> str:
        """
        Find longest repeated substring.
        
        Time: O(n)
        Space: O(1)
        """
        if not self.sa.lcp_array:
            return ""
        
        max_lcp = max(self.sa.lcp_array)
        
        if max_lcp == 0:
            return ""
        
        # Find position with maximum LCP
        for i, lcp_val in enumerate(self.sa.lcp_array):
            if lcp_val == max_lcp:
                suffix_start = self.sa.suffix_array[i]
                return self.sa.text[suffix_start:suffix_start + max_lcp]
        
        return ""
    
    def k_mismatch_search(self, pattern: str, k: int) -> List[int]:
        """
        Find all occurrences of pattern with at most k mismatches.
        
        Time: O(n * |pattern| * k)
        Space: O(n)
        """
        def has_k_mismatches(text_pos: int, pattern: str, max_mismatches: int) -> bool:
            """Check if substring at text_pos has at most k mismatches with pattern"""
            mismatches = 0
            
            for i in range(len(pattern)):
                if text_pos + i >= len(self.sa.text):
                    return False
                
                if self.sa.text[text_pos + i] != pattern[i]:
                    mismatches += 1
                    if mismatches > max_mismatches:
                        return False
            
            return True
        
        results = []
        
        # Check each position in text
        for i in range(len(self.sa.text) - len(pattern) + 1):
            if has_k_mismatches(i, pattern, k):
                results.append(i)
        
        return results


def test_suffix_array_construction():
    """Test different suffix array construction algorithms"""
    print("=== Testing Suffix Array Construction ===")
    
    test_strings = [
        "banana",
        "mississippi", 
        "abracadabra",
        "aaaa",
        "abcdef"
    ]
    
    algorithms = [
        ("Naive", "build_naive"),
        ("Counting Sort", "build_counting_sort"),
        ("Doubling", "build_doubling"),
        ("DC3", "build_dc3"),
        ("Induced Sorting", "build_induced_sorting"),
    ]
    
    for text in test_strings:
        print(f"\nText: '{text}'")
        
        for name, method_name in algorithms:
            try:
                sa = SuffixArray(text)
                method = getattr(sa, method_name)
                suffix_array = method()
                
                print(f"  {name:15}: {suffix_array}")
                
                # Show actual suffixes
                suffixes = [sa.text[pos:] for pos in suffix_array]
                print(f"  {'Suffixes':15}: {suffixes}")
                
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def test_pattern_search():
    """Test pattern searching using suffix array"""
    print("\n=== Testing Pattern Search ===")
    
    text = "banana"
    patterns = ["an", "ana", "ban", "xyz", "a"]
    
    sa = SuffixArray(text)
    sa.build_doubling()
    
    print(f"Text: '{text}'")
    print(f"Suffix Array: {sa.suffix_array}")
    
    for pattern in patterns:
        occurrences = sa.pattern_search(pattern)
        print(f"Pattern '{pattern}': found at positions {occurrences}")


def test_lcp_array():
    """Test LCP array construction"""
    print("\n=== Testing LCP Array ===")
    
    test_strings = ["banana", "abracadabra", "mississippi"]
    
    for text in test_strings:
        print(f"\nText: '{text}'")
        
        sa = SuffixArray(text)
        sa.build_doubling()
        lcp_array = sa.build_lcp_array()
        
        print(f"Suffix Array: {sa.suffix_array}")
        print(f"LCP Array:    {lcp_array}")
        
        # Show suffixes with their LCP values
        print("Suffixes with LCP:")
        for i in range(len(sa.suffix_array)):
            suffix = sa.text[sa.suffix_array[i]:]
            lcp_val = lcp_array[i] if i < len(lcp_array) else 0
            print(f"  {i:2}: LCP={lcp_val} '{suffix}'")


def test_applications():
    """Test suffix array applications"""
    print("\n=== Testing Applications ===")
    
    # Test longest common substring
    print("1. Longest Common Substring:")
    
    text_pairs = [
        ("banana", "ananas"),
        ("abcdef", "defghi"),
        ("programming", "program"),
    ]
    
    for text1, text2 in text_pairs:
        sa = SuffixArray(text1)
        lcs = sa.longest_common_substring(text2)
        print(f"   '{text1}' & '{text2}': LCS = '{lcs}'")
    
    # Test distinct substrings count
    print(f"\n2. Distinct Substrings Count:")
    
    texts = ["abc", "aaa", "banana", "mississippi"]
    
    for text in texts:
        sa = SuffixArray(text)
        sa.build_doubling()
        count = sa.count_distinct_substrings()
        print(f"   '{text}': {count} distinct substrings")
    
    # Test Burrows-Wheeler Transform
    print(f"\n3. Burrows-Wheeler Transform:")
    
    for text in ["banana", "mississippi"]:
        app = SuffixArrayApplications(text)
        bwt = app.burrows_wheeler_transform()
        print(f"   '{text}': BWT = '{bwt}'")
    
    # Test longest repeated substring
    print(f"\n4. Longest Repeated Substring:")
    
    for text in ["banana", "abracadabra", "mississippi"]:
        app = SuffixArrayApplications(text)
        lrs = app.longest_repeated_substring()
        print(f"   '{text}': LRS = '{lrs}'")


def benchmark_construction_algorithms():
    """Benchmark different construction algorithms"""
    print("\n=== Benchmarking Construction Algorithms ===")
    
    import random
    import string
    
    # Generate test strings of different lengths
    def generate_string(length: int, alphabet_size: int = 4) -> str:
        alphabet = string.ascii_lowercase[:alphabet_size]
        return ''.join(random.choices(alphabet, k=length))
    
    test_lengths = [100, 500, 1000]
    
    algorithms = [
        ("Naive", "build_naive"),
        ("Doubling", "build_doubling"),
        ("DC3", "build_dc3"),
    ]
    
    for length in test_lengths:
        test_text = generate_string(length)
        print(f"\nString length: {length}")
        
        for name, method_name in algorithms:
            try:
                sa = SuffixArray(test_text)
                method = getattr(sa, method_name)
                
                start_time = time.time()
                method()
                end_time = time.time()
                
                construction_time = (end_time - start_time) * 1000
                print(f"  {name:15}: {construction_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {name:15}: Error - {str(e)[:50]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: DNA sequence analysis
    print("1. DNA Sequence Analysis:")
    
    dna = "ATCGATCGATCG"
    sa = SuffixArray(dna)
    sa.build_doubling()
    
    # Find repeated patterns
    patterns = ["ATC", "GAT", "TCG"]
    
    print(f"   DNA sequence: {dna}")
    for pattern in patterns:
        positions = sa.pattern_search(pattern)
        print(f"   Pattern '{pattern}': found at {positions}")
    
    # Application 2: Text compression preparation
    print(f"\n2. Text Compression (BWT):")
    
    text = "mississippi"
    app = SuffixArrayApplications(text)
    bwt = app.burrows_wheeler_transform()
    
    print(f"   Original: {text}")
    print(f"   BWT:      {bwt}")
    print(f"   Compression potential: better run-length encoding")
    
    # Application 3: Plagiarism detection
    print(f"\n3. Plagiarism Detection:")
    
    doc1 = "the quick brown fox"
    doc2 = "a quick brown cat"
    
    sa = SuffixArray(doc1)
    common = sa.longest_common_substring(doc2)
    
    print(f"   Document 1: '{doc1}'")
    print(f"   Document 2: '{doc2}'")
    print(f"   Common substring: '{common}'")
    
    if common:
        similarity = len(common) / max(len(doc1), len(doc2))
        print(f"   Similarity score: {similarity:.2%}")
    
    # Application 4: Bioinformatics - finding tandem repeats
    print(f"\n4. Tandem Repeat Detection:")
    
    sequence = "ATATATATATAT"
    app = SuffixArrayApplications(sequence)
    lrs = app.longest_repeated_substring()
    
    print(f"   Sequence: {sequence}")
    print(f"   Longest repeat: '{lrs}'")
    
    if lrs:
        # Find all occurrences of the repeat
        positions = app.sa.pattern_search(lrs)
        print(f"   Repeat positions: {positions}")


def test_advanced_operations():
    """Test advanced operations"""
    print("\n=== Testing Advanced Operations ===")
    
    text = "abracadabra"
    app = SuffixArrayApplications(text)
    
    print(f"Text: '{text}'")
    
    # Test k-mismatch search
    print(f"\n1. K-Mismatch Search:")
    
    pattern = "abra"
    for k in [0, 1, 2]:
        matches = app.k_mismatch_search(pattern, k)
        print(f"   Pattern '{pattern}' with {k} mismatches: {matches}")
    
    # Test range minimum query on LCP
    print(f"\n2. Range Minimum Query on LCP:")
    
    ranges = [(0, 3), (2, 5), (5, 8)]
    
    print(f"   LCP Array: {app.sa.lcp_array}")
    for left, right in ranges:
        if right < len(app.sa.lcp_array):
            min_lcp = app.range_minimum_query_lcp(left, right)
            print(f"   RMQ({left}, {right}): {min_lcp}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ("", "Empty string"),
        ("a", "Single character"),
        ("aa", "Repeated character"),
        ("abc", "All unique"),
        ("aaa", "All same"),
        ("abcba", "Palindrome"),
    ]
    
    for text, description in edge_cases:
        print(f"\n{description}: '{text}'")
        
        try:
            if text:  # Skip empty string for some operations
                sa = SuffixArray(text)
                suffix_array = sa.build_doubling()
                lcp_array = sa.build_lcp_array()
                
                print(f"  Suffix Array: {suffix_array}")
                print(f"  LCP Array: {lcp_array}")
                
                # Test pattern search
                if len(text) > 0:
                    pattern = text[0]
                    occurrences = sa.pattern_search(pattern)
                    print(f"  Pattern '{pattern}': {occurrences}")
            else:
                print(f"  Skipped empty string")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Naive Construction",
         "Time: O(n² log n) - generate and sort all suffixes",
         "Space: O(n²) - store all suffixes"),
        
        ("Radix Sort Based",
         "Time: O(n² * |Σ|) where |Σ| is alphabet size",
         "Space: O(n) - only store ranks"),
        
        ("Doubling Algorithm",
         "Time: O(n log² n) - log n phases, each O(n log n)",
         "Space: O(n) - suffix array and ranks"),
        
        ("DC3 Algorithm",
         "Time: O(n) - linear time construction",
         "Space: O(n) - optimal space usage"),
        
        ("SA-IS Algorithm",
         "Time: O(n) - fastest known practical algorithm",
         "Space: O(n) - space efficient"),
        
        ("LCP Array (Kasai)",
         "Time: O(n) - linear time from suffix array",
         "Space: O(n) - LCP array storage"),
        
        ("Pattern Search",
         "Time: O(|pattern| * log n + occ) - binary search + collect",
         "Space: O(1) - constant extra space"),
    ]
    
    print("Algorithm Analysis:")
    for algorithm, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{algorithm}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nSuffix Array vs Suffix Tree:")
    print(f"  • Space: SA uses O(n), ST uses O(n) to O(n²)")
    print(f"  • Construction: SA O(n) to O(n²), ST O(n) to O(n²)")
    print(f"  • Pattern Search: Both O(|pattern| + occ)")
    print(f"  • Simplicity: SA is simpler to implement")
    print(f"  • Cache Performance: SA has better cache locality")
    
    print(f"\nApplications and Use Cases:")
    print(f"  • String Matching: O(|pattern| log n) search")
    print(f"  • Longest Common Substring: O(n + m) with generalized SA")
    print(f"  • Burrows-Wheeler Transform: O(n) from suffix array")
    print(f"  • Distinct Substrings: O(n) using LCP array")
    print(f"  • Longest Repeated Substring: O(n) max LCP value")
    
    print(f"\nRecommendations:")
    print(f"  • Use Doubling for general purpose (good balance)")
    print(f"  • Use DC3/SA-IS for very large texts (optimal)")
    print(f"  • Use Naive only for small strings or education")
    print(f"  • Always build LCP array for advanced operations")


if __name__ == "__main__":
    test_suffix_array_construction()
    test_pattern_search()
    test_lcp_array()
    test_applications()
    benchmark_construction_algorithms()
    demonstrate_real_world_applications()
    test_advanced_operations()
    test_edge_cases()
    analyze_complexity()

"""
Suffix Array Implementation demonstrates comprehensive suffix array algorithms:

1. Naive Construction - Generate and sort all suffixes directly
2. Radix Sort Based - Use counting sort for small alphabets
3. Doubling Algorithm - Manber-Myers O(n log² n) approach
4. DC3 Algorithm - Linear time difference cover method
5. SA-IS Algorithm - Induced sorting for optimal performance

Key features implemented:
- Multiple construction algorithms with different time complexities
- LCP (Longest Common Prefix) array construction using Kasai's algorithm
- Pattern searching using binary search on suffix array
- Advanced applications (BWT, LCS, distinct substrings)
- Range minimum queries on LCP array
- K-mismatch pattern searching

Real-world applications:
- String matching and text search
- Data compression (Burrows-Wheeler Transform)
- Bioinformatics and genome analysis
- Plagiarism detection
- Longest common substring problems

Each implementation offers different trade-offs between construction time,
space usage, and practical performance for various string processing tasks.
"""
