"""
String Compression and Encoding - Advanced Techniques
====================================================

Topics: Run-length encoding, Huffman coding, LZ compression
Companies: Amazon, Google, Microsoft, Netflix
Difficulty: Medium to Hard
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import heapq

class StringCompressionEncoding:
    
    # ==========================================
    # 1. RUN-LENGTH ENCODING
    # ==========================================
    
    def run_length_encode(self, s: str) -> str:
        """Basic run-length encoding"""
        if not s:
            return ""
        
        encoded = []
        current_char = s[0]
        count = 1
        
        for i in range(1, len(s)):
            if s[i] == current_char:
                count += 1
            else:
                encoded.append(f"{count}{current_char}")
                current_char = s[i]
                count = 1
        
        encoded.append(f"{count}{current_char}")
        return ''.join(encoded)
    
    def run_length_decode(self, encoded: str) -> str:
        """Decode run-length encoded string"""
        decoded = []
        i = 0
        
        while i < len(encoded):
            count_str = ""
            while i < len(encoded) and encoded[i].isdigit():
                count_str += encoded[i]
                i += 1
            
            if i < len(encoded):
                char = encoded[i]
                count = int(count_str) if count_str else 1
                decoded.append(char * count)
                i += 1
        
        return ''.join(decoded)
    
    def string_compression_leetcode(self, chars: List[str]) -> int:
        """LC 443: String Compression - In-place modification"""
        write = 0
        read = 0
        
        while read < len(chars):
            char = chars[read]
            count = 0
            
            # Count consecutive characters
            while read < len(chars) and chars[read] == char:
                read += 1
                count += 1
            
            # Write character
            chars[write] = char
            write += 1
            
            # Write count if > 1
            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1
        
        return write
    
    # ==========================================
    # 2. HUFFMAN CODING
    # ==========================================
    
    class HuffmanNode:
        def __init__(self, char: str = None, freq: int = 0, left=None, right=None):
            self.char = char
            self.freq = freq
            self.left = left
            self.right = right
        
        def __lt__(self, other):
            return self.freq < other.freq
    
    def build_huffman_tree(self, text: str) -> 'HuffmanNode':
        """Build Huffman tree from text"""
        if not text:
            return None
        
        # Count frequencies
        freq_map = Counter(text)
        
        # Create leaf nodes and add to priority queue
        heap = [self.HuffmanNode(char, freq) for char, freq in freq_map.items()]
        heapq.heapify(heap)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = self.HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def build_huffman_codes(self, root: 'HuffmanNode') -> Dict[str, str]:
        """Build Huffman codes from tree"""
        if not root:
            return {}
        
        codes = {}
        
        def dfs(node, code):
            if node.char is not None:  # Leaf node
                codes[node.char] = code if code else "0"  # Handle single character case
                return
            
            if node.left:
                dfs(node.left, code + "0")
            if node.right:
                dfs(node.right, code + "1")
        
        dfs(root, "")
        return codes
    
    def huffman_encode(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Encode text using Huffman coding"""
        if not text:
            return "", {}
        
        # Build tree and codes
        root = self.build_huffman_tree(text)
        codes = self.build_huffman_codes(root)
        
        # Encode text
        encoded = ''.join(codes[char] for char in text)
        return encoded, codes
    
    def huffman_decode(self, encoded: str, codes: Dict[str, str]) -> str:
        """Decode Huffman encoded string"""
        if not encoded or not codes:
            return ""
        
        # Reverse the codes dictionary
        reverse_codes = {code: char for char, code in codes.items()}
        
        decoded = []
        current_code = ""
        
        for bit in encoded:
            current_code += bit
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ""
        
        return ''.join(decoded)
    
    # ==========================================
    # 3. LZ77 COMPRESSION
    # ==========================================
    
    def lz77_encode(self, text: str, window_size: int = 12, buffer_size: int = 4) -> List[Tuple[int, int, str]]:
        """LZ77 encoding algorithm"""
        encoded = []
        i = 0
        
        while i < len(text):
            # Look for longest match in window
            best_match = (0, 0, text[i])
            
            # Search window
            start = max(0, i - window_size)
            
            for j in range(start, i):
                k = 0
                while (i + k < len(text) and 
                       j + k < i and 
                       k < buffer_size and
                       text[j + k] == text[i + k]):
                    k += 1
                
                if k > best_match[1]:
                    best_match = (i - j, k, text[i + k] if i + k < len(text) else '')
            
            encoded.append(best_match)
            i += max(1, best_match[1])
        
        return encoded
    
    def lz77_decode(self, encoded: List[Tuple[int, int, str]]) -> str:
        """LZ77 decoding algorithm"""
        decoded = []
        
        for offset, length, char in encoded:
            if length > 0:
                # Copy from previous position
                start_pos = len(decoded) - offset
                for i in range(length):
                    decoded.append(decoded[start_pos + i])
            
            if char:
                decoded.append(char)
        
        return ''.join(decoded)
    
    # ==========================================
    # 4. BASE ENCODING/DECODING
    # ==========================================
    
    def base64_encode(self, text: str) -> str:
        """Custom Base64 encoding implementation"""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        
        # Convert to binary
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        # Pad to multiple of 6
        while len(binary) % 6 != 0:
            binary += '0'
        
        # Convert to base64
        encoded = []
        for i in range(0, len(binary), 6):
            chunk = binary[i:i+6]
            index = int(chunk, 2)
            encoded.append(chars[index])
        
        # Add padding
        while len(encoded) % 4 != 0:
            encoded.append('=')
        
        return ''.join(encoded)
    
    def base64_decode(self, encoded: str) -> str:
        """Custom Base64 decoding implementation"""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        char_to_index = {char: i for i, char in enumerate(chars)}
        
        # Remove padding
        encoded = encoded.rstrip('=')
        
        # Convert to binary
        binary = []
        for char in encoded:
            if char in char_to_index:
                binary.append(format(char_to_index[char], '06b'))
        
        binary_str = ''.join(binary)
        
        # Convert to text
        decoded = []
        for i in range(0, len(binary_str), 8):
            if i + 8 <= len(binary_str):
                byte = binary_str[i:i+8]
                decoded.append(chr(int(byte, 2)))
        
        return ''.join(decoded)
    
    # ==========================================
    # 5. ADVANCED COMPRESSION TECHNIQUES
    # ==========================================
    
    def burrows_wheeler_transform(self, text: str) -> Tuple[str, int]:
        """Burrows-Wheeler Transform"""
        if not text:
            return "", 0
        
        # Add end marker
        text += '$'
        n = len(text)
        
        # Generate all rotations
        rotations = [text[i:] + text[:i] for i in range(n)]
        
        # Sort rotations
        rotations.sort()
        
        # Extract last column and find original index
        last_column = ''.join(rotation[-1] for rotation in rotations)
        original_index = rotations.index(text)
        
        return last_column, original_index
    
    def inverse_burrows_wheeler(self, last_column: str, original_index: int) -> str:
        """Inverse Burrows-Wheeler Transform"""
        if not last_column:
            return ""
        
        n = len(last_column)
        
        # Create table with indices
        table = [(last_column[i], i) for i in range(n)]
        table.sort()
        
        # Reconstruct original string
        result = []
        current_index = original_index
        
        for _ in range(n):
            result.append(table[current_index][0])
            current_index = table[current_index][1]
        
        # Remove end marker and reverse
        original = ''.join(reversed(result[1:]))
        return original
    
    def move_to_front_encode(self, text: str) -> List[int]:
        """Move-to-Front encoding"""
        if not text:
            return []
        
        # Initialize alphabet
        alphabet = list(set(text))
        alphabet.sort()
        
        encoded = []
        
        for char in text:
            index = alphabet.index(char)
            encoded.append(index)
            
            # Move to front
            alphabet.pop(index)
            alphabet.insert(0, char)
        
        return encoded
    
    def move_to_front_decode(self, encoded: List[int], alphabet: List[str]) -> str:
        """Move-to-Front decoding"""
        if not encoded:
            return ""
        
        alphabet = alphabet.copy()
        decoded = []
        
        for index in encoded:
            char = alphabet[index]
            decoded.append(char)
            
            # Move to front
            alphabet.pop(index)
            alphabet.insert(0, char)
        
        return ''.join(decoded)
    
    # ==========================================
    # 6. COMPRESSION ANALYSIS
    # ==========================================
    
    def compression_ratio(self, original: str, compressed: str) -> float:
        """Calculate compression ratio"""
        if not original:
            return 0.0
        return len(compressed) / len(original)
    
    def entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        import math
        
        freq_map = Counter(text)
        length = len(text)
        
        entropy = 0
        for count in freq_map.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def analyze_compression(self, text: str) -> Dict[str, any]:
        """Comprehensive compression analysis"""
        analysis = {}
        
        # Original metrics
        analysis['original_length'] = len(text)
        analysis['entropy'] = self.entropy(text)
        analysis['unique_chars'] = len(set(text))
        
        # Run-length encoding
        rle = self.run_length_encode(text)
        analysis['rle_length'] = len(rle)
        analysis['rle_ratio'] = self.compression_ratio(text, rle)
        
        # Huffman encoding
        huffman_encoded, codes = self.huffman_encode(text)
        analysis['huffman_length'] = len(huffman_encoded)
        analysis['huffman_ratio'] = len(huffman_encoded) / (len(text) * 8) if text else 0
        analysis['huffman_codes'] = len(codes)
        
        # LZ77 encoding
        lz77_encoded = self.lz77_encode(text)
        analysis['lz77_tokens'] = len(lz77_encoded)
        
        return analysis

# Test Examples
def run_examples():
    sce = StringCompressionEncoding()
    
    print("=== STRING COMPRESSION AND ENCODING EXAMPLES ===\n")
    
    # Run-length encoding
    print("1. RUN-LENGTH ENCODING:")
    text = "aaabbccccaa"
    rle = sce.run_length_encode(text)
    decoded = sce.run_length_decode(rle)
    print(f"Original: '{text}'")
    print(f"RLE: '{rle}'")
    print(f"Decoded: '{decoded}'")
    
    # Huffman coding
    print("\n2. HUFFMAN CODING:")
    text = "hello world"
    encoded, codes = sce.huffman_encode(text)
    decoded_huffman = sce.huffman_decode(encoded, codes)
    print(f"Original: '{text}'")
    print(f"Codes: {codes}")
    print(f"Encoded: '{encoded}'")
    print(f"Decoded: '{decoded_huffman}'")
    
    # LZ77 compression
    print("\n3. LZ77 COMPRESSION:")
    text = "abcabcabc"
    lz77_encoded = sce.lz77_encode(text)
    lz77_decoded = sce.lz77_decode(lz77_encoded)
    print(f"Original: '{text}'")
    print(f"LZ77: {lz77_encoded}")
    print(f"Decoded: '{lz77_decoded}'")
    
    # Base64 encoding
    print("\n4. BASE64 ENCODING:")
    text = "Hello"
    b64_encoded = sce.base64_encode(text)
    b64_decoded = sce.base64_decode(b64_encoded)
    print(f"Original: '{text}'")
    print(f"Base64: '{b64_encoded}'")
    print(f"Decoded: '{b64_decoded}'")
    
    # Burrows-Wheeler Transform
    print("\n5. BURROWS-WHEELER TRANSFORM:")
    text = "banana"
    bwt, index = sce.burrows_wheeler_transform(text)
    inverse_bwt = sce.inverse_burrows_wheeler(bwt, index)
    print(f"Original: '{text}'")
    print(f"BWT: '{bwt}', Index: {index}")
    print(f"Inverse: '{inverse_bwt}'")
    
    # Compression analysis
    print("\n6. COMPRESSION ANALYSIS:")
    analysis = sce.analyze_compression("aaabbccccdddddeeeee")
    for key, value in analysis.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    run_examples() 