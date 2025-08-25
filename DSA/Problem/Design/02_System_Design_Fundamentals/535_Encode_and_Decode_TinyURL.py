"""
535. Encode and Decode TinyURL - Multiple Approaches
Difficulty: Medium

TinyURL is a URL shortening service where you enter a URL such as 
https://leetcode.com/problems/design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk.

Design a class to encode a URL and decode a shortened URL.

There is no restriction on how your encode/decode algorithm should work. You just need to ensure 
that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.

Implement the Solution class:
- Solution() Initializes the object of the system.
- String encode(String longUrl) Encodes a URL to a shortened URL.
- String decode(String shortUrl) Decodes a shortened URL to its original URL.
"""

import hashlib
import random
import string
import time
from typing import Dict, Optional

class Codec:
    """
    Approach 1: Counter-based Encoding
    
    Use a simple counter to generate short codes.
    
    Time Complexity: O(1) for both encode and decode
    Space Complexity: O(n) where n is the number of URLs
    """
    
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.counter = 0
        self.base_url = "http://tinyurl.com/"
    
    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL."""
        if longUrl in self.url_to_code:
            return self.base_url + self.url_to_code[longUrl]
        
        self.counter += 1
        code = str(self.counter)
        
        self.url_to_code[longUrl] = code
        self.code_to_url[code] = longUrl
        
        return self.base_url + code
    
    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL."""
        code = shortUrl.replace(self.base_url, "")
        return self.code_to_url.get(code, "")

class CodecBase62:
    """
    Approach 2: Base62 Encoding
    
    Use base62 encoding for more compact URLs.
    
    Time Complexity: O(1) for both encode and decode
    Space Complexity: O(n) where n is the number of URLs
    """
    
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.counter = 0
        self.base_url = "http://tinyurl.com/"
        self.alphabet = string.ascii_letters + string.digits  # 62 characters
    
    def encode(self, longUrl: str) -> str:
        if longUrl in self.url_to_code:
            return self.base_url + self.url_to_code[longUrl]
        
        self.counter += 1
        code = self._base62_encode(self.counter)
        
        self.url_to_code[longUrl] = code
        self.code_to_url[code] = longUrl
        
        return self.base_url + code
    
    def decode(self, shortUrl: str) -> str:
        code = shortUrl.replace(self.base_url, "")
        return self.code_to_url.get(code, "")
    
    def _base62_encode(self, num: int) -> str:
        if num == 0:
            return self.alphabet[0]
        
        result = ""
        while num:
            result = self.alphabet[num % 62] + result
            num //= 62
        
        return result

class CodecRandom:
    """
    Approach 3: Random Code Generation
    
    Generate random codes for URLs.
    
    Time Complexity: O(1) average for encode and decode
    Space Complexity: O(n) where n is the number of URLs
    """
    
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.base_url = "http://tinyurl.com/"
        self.code_length = 6
    
    def encode(self, longUrl: str) -> str:
        if longUrl in self.url_to_code:
            return self.base_url + self.url_to_code[longUrl]
        
        code = self._generate_random_code()
        
        # Handle collisions
        while code in self.code_to_url:
            code = self._generate_random_code()
        
        self.url_to_code[longUrl] = code
        self.code_to_url[code] = longUrl
        
        return self.base_url + code
    
    def decode(self, shortUrl: str) -> str:
        code = shortUrl.replace(self.base_url, "")
        return self.code_to_url.get(code, "")
    
    def _generate_random_code(self) -> str:
        alphabet = string.ascii_letters + string.digits
        return ''.join(random.choices(alphabet, k=self.code_length))

class CodecHash:
    """
    Approach 4: Hash-based Encoding
    
    Use hash functions to generate codes.
    
    Time Complexity: O(1) for both encode and decode
    Space Complexity: O(n) where n is the number of URLs
    """
    
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.base_url = "http://tinyurl.com/"
    
    def encode(self, longUrl: str) -> str:
        if longUrl in self.url_to_code:
            return self.base_url + self.url_to_code[longUrl]
        
        # Generate hash and take first 6 characters
        hash_value = hashlib.md5(longUrl.encode()).hexdigest()
        code = hash_value[:6]
        
        # Handle collisions by adding suffix
        original_code = code
        suffix = 0
        while code in self.code_to_url:
            suffix += 1
            code = original_code + str(suffix)
        
        self.url_to_code[longUrl] = code
        self.code_to_url[code] = longUrl
        
        return self.base_url + code
    
    def decode(self, shortUrl: str) -> str:
        code = shortUrl.replace(self.base_url, "")
        return self.code_to_url.get(code, "")

class CodecWithExpiration:
    """
    Approach 5: Enhanced with Expiration
    
    Add expiration time for URLs.
    
    Time Complexity: O(1) for both encode and decode
    Space Complexity: O(n) where n is the number of URLs
    """
    
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.expiration_times = {}  # code -> expiration_timestamp
        self.base_url = "http://tinyurl.com/"
        self.counter = 0
        self.default_expiration = 3600  # 1 hour in seconds
    
    def encode(self, longUrl: str, expiration_seconds: Optional[int] = None) -> str:
        if longUrl in self.url_to_code:
            return self.base_url + self.url_to_code[longUrl]
        
        self.counter += 1
        code = self._base62_encode(self.counter)
        
        # Set expiration time
        expiration = expiration_seconds or self.default_expiration
        expiration_time = time.time() + expiration
        
        self.url_to_code[longUrl] = code
        self.code_to_url[code] = longUrl
        self.expiration_times[code] = expiration_time
        
        return self.base_url + code
    
    def decode(self, shortUrl: str) -> str:
        code = shortUrl.replace(self.base_url, "")
        
        # Check if expired
        if code in self.expiration_times:
            if time.time() > self.expiration_times[code]:
                # Clean up expired entry
                long_url = self.code_to_url.get(code, "")
                if long_url in self.url_to_code:
                    del self.url_to_code[long_url]
                if code in self.code_to_url:
                    del self.code_to_url[code]
                del self.expiration_times[code]
                return ""
        
        return self.code_to_url.get(code, "")
    
    def _base62_encode(self, num: int) -> str:
        alphabet = string.ascii_letters + string.digits
        if num == 0:
            return alphabet[0]
        
        result = ""
        while num:
            result = alphabet[num % 62] + result
            num //= 62
        
        return result
    
    def cleanup_expired(self) -> int:
        """Clean up expired URLs and return count removed"""
        current_time = time.time()
        expired_codes = []
        
        for code, expiration_time in self.expiration_times.items():
            if current_time > expiration_time:
                expired_codes.append(code)
        
        for code in expired_codes:
            long_url = self.code_to_url.get(code, "")
            if long_url in self.url_to_code:
                del self.url_to_code[long_url]
            if code in self.code_to_url:
                del self.code_to_url[code]
            del self.expiration_times[code]
        
        return len(expired_codes)


def test_basic_functionality():
    """Test basic encode/decode functionality"""
    print("=== Testing Basic Encode/Decode Functionality ===")
    
    implementations = [
        ("Counter-based", Codec),
        ("Base62", CodecBase62),
        ("Random", CodecRandom),
        ("Hash-based", CodecHash)
    ]
    
    test_urls = [
        "https://leetcode.com/problems/design-tinyurl",
        "https://www.google.com",
        "https://github.com/user/repo",
        "https://stackoverflow.com/questions/123456"
    ]
    
    for name, CodecClass in implementations:
        print(f"\n{name}:")
        
        codec = CodecClass()
        
        for url in test_urls:
            # Test encoding
            short_url = codec.encode(url)
            print(f"  Encode: {url[:30]}... -> {short_url}")
            
            # Test decoding
            decoded = codec.decode(short_url)
            print(f"  Decode: {short_url} -> {'✓' if decoded == url else '✗'}")

def test_duplicate_encoding():
    """Test handling of duplicate URLs"""
    print("\n=== Testing Duplicate URL Handling ===")
    
    codec = Codec()
    
    url = "https://example.com"
    
    # Encode same URL multiple times
    short1 = codec.encode(url)
    short2 = codec.encode(url)
    short3 = codec.encode(url)
    
    print(f"Original URL: {url}")
    print(f"First encoding: {short1}")
    print(f"Second encoding: {short2}")
    print(f"Third encoding: {short3}")
    print(f"All same: {'✓' if short1 == short2 == short3 else '✗'}")

def test_collision_handling():
    """Test collision handling in random codec"""
    print("\n=== Testing Collision Handling ===")
    
    # Test with very short codes to force collisions
    class ShortCodec(CodecRandom):
        def __init__(self):
            super().__init__()
            self.code_length = 2  # Very short to force collisions
    
    codec = ShortCodec()
    
    urls = [f"https://example{i}.com" for i in range(100)]
    short_urls = []
    
    for url in urls:
        short_url = codec.encode(url)
        short_urls.append(short_url)
    
    # Check for uniqueness
    unique_short_urls = set(short_urls)
    print(f"Encoded {len(urls)} URLs")
    print(f"Generated {len(unique_short_urls)} unique short URLs")
    print(f"Collision handling: {'✓' if len(unique_short_urls) == len(urls) else '✗'}")

def test_expiration_functionality():
    """Test expiration functionality"""
    print("\n=== Testing Expiration Functionality ===")
    
    codec = CodecWithExpiration()
    
    # Test with short expiration
    url = "https://temporary.com"
    short_url = codec.encode(url, expiration_seconds=1)  # 1 second expiration
    
    print(f"Encoded: {url} -> {short_url}")
    
    # Decode immediately
    decoded = codec.decode(short_url)
    print(f"Decode immediately: {'✓' if decoded == url else '✗'}")
    
    # Wait for expiration
    print("Waiting for expiration...")
    time.sleep(1.1)
    
    # Try to decode after expiration
    decoded_after = codec.decode(short_url)
    print(f"Decode after expiration: {'✓' if decoded_after == '' else '✗'}")
    
    # Test cleanup
    print(f"Expired URLs cleaned up: {codec.cleanup_expired()}")

def test_performance():
    """Test performance of different implementations"""
    print("\n=== Testing Performance ===")
    
    import time
    
    implementations = [
        ("Counter-based", Codec),
        ("Base62", CodecBase62),
        ("Random", CodecRandom),
        ("Hash-based", CodecHash)
    ]
    
    num_urls = 10000
    test_urls = [f"https://example{i}.com/path/{i}" for i in range(num_urls)]
    
    for name, CodecClass in implementations:
        codec = CodecClass()
        
        # Test encoding performance
        start_time = time.time()
        short_urls = []
        for url in test_urls:
            short_url = codec.encode(url)
            short_urls.append(short_url)
        encode_time = (time.time() - start_time) * 1000
        
        # Test decoding performance
        start_time = time.time()
        for short_url in short_urls:
            codec.decode(short_url)
        decode_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Encode {num_urls} URLs: {encode_time:.2f}ms")
        print(f"    Decode {num_urls} URLs: {decode_time:.2f}ms")

def test_code_length_analysis():
    """Analyze code length for different approaches"""
    print("\n=== Analyzing Code Lengths ===")
    
    implementations = [
        ("Counter-based", Codec),
        ("Base62", CodecBase62),
        ("Random", CodecRandom),
        ("Hash-based", CodecHash)
    ]
    
    num_urls = 1000
    test_urls = [f"https://test{i}.com" for i in range(num_urls)]
    
    for name, CodecClass in implementations:
        codec = CodecClass()
        
        code_lengths = []
        for url in test_urls:
            short_url = codec.encode(url)
            code = short_url.replace(codec.base_url, "")
            code_lengths.append(len(code))
        
        avg_length = sum(code_lengths) / len(code_lengths)
        min_length = min(code_lengths)
        max_length = max(code_lengths)
        
        print(f"  {name}:")
        print(f"    Average code length: {avg_length:.1f}")
        print(f"    Min/Max length: {min_length}/{max_length}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Social media link sharing
    print("Application 1: Social Media Link Sharing")
    social_codec = CodecRandom()
    
    long_url = "https://news.example.com/article/very-long-article-title-with-many-words-and-details"
    short_url = social_codec.encode(long_url)
    
    print(f"  Original: {long_url}")
    print(f"  Shortened: {short_url}")
    print(f"  Space saved: {len(long_url) - len(short_url)} characters")
    
    # Application 2: Email campaigns with tracking
    print(f"\nApplication 2: Email Campaign Tracking")
    campaign_codec = CodecBase62()
    
    campaign_urls = [
        "https://shop.example.com/product/123?utm_source=email&utm_campaign=summer_sale",
        "https://blog.example.com/article/tips?utm_source=email&utm_campaign=newsletter",
        "https://app.example.com/signup?utm_source=email&utm_campaign=onboarding"
    ]
    
    for i, url in enumerate(campaign_urls, 1):
        short_url = campaign_codec.encode(url)
        print(f"  Campaign link {i}: {short_url}")
    
    # Application 3: QR code generation
    print(f"\nApplication 3: QR Code URLs")
    qr_codec = CodecWithExpiration()
    
    # Short URLs are better for QR codes
    qr_urls = [
        "https://restaurant.example.com/menu/daily-specials",
        "https://event.example.com/conference/2024/schedule"
    ]
    
    for url in qr_urls:
        short_url = qr_codec.encode(url, expiration_seconds=86400)  # 24 hours
        print(f"  QR URL: {short_url}")

def benchmark_collision_rates():
    """Benchmark collision rates for different approaches"""
    print("\n=== Benchmarking Collision Rates ===")
    
    # Test hash-based approach with different hash lengths
    hash_lengths = [4, 6, 8]
    num_urls = 10000
    
    for length in hash_lengths:
        class CustomHashCodec(CodecHash):
            def encode(self, longUrl: str) -> str:
                if longUrl in self.url_to_code:
                    return self.base_url + self.url_to_code[longUrl]
                
                hash_value = hashlib.md5(longUrl.encode()).hexdigest()
                code = hash_value[:length]  # Use custom length
                
                original_code = code
                suffix = 0
                while code in self.code_to_url:
                    suffix += 1
                    code = original_code + str(suffix)
                
                self.url_to_code[longUrl] = code
                self.code_to_url[code] = longUrl
                
                return self.base_url + code
        
        codec = CustomHashCodec()
        collisions = 0
        
        for i in range(num_urls):
            url = f"https://test{i}.example.com"
            short_url = codec.encode(url)
            code = short_url.replace(codec.base_url, "")
            
            # Count codes with suffixes (indicating collisions)
            if any(char.isdigit() for char in code[length:]):
                collisions += 1
        
        collision_rate = (collisions / num_urls) * 100
        print(f"  Hash length {length}: {collisions} collisions ({collision_rate:.2f}%)")

if __name__ == "__main__":
    test_basic_functionality()
    test_duplicate_encoding()
    test_collision_handling()
    test_expiration_functionality()
    test_performance()
    test_code_length_analysis()
    demonstrate_applications()
    benchmark_collision_rates()

"""
URL Encoding/Decoding Design demonstrates key concepts:

Core Approaches:
1. Counter-based - Simple sequential numbering
2. Base62 - Compact encoding using alphanumeric characters
3. Random - Unpredictable codes for security
4. Hash-based - Deterministic encoding based on URL content
5. With Expiration - Time-limited URLs for security

Key Design Considerations:
- Code length vs. collision probability
- Predictability vs. security
- Performance vs. features
- Memory usage and cleanup

Real-world Applications:
- Social media link sharing (Twitter, Facebook)
- Email marketing campaigns
- QR code generation
- Temporary file sharing
- API endpoint shortening
- Analytics and click tracking

The choice of approach depends on specific requirements
like security, code length, and expected volume.
"""
