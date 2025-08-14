"""
Trie Applications and Real-World Use Cases
==========================================

Topics: Auto-complete, spell check, IP routing, URL routing, search engines
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber
Difficulty: Medium to Hard
Time Complexity: Varies by application (O(m) to O(n log n))
Space Complexity: O(n*m) for trie storage plus application-specific space
"""

from typing import List, Tuple, Optional, Dict, Any, Set, Union
from collections import defaultdict, Counter, deque
import heapq
import time

class TrieApplications:
    
    def __init__(self):
        """Initialize with application tracking"""
        self.application_count = 0
        self.performance_metrics = {}
    
    # ==========================================
    # 1. AUTO-COMPLETE SYSTEM
    # ==========================================
    
    def demonstrate_autocomplete_system(self) -> None:
        """
        Demonstrate a complete auto-complete system
        
        Features:
        - Frequency-based ranking
        - Recency tracking
        - Typo tolerance
        - Performance optimization
        """
        print("=== AUTO-COMPLETE SYSTEM ===")
        print("Features: Frequency ranking, recency, typo tolerance")
        print()
        
        autocomplete = AutoCompleteSystem()
        
        # Simulate user input and learning
        user_inputs = [
            "apple", "application", "apply", "apple", "app", "application",
            "amazon", "amazing", "alphabet", "apple", "android", "apple"
        ]
        
        print("Training the auto-complete system:")
        for input_word in user_inputs:
            autocomplete.input(input_word)
            print(f"  User typed: '{input_word}' (frequency updated)")
        
        print(f"\nSystem learned from {len(user_inputs)} inputs")
        print()
        
        # Test auto-complete suggestions
        test_queries = ["app", "a", "amaz", "xyz"]
        
        print("Auto-complete suggestions:")
        for query in test_queries:
            suggestions = autocomplete.get_suggestions(query)
            print(f"  Query '{query}': {suggestions}")
        
        print()
        
        # Test typo tolerance
        print("Typo tolerance (edit distance ≤ 1):")
        typo_queries = ["aple", "aplicaton", "amzon"]
        for query in typo_queries:
            suggestions = autocomplete.get_suggestions_with_typo_tolerance(query)
            print(f"  Query '{query}': {suggestions}")


class AutoCompleteTrieNode:
    """Trie node optimized for auto-complete"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.last_accessed = 0
        self.suggestions = []  # Cache top suggestions for this prefix


class AutoCompleteSystem:
    """
    Production-ready auto-complete system
    
    Features:
    - Weighted ranking (frequency + recency)
    - Suggestion caching for performance
    - Typo tolerance using edit distance
    - Incremental learning from user inputs
    """
    
    def __init__(self, max_suggestions: int = 5):
        self.root = AutoCompleteTrieNode()
        self.max_suggestions = max_suggestions
        self.time_counter = 0
        self.total_inputs = 0
    
    def input(self, word: str) -> None:
        """
        Process user input and update frequencies
        
        Time: O(m), Space: O(m)
        """
        if not word:
            return
        
        word = word.lower().strip()
        current = self.root
        self.time_counter += 1
        self.total_inputs += 1
        
        # Navigate and create path
        for char in word:
            if char not in current.children:
                current.children[char] = AutoCompleteTrieNode()
            current = current.children[char]
        
        # Update word statistics
        current.is_end_of_word = True
        current.frequency += 1
        current.last_accessed = self.time_counter
        
        # Invalidate cached suggestions along the path
        self._invalidate_suggestions_cache(word)
    
    def get_suggestions(self, prefix: str, use_cache: bool = True) -> List[str]:
        """
        Get top suggestions for given prefix
        
        Time: O(p + k log k) where p=prefix length, k=suggestions found
        Space: O(k)
        """
        if not prefix:
            return []
        
        prefix = prefix.lower().strip()
        current = self.root
        
        # Navigate to prefix node
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]
        
        # Check cache first
        if use_cache and current.suggestions:
            return current.suggestions[:self.max_suggestions]
        
        # Collect all words with this prefix
        words_with_scores = []
        
        def _collect_words(node: AutoCompleteTrieNode, current_word: str):
            if node.is_end_of_word:
                # Calculate score: frequency + recency bonus
                recency_bonus = (node.last_accessed / max(1, self.time_counter)) * 10
                score = node.frequency + recency_bonus
                words_with_scores.append((current_word, score, node.frequency))
            
            for char, child in node.children.items():
                _collect_words(child, current_word + char)
        
        _collect_words(current, prefix)
        
        # Sort by score and cache results
        words_with_scores.sort(key=lambda x: x[1], reverse=True)
        suggestions = [word for word, score, freq in words_with_scores[:self.max_suggestions]]
        
        # Cache suggestions
        current.suggestions = suggestions
        
        return suggestions
    
    def get_suggestions_with_typo_tolerance(self, query: str, max_edit_distance: int = 1) -> List[str]:
        """
        Get suggestions with typo tolerance using edit distance
        
        Time: O(n * m * d) where n=words, m=word length, d=edit distance
        Space: O(m * d)
        """
        if not query:
            return []
        
        query = query.lower().strip()
        suggestions_with_distance = []
        
        def _edit_distance(s1: str, s2: str) -> int:
            """Calculate edit distance between two strings"""
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            
            distances = list(range(len(s1) + 1))
            
            for i2, c2 in enumerate(s2):
                new_distances = [i2 + 1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        new_distances.append(distances[i1])
                    else:
                        new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
                distances = new_distances
            
            return distances[-1]
        
        def _collect_words_with_distance(node: AutoCompleteTrieNode, prefix: str):
            if node.is_end_of_word:
                distance = _edit_distance(query, prefix)
                if distance <= max_edit_distance:
                    score = node.frequency + (1.0 / (distance + 1)) * 5  # Distance bonus
                    suggestions_with_distance.append((prefix, score, distance))
            
            for char, child in node.children.items():
                if len(prefix) < len(query) + max_edit_distance:  # Pruning
                    _collect_words_with_distance(child, prefix + char)
        
        _collect_words_with_distance(self.root, "")
        
        # Sort by score and return top suggestions
        suggestions_with_distance.sort(key=lambda x: (x[2], -x[1]))  # Distance first, then score
        return [word for word, score, distance in suggestions_with_distance[:self.max_suggestions]]
    
    def _invalidate_suggestions_cache(self, word: str) -> None:
        """Invalidate cached suggestions for all prefixes of word"""
        current = self.root
        
        for i, char in enumerate(word):
            if char in current.children:
                current = current.children[char]
                current.suggestions = []  # Clear cache
            else:
                break
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        def _count_nodes(node: AutoCompleteTrieNode) -> int:
            count = 1
            for child in node.children.values():
                count += _count_nodes(child)
            return count
        
        total_nodes = _count_nodes(self.root)
        
        return {
            'total_inputs': self.total_inputs,
            'total_nodes': total_nodes,
            'time_counter': self.time_counter,
            'cache_efficiency': 'Not measured'  # Would need cache hit/miss tracking
        }


# ==========================================
# 2. SPELL CHECKER SYSTEM
# ==========================================

class SpellCheckerSystem:
    """
    Advanced spell checker using trie with multiple correction strategies
    
    Features:
    - Multiple edit distance algorithms
    - Phonetic similarity (Soundex)
    - Context-aware suggestions
    - Custom dictionary support
    """
    
    def __init__(self, dictionary: List[str]):
        self.trie = AutoCompleteTrieNode()
        self.word_set = set()
        self.soundex_map = defaultdict(list)
        
        print("Building spell checker dictionary...")
        self._build_dictionary(dictionary)
        print(f"  Loaded {len(dictionary)} words")
        print(f"  Created {len(self.soundex_map)} soundex groups")
    
    def _build_dictionary(self, dictionary: List[str]) -> None:
        """Build trie and soundex mapping from dictionary"""
        for word in dictionary:
            word = word.lower().strip()
            if word:
                self._insert_word(word)
                self.word_set.add(word)
                soundex_code = self._calculate_soundex(word)
                self.soundex_map[soundex_code].append(word)
    
    def _insert_word(self, word: str) -> None:
        """Insert word into trie"""
        current = self.trie
        
        for char in word:
            if char not in current.children:
                current.children[char] = AutoCompleteTrieNode()
            current = current.children[char]
        
        current.is_end_of_word = True
        current.frequency = 1
    
    def _calculate_soundex(self, word: str) -> str:
        """
        Calculate Soundex code for phonetic similarity
        
        Soundex algorithm converts words to phonetic representation
        """
        if not word:
            return "0000"
        
        word = word.upper()
        soundex = word[0]  # Keep first letter
        
        # Soundex mapping
        mapping = {
            'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
            'L': '4', 'MN': '5', 'R': '6'
        }
        
        # Create mapping dict
        char_to_code = {}
        for chars, code in mapping.items():
            for char in chars:
                char_to_code[char] = code
        
        # Convert remaining characters
        for char in word[1:]:
            if char in char_to_code:
                code = char_to_code[char]
                if code != soundex[-1]:  # Avoid consecutive duplicates
                    soundex += code
        
        # Pad or truncate to 4 characters
        soundex = (soundex + "000")[:4]
        return soundex
    
    def check_spelling(self, word: str) -> Tuple[bool, List[str]]:
        """
        Check spelling and provide suggestions
        
        Returns: (is_correct, suggestions)
        """
        word = word.lower().strip()
        
        if word in self.word_set:
            return True, []
        
        print(f"Spell checking '{word}':")
        print(f"  ✗ Not found in dictionary")
        
        # Generate suggestions using multiple strategies
        suggestions = set()
        
        # Strategy 1: Edit distance suggestions
        edit_suggestions = self._get_edit_distance_suggestions(word, max_distance=2)
        suggestions.update(edit_suggestions[:3])
        print(f"  Edit distance suggestions: {edit_suggestions[:3]}")
        
        # Strategy 2: Phonetic suggestions (Soundex)
        soundex_code = self._calculate_soundex(word)
        phonetic_suggestions = self.soundex_map.get(soundex_code, [])[:3]
        suggestions.update(phonetic_suggestions)
        print(f"  Phonetic suggestions (soundex {soundex_code}): {phonetic_suggestions}")
        
        # Strategy 3: Prefix-based suggestions
        prefix_suggestions = self._get_prefix_suggestions(word)
        suggestions.update(prefix_suggestions[:2])
        print(f"  Prefix suggestions: {prefix_suggestions[:2]}")
        
        final_suggestions = list(suggestions)[:5]
        print(f"  Final suggestions: {final_suggestions}")
        
        return False, final_suggestions
    
    def _get_edit_distance_suggestions(self, word: str, max_distance: int = 2) -> List[str]:
        """Get suggestions based on edit distance"""
        suggestions = []
        
        def _edit_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance"""
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            
            distances = list(range(len(s1) + 1))
            
            for i2, c2 in enumerate(s2):
                new_distances = [i2 + 1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        new_distances.append(distances[i1])
                    else:
                        new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
                distances = new_distances
            
            return distances[-1]
        
        for dict_word in self.word_set:
            if abs(len(word) - len(dict_word)) <= max_distance:
                distance = _edit_distance(word, dict_word)
                if distance <= max_distance:
                    suggestions.append((dict_word, distance))
        
        # Sort by distance and return words only
        suggestions.sort(key=lambda x: x[1])
        return [word for word, distance in suggestions[:10]]
    
    def _get_prefix_suggestions(self, word: str) -> List[str]:
        """Get suggestions based on common prefixes"""
        suggestions = []
        
        # Try different prefix lengths
        for prefix_len in range(min(3, len(word)), len(word)):
            prefix = word[:prefix_len]
            
            # Find words with this prefix
            current = self.trie
            for char in prefix:
                if char not in current.children:
                    break
                current = current.children[char]
            else:
                # Collect words with this prefix
                def _collect_prefix_words(node: AutoCompleteTrieNode, current_word: str):
                    if node.is_end_of_word and len(suggestions) < 5:
                        suggestions.append(current_word)
                    
                    for char, child in node.children.items():
                        if len(suggestions) < 5:
                            _collect_prefix_words(child, current_word + char)
                
                _collect_prefix_words(current, prefix)
                
                if suggestions:
                    break
        
        return suggestions[:5]


# ==========================================
# 3. IP ROUTING TABLE
# ==========================================

class IPRoutingTable:
    """
    IP routing table implementation using trie for longest prefix matching
    
    Used in network routers for efficient IP address lookup
    """
    
    class RouteEntry:
        def __init__(self, network: str, gateway: str, interface: str, metric: int = 1):
            self.network = network
            self.gateway = gateway
            self.interface = interface
            self.metric = metric
    
    class IPTrieNode:
        def __init__(self):
            self.children = {'0': None, '1': None}  # Binary trie
            self.route_entry = None  # Route information if this is a valid prefix
    
    def __init__(self):
        self.root = self.IPTrieNode()
        self.route_count = 0
    
    def _ip_to_binary(self, ip: str, prefix_length: int) -> str:
        """Convert IP address to binary string"""
        parts = ip.split('.')
        binary = ''
        
        for part in parts:
            binary += format(int(part), '08b')
        
        return binary[:prefix_length]
    
    def add_route(self, network: str, gateway: str, interface: str, metric: int = 1) -> None:
        """
        Add route to routing table
        
        Format: "192.168.1.0/24" -> network with prefix length
        """
        print(f"Adding route: {network} via {gateway} ({interface})")
        
        # Parse network and prefix length
        ip, prefix_length = network.split('/')
        prefix_length = int(prefix_length)
        
        # Convert to binary
        binary_prefix = self._ip_to_binary(ip, prefix_length)
        print(f"  Binary prefix ({prefix_length} bits): {binary_prefix}")
        
        # Insert into trie
        current = self.root
        
        for bit in binary_prefix:
            if current.children[bit] is None:
                current.children[bit] = self.IPTrieNode()
            current = current.children[bit]
        
        # Store route entry
        current.route_entry = self.RouteEntry(network, gateway, interface, metric)
        self.route_count += 1
        print(f"  Route added successfully")
    
    def lookup_route(self, ip: str) -> Optional['IPRoutingTable.RouteEntry']:
        """
        Perform longest prefix matching for IP lookup
        
        Time: O(32) = O(1) for IPv4
        Space: O(1)
        """
        print(f"Looking up route for IP: {ip}")
        
        # Convert IP to binary
        binary_ip = self._ip_to_binary(ip, 32)
        print(f"  Binary IP: {binary_ip}")
        
        # Traverse trie to find longest matching prefix
        current = self.root
        best_match = None
        
        for i, bit in enumerate(binary_ip):
            # Check if current node has a route
            if current.route_entry is not None:
                best_match = current.route_entry
                print(f"    Found match at bit {i}: {best_match.network}")
            
            # Continue traversal
            if current.children[bit] is None:
                break
            current = current.children[bit]
        
        # Check final node
        if current.route_entry is not None:
            best_match = current.route_entry
            print(f"    Found final match: {best_match.network}")
        
        if best_match:
            print(f"  ✓ Route found: {best_match.network} via {best_match.gateway}")
            return best_match
        else:
            print(f"  ✗ No route found")
            return None
    
    def display_routing_table(self) -> None:
        """Display all routes in the routing table"""
        print("Routing Table:")
        print(f"{'Network':<18} {'Gateway':<15} {'Interface':<10} {'Metric'}")
        print("-" * 60)
        
        def _collect_routes(node: IPRoutingTable.IPTrieNode, prefix: str):
            if node.route_entry:
                entry = node.route_entry
                print(f"{entry.network:<18} {entry.gateway:<15} {entry.interface:<10} {entry.metric}")
            
            for bit, child in node.children.items():
                if child is not None:
                    _collect_routes(child, prefix + bit)
        
        _collect_routes(self.root, "")
        print(f"\nTotal routes: {self.route_count}")


# ==========================================
# 4. URL ROUTING SYSTEM
# ==========================================

class URLRoutingSystem:
    """
    Web application URL routing using trie for fast path matching
    
    Supports:
    - Static routes: /users/profile
    - Dynamic routes: /users/{id}/posts
    - Wildcard routes: /static/*
    - Route parameters and query handling
    """
    
    class RouteHandler:
        def __init__(self, handler_name: str, method: str, parameters: List[str] = None):
            self.handler_name = handler_name
            self.method = method
            self.parameters = parameters or []
    
    class URLTrieNode:
        def __init__(self):
            self.children = {}
            self.route_handler = None
            self.is_parameter = False  # True if this is a {param} node
            self.parameter_name = None
            self.is_wildcard = False  # True if this is a * node
    
    def __init__(self):
        self.root = self.URLTrieNode()
        self.route_count = 0
    
    def add_route(self, path: str, handler_name: str, method: str = "GET") -> None:
        """
        Add route to URL routing system
        
        Examples:
        - /users/profile
        - /users/{id}/posts
        - /static/*
        """
        print(f"Adding route: {method} {path} -> {handler_name}")
        
        # Parse path segments
        segments = [seg for seg in path.split('/') if seg]
        current = self.root
        parameters = []
        
        for segment in segments:
            if segment.startswith('{') and segment.endswith('}'):
                # Dynamic parameter segment
                param_name = segment[1:-1]
                parameters.append(param_name)
                
                # Use special key for parameter nodes
                param_key = f"<{param_name}>"
                
                if param_key not in current.children:
                    current.children[param_key] = self.URLTrieNode()
                    current.children[param_key].is_parameter = True
                    current.children[param_key].parameter_name = param_name
                
                current = current.children[param_key]
                print(f"  Added parameter segment: {param_name}")
            
            elif segment == '*':
                # Wildcard segment
                if '*' not in current.children:
                    current.children['*'] = self.URLTrieNode()
                    current.children['*'].is_wildcard = True
                
                current = current.children['*']
                print(f"  Added wildcard segment")
            
            else:
                # Static segment
                if segment not in current.children:
                    current.children[segment] = self.URLTrieNode()
                
                current = current.children[segment]
                print(f"  Added static segment: {segment}")
        
        # Store route handler
        current.route_handler = self.RouteHandler(handler_name, method, parameters)
        self.route_count += 1
        print(f"  Route registered successfully")
    
    def match_route(self, path: str, method: str = "GET") -> Tuple[Optional['URLRoutingSystem.RouteHandler'], Dict[str, str]]:
        """
        Match incoming request to registered route
        
        Returns: (handler, parameters_dict)
        """
        print(f"Matching route: {method} {path}")
        
        segments = [seg for seg in path.split('/') if seg]
        
        def _match_recursive(node: URLRoutingSystem.URLTrieNode, 
                           segment_index: int, 
                           parameters: Dict[str, str]) -> Tuple[Optional['URLRoutingSystem.RouteHandler'], Dict[str, str]]:
            
            # Base case: reached end of path
            if segment_index == len(segments):
                if node.route_handler and node.route_handler.method == method:
                    return node.route_handler, parameters
                return None, {}
            
            current_segment = segments[segment_index]
            
            # Try exact match first
            if current_segment in node.children:
                result = _match_recursive(node.children[current_segment], 
                                        segment_index + 1, 
                                        parameters.copy())
                if result[0]:
                    return result
            
            # Try parameter match
            for child_key, child_node in node.children.items():
                if child_node.is_parameter:
                    new_params = parameters.copy()
                    new_params[child_node.parameter_name] = current_segment
                    
                    result = _match_recursive(child_node, 
                                            segment_index + 1, 
                                            new_params)
                    if result[0]:
                        print(f"    Matched parameter {child_node.parameter_name} = {current_segment}")
                        return result
            
            # Try wildcard match (matches remaining path)
            if '*' in node.children:
                wildcard_node = node.children['*']
                if wildcard_node.route_handler and wildcard_node.route_handler.method == method:
                    remaining_path = '/'.join(segments[segment_index:])
                    print(f"    Matched wildcard with remaining path: {remaining_path}")
                    return wildcard_node.route_handler, parameters
            
            return None, {}
        
        handler, params = _match_recursive(self.root, 0, {})
        
        if handler:
            print(f"  ✓ Matched: {handler.handler_name}")
            if params:
                print(f"    Parameters: {params}")
        else:
            print(f"  ✗ No matching route found")
        
        return handler, params
    
    def display_routes(self) -> None:
        """Display all registered routes"""
        print("Registered Routes:")
        print(f"{'Method':<6} {'Path':<30} {'Handler':<20}")
        print("-" * 60)
        
        def _collect_routes(node: URLRoutingSystem.URLTrieNode, path_segments: List[str]):
            if node.route_handler:
                path = '/' + '/'.join(path_segments) if path_segments else '/'
                handler = node.route_handler
                print(f"{handler.method:<6} {path:<30} {handler.handler_name:<20}")
            
            for segment, child in node.children.items():
                new_segments = path_segments + [segment]
                _collect_routes(child, new_segments)
        
        _collect_routes(self.root, [])
        print(f"\nTotal routes: {self.route_count}")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_trie_applications():
    """Demonstrate all real-world trie applications"""
    print("=== TRIE APPLICATIONS DEMONSTRATION ===\n")
    
    applications = TrieApplications()
    
    # 1. Auto-complete system
    applications.demonstrate_autocomplete_system()
    print("\n" + "="*60 + "\n")
    
    # 2. Spell checker
    print("=== SPELL CHECKER SYSTEM ===")
    
    dictionary = [
        "apple", "application", "apply", "appreciate", "approach", "appropriate",
        "amazon", "amazing", "ambulance", "america", "amount", "animal",
        "example", "excellent", "exercise", "experience", "explain", "express"
    ]
    
    spell_checker = SpellCheckerSystem(dictionary)
    
    test_words = ["aple", "aplicaton", "amzing", "excelent", "correct"]
    
    print("\nSpell checking tests:")
    for word in test_words:
        is_correct, suggestions = spell_checker.check_spelling(word)
        if is_correct:
            print(f"'{word}': ✓ Correct spelling")
        else:
            print(f"'{word}': ✗ Misspelled, suggestions: {suggestions}")
        print()
    
    print("="*60 + "\n")
    
    # 3. IP Routing Table
    print("=== IP ROUTING TABLE ===")
    
    routing_table = IPRoutingTable()
    
    # Add sample routes
    routes = [
        ("192.168.1.0/24", "192.168.1.1", "eth0"),
        ("192.168.0.0/16", "10.0.0.1", "eth1"),
        ("10.0.0.0/8", "172.16.0.1", "eth2"),
        ("0.0.0.0/0", "8.8.8.8", "wan0")  # Default route
    ]
    
    print("Building routing table:")
    for network, gateway, interface in routes:
        routing_table.add_route(network, gateway, interface)
        print()
    
    routing_table.display_routing_table()
    print()
    
    # Test IP lookups
    test_ips = ["192.168.1.100", "192.168.5.1", "10.1.1.1", "8.8.8.8"]
    
    print("IP route lookups:")
    for ip in test_ips:
        routing_table.lookup_route(ip)
        print()
    
    print("="*60 + "\n")
    
    # 4. URL Routing System
    print("=== URL ROUTING SYSTEM ===")
    
    url_router = URLRoutingSystem()
    
    # Add sample routes
    routes = [
        ("/", "home_handler", "GET"),
        ("/users", "list_users", "GET"),
        ("/users/{id}", "get_user", "GET"),
        ("/users/{id}/posts", "get_user_posts", "GET"),
        ("/users/{id}/posts/{post_id}", "get_post", "GET"),
        ("/api/v1/*", "api_handler", "GET"),
        ("/static/*", "static_file_handler", "GET")
    ]
    
    print("Registering URL routes:")
    for path, handler, method in routes:
        url_router.add_route(path, handler, method)
        print()
    
    url_router.display_routes()
    print()
    
    # Test URL matching
    test_urls = [
        "/",
        "/users",
        "/users/123",
        "/users/456/posts",
        "/users/789/posts/101",
        "/api/v1/data/export",
        "/static/css/style.css",
        "/nonexistent"
    ]
    
    print("URL route matching:")
    for url in test_urls:
        handler, params = url_router.match_route(url)
        print(f"URL: {url}")
        if handler:
            print(f"  Handler: {handler.handler_name}")
            if params:
                print(f"  Parameters: {params}")
        else:
            print(f"  No matching route")
        print()


if __name__ == "__main__":
    demonstrate_trie_applications()
    
    print("\n=== TRIE APPLICATIONS MASTERY GUIDE ===")
    
    print("\n🎯 APPLICATION CATEGORIES:")
    print("• Text Processing: Auto-complete, spell check, search suggestions")
    print("• Network Systems: IP routing, URL routing, DNS resolution")
    print("• Web Development: Route matching, template engines")
    print("• Data Storage: Prefix-based databases, key-value stores")
    print("• Security: Pattern matching, intrusion detection")
    
    print("\n📊 PERFORMANCE CHARACTERISTICS:")
    print("• Auto-complete: O(p + k log k) where p=prefix, k=suggestions")
    print("• Spell check: O(n * m * d) where n=words, m=length, d=edit distance")
    print("• IP routing: O(32) = O(1) for IPv4 longest prefix matching")
    print("• URL routing: O(path_segments) for route matching")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Cache frequently accessed results")
    print("• Use compressed tries for memory efficiency")
    print("• Implement lazy loading for large dictionaries")
    print("• Add bloom filters for negative lookups")
    print("• Use parallel processing for bulk operations")
    
    print("\n🔧 IMPLEMENTATION CONSIDERATIONS:")
    print("• Handle Unicode and internationalization")
    print("• Implement proper memory management")
    print("• Add comprehensive error handling")
    print("• Consider thread safety for concurrent access")
    print("• Design for horizontal scalability")
    
    print("\n🏆 REAL-WORLD SYSTEMS:")
    print("• Search Engines: Query auto-complete, spell correction")
    print("• Social Media: Username suggestions, hashtag completion")
    print("• E-commerce: Product search, category navigation")
    print("• IDE/Editors: Code completion, symbol lookup")
    print("• Networking: Router tables, DNS caching")
    
    print("\n🎓 ADVANCED TOPICS:")
    print("• Distributed trie systems")
    print("• Persistent trie data structures")
    print("• Approximate string matching algorithms")
    print("• Machine learning integration for rankings")
    print("• Real-time synchronization and updates")
