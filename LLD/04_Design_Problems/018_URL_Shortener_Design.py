"""
URL SHORTENER DESIGN - Complete System Design
============================================

Problem Statement:
Design a comprehensive URL shortening service that handles:
- URL shortening with custom and auto-generated short codes
- URL expansion and redirection with analytics
- User management and authentication
- URL analytics and click tracking
- Custom domains and branded short links
- URL expiration and lifecycle management
- Bulk URL operations and API access
- Rate limiting and abuse prevention
- URL validation and security scanning
- High availability and scalability

Requirements:
- Support both custom and auto-generated short codes
- Provide comprehensive analytics (clicks, geographic data, referrers)
- Handle high-throughput URL redirections efficiently
- Support custom domains and white-labeling
- Implement URL expiration and access controls
- Provide RESTful API for integration
- Handle malicious URL detection and blocking
- Support bulk operations for enterprise users
- Implement user management with different access levels
- Ensure data persistence and backup

Design Patterns Used:
- Factory: Short code generation strategies
- Strategy: Different encoding and analytics strategies
- Observer: Analytics and event tracking
- Decorator: URL validation and security checks
- Repository: Data persistence abstraction
- Facade: Simplified API interface
- Command: URL operations with undo capability
- Proxy: URL validation and security proxy
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import base64
import string
import random
import re
import requests
import json
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import threading
import time
from collections import defaultdict, Counter


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class URLStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    DISABLED = "disabled"
    BLOCKED = "blocked"
    PENDING = "pending"


class UserRole(Enum):
    ANONYMOUS = "anonymous"
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class AnalyticsEvent(Enum):
    CLICK = "click"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BLOCK = "block"


class SecurityThreat(Enum):
    MALWARE = "malware"
    PHISHING = "phishing"
    SPAM = "spam"
    ADULT = "adult"
    SUSPICIOUS = "suspicious"


@dataclass
class ShortenedURL:
    """Shortened URL data model."""
    short_code: str
    original_url: str
    user_id: Optional[str] = None
    custom_domain: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: URLStatus = URLStatus.ACTIVE
    
    # Analytics
    click_count: int = 0
    last_clicked: Optional[datetime] = None
    
    # Security
    is_safe: bool = True
    security_threats: List[SecurityThreat] = field(default_factory=list)
    last_security_check: Optional[datetime] = None
    
    # Settings
    password: Optional[str] = None
    max_clicks: Optional[int] = None
    
    def __post_init__(self):
        if not self.short_code:
            self.short_code = str(uuid.uuid4())[:8]


@dataclass
class ClickEvent:
    """URL click event data."""
    short_code: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    referrer: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    device_type: Optional[str] = None
    browser: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class User:
    """User data model."""
    user_id: str
    email: str
    role: UserRole = UserRole.FREE
    api_key: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # Limits and quotas
    monthly_url_limit: int = 100
    monthly_clicks_limit: int = 1000
    custom_domain_limit: int = 0
    
    # Current usage
    urls_created_this_month: int = 0
    clicks_this_month: int = 0
    
    # Settings
    default_expiration: Optional[timedelta] = None
    analytics_enabled: bool = True
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = hashlib.sha256(f"{self.user_id}{self.email}".encode()).hexdigest()[:32]


@dataclass
class CustomDomain:
    """Custom domain configuration."""
    domain: str
    user_id: str
    is_verified: bool = False
    ssl_enabled: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    verification_token: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AnalyticsData:
    """Analytics aggregation data."""
    total_clicks: int = 0
    unique_clicks: int = 0
    clicks_by_day: Dict[str, int] = field(default_factory=dict)
    clicks_by_country: Dict[str, int] = field(default_factory=dict)
    clicks_by_referrer: Dict[str, int] = field(default_factory=dict)
    clicks_by_device: Dict[str, int] = field(default_factory=dict)
    clicks_by_browser: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# SHORT CODE GENERATION STRATEGIES
# ============================================================================

class ShortCodeGenerator(ABC):
    """Abstract short code generator."""
    
    @abstractmethod
    def generate(self, url: str, custom_code: str = None) -> str:
        """Generate short code for URL."""
        pass
    
    @abstractmethod
    def is_valid_custom_code(self, code: str) -> bool:
        """Validate custom short code."""
        pass


class Base62Generator(ShortCodeGenerator):
    """Base62 short code generator."""
    
    def __init__(self, length: int = 6):
        self.length = length
        self.charset = string.ascii_letters + string.digits
    
    def generate(self, url: str, custom_code: str = None) -> str:
        """Generate base62 short code."""
        if custom_code:
            if self.is_valid_custom_code(custom_code):
                return custom_code
            else:
                raise ValueError("Invalid custom code")
        
        # Generate hash-based code
        hash_value = hashlib.md5(url.encode()).hexdigest()
        code = ""
        
        for i in range(self.length):
            index = int(hash_value[i * 2:i * 2 + 2], 16) % len(self.charset)
            code += self.charset[index]
        
        return code
    
    def is_valid_custom_code(self, code: str) -> bool:
        """Validate custom code."""
        if not code or len(code) < 3 or len(code) > 20:
            return False
        
        return all(c in self.charset for c in code)


class RandomGenerator(ShortCodeGenerator):
    """Random short code generator."""
    
    def __init__(self, length: int = 6):
        self.length = length
        self.charset = string.ascii_letters + string.digits
    
    def generate(self, url: str, custom_code: str = None) -> str:
        """Generate random short code."""
        if custom_code:
            if self.is_valid_custom_code(custom_code):
                return custom_code
            else:
                raise ValueError("Invalid custom code")
        
        return ''.join(random.choices(self.charset, k=self.length))
    
    def is_valid_custom_code(self, code: str) -> bool:
        """Validate custom code."""
        if not code or len(code) < 3 or len(code) > 20:
            return False
        
        return all(c in self.charset for c in code)


class IncrementalGenerator(ShortCodeGenerator):
    """Incremental short code generator."""
    
    def __init__(self, start_value: int = 1000000):
        self.current_value = start_value
        self.charset = string.ascii_letters + string.digits
        self._lock = threading.Lock()
    
    def generate(self, url: str, custom_code: str = None) -> str:
        """Generate incremental short code."""
        if custom_code:
            if self.is_valid_custom_code(custom_code):
                return custom_code
            else:
                raise ValueError("Invalid custom code")
        
        with self._lock:
            value = self.current_value
            self.current_value += 1
            return self._encode_number(value)
    
    def _encode_number(self, number: int) -> str:
        """Encode number to base62."""
        if number == 0:
            return self.charset[0]
        
        result = ""
        base = len(self.charset)
        
        while number > 0:
            result = self.charset[number % base] + result
            number //= base
        
        return result
    
    def is_valid_custom_code(self, code: str) -> bool:
        """Validate custom code."""
        if not code or len(code) < 3 or len(code) > 20:
            return False
        
        return all(c in self.charset for c in code)


# ============================================================================
# URL VALIDATION AND SECURITY
# ============================================================================

class URLValidator:
    """URL validation and security checker."""
    
    def __init__(self):
        self.blocked_domains = set([
            "malware.com", "phishing.com", "spam.com"  # Example blocked domains
        ])
        
        self.suspicious_patterns = [
            r"bit\.ly/[A-Za-z0-9]{6,}",  # Suspicious shortened URLs
            r"tinyurl\.com/[A-Za-z0-9]{6,}",
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # Raw IP addresses
        ]
    
    def validate_url(self, url: str) -> Tuple[bool, List[str]]:
        """Validate URL and return issues if any."""
        issues = []
        
        # Basic URL format validation
        if not self._is_valid_url_format(url):
            issues.append("Invalid URL format")
        
        # Check for blocked domains
        parsed = urlparse(url)
        if parsed.netloc.lower() in self.blocked_domains:
            issues.append("Domain is blocked")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                issues.append("URL contains suspicious patterns")
                break
        
        # Check URL accessibility (simplified)
        if not self._is_url_accessible(url):
            issues.append("URL is not accessible")
        
        return len(issues) == 0, issues
    
    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL format is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_url_accessible(self, url: str) -> bool:
        """Check if URL is accessible (simplified check)."""
        try:
            # In a real implementation, you might use a headless browser
            # or more sophisticated checking
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code < 400
        except:
            return False
    
    def scan_for_threats(self, url: str) -> List[SecurityThreat]:
        """Scan URL for security threats."""
        threats = []
        
        # Simplified threat detection
        url_lower = url.lower()
        
        if any(keyword in url_lower for keyword in ["malware", "virus", "trojan"]):
            threats.append(SecurityThreat.MALWARE)
        
        if any(keyword in url_lower for keyword in ["phishing", "fake", "scam"]):
            threats.append(SecurityThreat.PHISHING)
        
        if any(keyword in url_lower for keyword in ["spam", "advertisement"]):
            threats.append(SecurityThreat.SPAM)
        
        if any(keyword in url_lower for keyword in ["adult", "xxx", "porn"]):
            threats.append(SecurityThreat.ADULT)
        
        # Check for suspicious characteristics
        parsed = urlparse(url)
        if (len(parsed.netloc) > 50 or  # Very long domain
            parsed.netloc.count('.') > 5 or  # Too many subdomains
            any(char in parsed.netloc for char in ['_', '-'] * 3)):  # Suspicious characters
            threats.append(SecurityThreat.SUSPICIOUS)
        
        return threats


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """URL analytics and tracking engine."""
    
    def __init__(self):
        self.click_events: List[ClickEvent] = []
        self.analytics_cache: Dict[str, AnalyticsData] = {}
        self._lock = threading.Lock()
    
    def record_click(self, event: ClickEvent) -> None:
        """Record a click event."""
        with self._lock:
            self.click_events.append(event)
            
            # Update cache
            if event.short_code not in self.analytics_cache:
                self.analytics_cache[event.short_code] = AnalyticsData()
            
            analytics = self.analytics_cache[event.short_code]
            analytics.total_clicks += 1
            
            # Update daily clicks
            day_key = event.timestamp.strftime("%Y-%m-%d")
            analytics.clicks_by_day[day_key] = analytics.clicks_by_day.get(day_key, 0) + 1
            
            # Update geographic data
            if event.country:
                analytics.clicks_by_country[event.country] = analytics.clicks_by_country.get(event.country, 0) + 1
            
            # Update referrer data
            if event.referrer:
                analytics.clicks_by_referrer[event.referrer] = analytics.clicks_by_referrer.get(event.referrer, 0) + 1
            
            # Update device data
            if event.device_type:
                analytics.clicks_by_device[event.device_type] = analytics.clicks_by_device.get(event.device_type, 0) + 1
            
            # Update browser data
            if event.browser:
                analytics.clicks_by_browser[event.browser] = analytics.clicks_by_browser.get(event.browser, 0) + 1
    
    def get_analytics(self, short_code: str, date_range: Tuple[datetime, datetime] = None) -> AnalyticsData:
        """Get analytics for a short code."""
        if short_code in self.analytics_cache:
            analytics = self.analytics_cache[short_code]
            
            if date_range:
                # Filter by date range
                start_date, end_date = date_range
                filtered_analytics = AnalyticsData()
                
                for day, clicks in analytics.clicks_by_day.items():
                    day_date = datetime.strptime(day, "%Y-%m-%d")
                    if start_date <= day_date <= end_date:
                        filtered_analytics.clicks_by_day[day] = clicks
                        filtered_analytics.total_clicks += clicks
                
                return filtered_analytics
            
            return analytics
        
        return AnalyticsData()
    
    def get_top_performing_urls(self, user_id: str = None, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top performing URLs by clicks."""
        url_clicks = defaultdict(int)
        
        for event in self.click_events:
            if user_id is None:  # Global stats
                url_clicks[event.short_code] += 1
        
        return sorted(url_clicks.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_click_trends(self, short_code: str, days: int = 30) -> Dict[str, int]:
        """Get click trends for the past N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        analytics = self.get_analytics(short_code, (start_date, end_date))
        return analytics.clicks_by_day
    
    def parse_user_agent(self, user_agent: str) -> Tuple[str, str]:
        """Parse user agent to extract device type and browser."""
        # Simplified user agent parsing
        user_agent_lower = user_agent.lower()
        
        # Device type detection
        if any(device in user_agent_lower for device in ['mobile', 'android', 'iphone']):
            device_type = 'mobile'
        elif any(device in user_agent_lower for device in ['tablet', 'ipad']):
            device_type = 'tablet'
        else:
            device_type = 'desktop'
        
        # Browser detection
        if 'chrome' in user_agent_lower:
            browser = 'chrome'
        elif 'firefox' in user_agent_lower:
            browser = 'firefox'
        elif 'safari' in user_agent_lower:
            browser = 'safari'
        elif 'edge' in user_agent_lower:
            browser = 'edge'
        else:
            browser = 'other'
        
        return device_type, browser


# ============================================================================
# URL SHORTENER SERVICE
# ============================================================================

class URLShortenerService:
    """Main URL shortener service."""
    
    def __init__(self, base_domain: str = "short.ly"):
        self.base_domain = base_domain
        self.generator = Base62Generator()
        self.validator = URLValidator()
        self.analytics = AnalyticsEngine()
        
        # Storage (in-memory for demo)
        self.urls: Dict[str, ShortenedURL] = {}
        self.users: Dict[str, User] = {}
        self.custom_domains: Dict[str, CustomDomain] = {}
        
        # Caching
        self.url_cache: Dict[str, str] = {}  # short_code -> original_url
        
        # Threading
        self._lock = threading.Lock()
        
        print(f"🔗 URL Shortener Service initialized with domain: {base_domain}")
    
    def create_user(self, email: str, role: UserRole = UserRole.FREE) -> User:
        """Create a new user."""
        user_id = str(uuid.uuid4())
        user = User(user_id=user_id, email=email, role=role)
        
        # Set limits based on role
        if role == UserRole.FREE:
            user.monthly_url_limit = 100
            user.monthly_clicks_limit = 1000
            user.custom_domain_limit = 0
        elif role == UserRole.PREMIUM:
            user.monthly_url_limit = 1000
            user.monthly_clicks_limit = 10000
            user.custom_domain_limit = 1
        elif role == UserRole.ENTERPRISE:
            user.monthly_url_limit = 10000
            user.monthly_clicks_limit = 100000
            user.custom_domain_limit = 10
        
        with self._lock:
            self.users[user_id] = user
        
        return user
    
    def shorten_url(self, original_url: str, user_id: str = None, 
                   custom_code: str = None, custom_domain: str = None,
                   expires_at: datetime = None, **kwargs) -> ShortenedURL:
        """Shorten a URL."""
        # Validate URL
        is_valid, issues = self.validator.validate_url(original_url)
        if not is_valid:
            raise ValueError(f"Invalid URL: {', '.join(issues)}")
        
        # Check user limits
        if user_id:
            user = self.users.get(user_id)
            if user and user.urls_created_this_month >= user.monthly_url_limit:
                raise ValueError("Monthly URL limit exceeded")
        
        # Generate short code
        short_code = self.generator.generate(original_url, custom_code)
        
        # Check if short code already exists
        with self._lock:
            if short_code in self.urls:
                if custom_code:
                    raise ValueError("Custom code already exists")
                else:
                    # Generate a new one
                    attempts = 0
                    while short_code in self.urls and attempts < 10:
                        short_code = self.generator.generate(original_url + str(attempts))
                        attempts += 1
                    
                    if short_code in self.urls:
                        raise ValueError("Failed to generate unique short code")
        
        # Security scan
        threats = self.validator.scan_for_threats(original_url)
        is_safe = len(threats) == 0
        
        # Create shortened URL
        shortened_url = ShortenedURL(
            short_code=short_code,
            original_url=original_url,
            user_id=user_id,
            custom_domain=custom_domain,
            expires_at=expires_at,
            is_safe=is_safe,
            security_threats=threats,
            last_security_check=datetime.now(),
            **kwargs
        )
        
        # Store URL
        with self._lock:
            self.urls[short_code] = shortened_url
            self.url_cache[short_code] = original_url
            
            # Update user stats
            if user_id and user_id in self.users:
                self.users[user_id].urls_created_this_month += 1
        
        return shortened_url
    
    def expand_url(self, short_code: str, track_click: bool = True,
                  request_info: Dict[str, Any] = None) -> Tuple[str, bool]:
        """Expand short URL and optionally track click."""
        with self._lock:
            if short_code not in self.urls:
                raise ValueError("Short code not found")
            
            shortened_url = self.urls[short_code]
            
            # Check if URL is active
            if shortened_url.status != URLStatus.ACTIVE:
                raise ValueError(f"URL is {shortened_url.status.value}")
            
            # Check expiration
            if (shortened_url.expires_at and 
                datetime.now() > shortened_url.expires_at):
                shortened_url.status = URLStatus.EXPIRED
                raise ValueError("URL has expired")
            
            # Check max clicks
            if (shortened_url.max_clicks and 
                shortened_url.click_count >= shortened_url.max_clicks):
                shortened_url.status = URLStatus.DISABLED
                raise ValueError("Maximum clicks reached")
            
            # Check password protection
            if shortened_url.password:
                provided_password = request_info.get('password') if request_info else None
                if provided_password != shortened_url.password:
                    raise ValueError("Password required")
            
            # Track click
            if track_click:
                self._track_click(shortened_url, request_info)
            
            return shortened_url.original_url, shortened_url.is_safe
    
    def _track_click(self, shortened_url: ShortenedURL, request_info: Dict[str, Any] = None) -> None:
        """Track a click event."""
        # Update URL stats
        shortened_url.click_count += 1
        shortened_url.last_clicked = datetime.now()
        
        # Update user stats
        if shortened_url.user_id and shortened_url.user_id in self.users:
            self.users[shortened_url.user_id].clicks_this_month += 1
        
        # Create click event
        if request_info:
            user_agent = request_info.get('user_agent', '')
            device_type, browser = self.analytics.parse_user_agent(user_agent)
            
            click_event = ClickEvent(
                short_code=shortened_url.short_code,
                timestamp=datetime.now(),
                ip_address=request_info.get('ip_address', ''),
                user_agent=user_agent,
                referrer=request_info.get('referrer'),
                country=request_info.get('country'),
                city=request_info.get('city'),
                device_type=device_type,
                browser=browser
            )
            
            self.analytics.record_click(click_event)
    
    def get_url_info(self, short_code: str, user_id: str = None) -> Optional[ShortenedURL]:
        """Get information about a shortened URL."""
        if short_code not in self.urls:
            return None
        
        url = self.urls[short_code]
        
        # Check if user has access
        if user_id and url.user_id and url.user_id != user_id:
            # Check if user is admin
            user = self.users.get(user_id)
            if not user or user.role != UserRole.ADMIN:
                return None
        
        return url
    
    def update_url(self, short_code: str, user_id: str = None, **updates) -> bool:
        """Update a shortened URL."""
        with self._lock:
            if short_code not in self.urls:
                return False
            
            url = self.urls[short_code]
            
            # Check permissions
            if user_id and url.user_id and url.user_id != user_id:
                user = self.users.get(user_id)
                if not user or user.role != UserRole.ADMIN:
                    return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(url, key):
                    setattr(url, key, value)
            
            return True
    
    def delete_url(self, short_code: str, user_id: str = None) -> bool:
        """Delete a shortened URL."""
        with self._lock:
            if short_code not in self.urls:
                return False
            
            url = self.urls[short_code]
            
            # Check permissions
            if user_id and url.user_id and url.user_id != user_id:
                user = self.users.get(user_id)
                if not user or user.role != UserRole.ADMIN:
                    return False
            
            # Delete URL
            del self.urls[short_code]
            self.url_cache.pop(short_code, None)
            
            return True
    
    def bulk_shorten(self, urls: List[str], user_id: str = None) -> List[Tuple[str, str, bool]]:
        """Bulk shorten multiple URLs."""
        results = []
        
        for url in urls:
            try:
                shortened = self.shorten_url(url, user_id)
                results.append((url, shortened.short_code, True))
            except Exception as e:
                results.append((url, str(e), False))
        
        return results
    
    def get_user_urls(self, user_id: str, limit: int = 100, offset: int = 0) -> List[ShortenedURL]:
        """Get URLs created by a user."""
        user_urls = [url for url in self.urls.values() if url.user_id == user_id]
        user_urls.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_urls[offset:offset + limit]
    
    def get_analytics_summary(self, user_id: str = None) -> Dict[str, Any]:
        """Get analytics summary."""
        if user_id:
            # User-specific analytics
            user_urls = [url for url in self.urls.values() if url.user_id == user_id]
            total_urls = len(user_urls)
            total_clicks = sum(url.click_count for url in user_urls)
        else:
            # Global analytics
            total_urls = len(self.urls)
            total_clicks = sum(url.click_count for url in self.urls.values())
        
        # Top performing URLs
        top_urls = self.analytics.get_top_performing_urls(user_id, limit=5)
        
        return {
            'total_urls': total_urls,
            'total_clicks': total_clicks,
            'average_clicks_per_url': total_clicks / max(1, total_urls),
            'top_urls': top_urls,
            'active_urls': len([url for url in self.urls.values() 
                              if url.status == URLStatus.ACTIVE and 
                              (not user_id or url.user_id == user_id)])
        }
    
    def add_custom_domain(self, domain: str, user_id: str) -> CustomDomain:
        """Add custom domain for user."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        if len([d for d in self.custom_domains.values() if d.user_id == user_id]) >= user.custom_domain_limit:
            raise ValueError("Custom domain limit exceeded")
        
        custom_domain = CustomDomain(domain=domain, user_id=user_id)
        
        with self._lock:
            self.custom_domains[domain] = custom_domain
        
        return custom_domain
    
    def verify_custom_domain(self, domain: str, verification_token: str) -> bool:
        """Verify custom domain ownership."""
        with self._lock:
            if domain not in self.custom_domains:
                return False
            
            custom_domain = self.custom_domains[domain]
            if custom_domain.verification_token == verification_token:
                custom_domain.is_verified = True
                return True
            
            return False


# ============================================================================
# API INTERFACE
# ============================================================================

class URLShortenerAPI:
    """REST API interface for URL shortener."""
    
    def __init__(self, service: URLShortenerService):
        self.service = service
        self.rate_limiter = {}  # Simplified rate limiting
    
    def authenticate_user(self, api_key: str) -> Optional[str]:
        """Authenticate user by API key."""
        for user_id, user in self.service.users.items():
            if user.api_key == api_key:
                return user_id
        return None
    
    def shorten_endpoint(self, request_data: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
        """POST /shorten endpoint."""
        try:
            # Authentication
            user_id = None
            if api_key:
                user_id = self.authenticate_user(api_key)
                if not user_id:
                    return {"error": "Invalid API key", "status": 401}
            
            # Extract request data
            original_url = request_data.get('url')
            if not original_url:
                return {"error": "URL is required", "status": 400}
            
            custom_code = request_data.get('custom_code')
            custom_domain = request_data.get('custom_domain')
            expires_at = request_data.get('expires_at')
            
            if expires_at:
                expires_at = datetime.fromisoformat(expires_at)
            
            # Shorten URL
            shortened = self.service.shorten_url(
                original_url=original_url,
                user_id=user_id,
                custom_code=custom_code,
                custom_domain=custom_domain,
                expires_at=expires_at,
                title=request_data.get('title'),
                description=request_data.get('description'),
                tags=request_data.get('tags', []),
                password=request_data.get('password'),
                max_clicks=request_data.get('max_clicks')
            )
            
            # Build short URL
            domain = custom_domain if custom_domain else self.service.base_domain
            short_url = f"https://{domain}/{shortened.short_code}"
            
            return {
                "short_url": short_url,
                "short_code": shortened.short_code,
                "original_url": shortened.original_url,
                "created_at": shortened.created_at.isoformat(),
                "expires_at": shortened.expires_at.isoformat() if shortened.expires_at else None,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": 400}
    
    def expand_endpoint(self, short_code: str, request_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """GET /{short_code} endpoint."""
        try:
            original_url, is_safe = self.service.expand_url(
                short_code=short_code,
                track_click=True,
                request_info=request_info
            )
            
            return {
                "original_url": original_url,
                "is_safe": is_safe,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": 404 if "not found" in str(e) else 400}
    
    def analytics_endpoint(self, short_code: str, api_key: str = None) -> Dict[str, Any]:
        """GET /analytics/{short_code} endpoint."""
        try:
            # Authentication
            user_id = None
            if api_key:
                user_id = self.authenticate_user(api_key)
                if not user_id:
                    return {"error": "Invalid API key", "status": 401}
            
            # Get URL info
            url_info = self.service.get_url_info(short_code, user_id)
            if not url_info:
                return {"error": "URL not found or access denied", "status": 404}
            
            # Get analytics
            analytics = self.service.analytics.get_analytics(short_code)
            
            return {
                "short_code": short_code,
                "total_clicks": analytics.total_clicks,
                "unique_clicks": analytics.unique_clicks,
                "clicks_by_day": analytics.clicks_by_day,
                "clicks_by_country": analytics.clicks_by_country,
                "clicks_by_referrer": analytics.clicks_by_referrer,
                "clicks_by_device": analytics.clicks_by_device,
                "clicks_by_browser": analytics.clicks_by_browser,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": 400}
    
    def bulk_shorten_endpoint(self, request_data: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
        """POST /bulk endpoint."""
        try:
            # Authentication
            user_id = None
            if api_key:
                user_id = self.authenticate_user(api_key)
                if not user_id:
                    return {"error": "Invalid API key", "status": 401}
            
            # Extract URLs
            urls = request_data.get('urls', [])
            if not urls or len(urls) > 100:  # Limit bulk operations
                return {"error": "Invalid URLs list (max 100)", "status": 400}
            
            # Bulk shorten
            results = self.service.bulk_shorten(urls, user_id)
            
            return {
                "results": [
                    {
                        "original_url": url,
                        "short_code": code,
                        "success": success,
                        "short_url": f"https://{self.service.base_domain}/{code}" if success else None
                    }
                    for url, code, success in results
                ],
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": 400}


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_url_shortener():
    """Demonstrate the URL shortener system."""
    print("=== URL SHORTENER SYSTEM DEMONSTRATION ===\n")
    
    # Initialize service
    print("1. SERVICE INITIALIZATION:")
    
    service = URLShortenerService("demo.ly")
    api = URLShortenerAPI(service)
    
    print("   ✓ URL shortener service initialized")
    print("   ✓ API interface created")
    print()
    
    # Create users
    print("2. USER CREATION:")
    
    free_user = service.create_user("free@example.com", UserRole.FREE)
    premium_user = service.create_user("premium@example.com", UserRole.PREMIUM)
    enterprise_user = service.create_user("enterprise@example.com", UserRole.ENTERPRISE)
    
    print(f"   ✓ Free user created: {free_user.email} (limit: {free_user.monthly_url_limit} URLs)")
    print(f"   ✓ Premium user created: {premium_user.email} (limit: {premium_user.monthly_url_limit} URLs)")
    print(f"   ✓ Enterprise user created: {enterprise_user.email} (limit: {enterprise_user.monthly_url_limit} URLs)")
    print()
    
    # Test URL shortening
    print("3. URL SHORTENING TEST:")
    
    test_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.python.org",
        "https://www.example.com"
    ]
    
    shortened_urls = []
    for i, url in enumerate(test_urls):
        try:
            shortened = service.shorten_url(
                original_url=url,
                user_id=premium_user.user_id,
                title=f"Test URL {i+1}",
                tags=["demo", "test"]
            )
            shortened_urls.append(shortened)
            
            print(f"   ✓ {url} -> {shortened.short_code}")
        except Exception as e:
            print(f"   ✗ Failed to shorten {url}: {e}")
    
    print()
    
    # Test custom short codes
    print("4. CUSTOM SHORT CODE TEST:")
    
    try:
        custom_url = service.shorten_url(
            original_url="https://www.custom-example.com",
            user_id=premium_user.user_id,
            custom_code="CUSTOM123",
            title="Custom Short Code Demo"
        )
        print(f"   ✓ Custom code created: {custom_url.short_code}")
    except Exception as e:
        print(f"   ✗ Custom code failed: {e}")
    
    # Try duplicate custom code
    try:
        service.shorten_url(
            original_url="https://www.another-site.com",
            user_id=premium_user.user_id,
            custom_code="CUSTOM123"
        )
        print("   ✗ Duplicate custom code should have failed")
    except Exception as e:
        print(f"   ✓ Duplicate custom code correctly rejected: {e}")
    
    print()
    
    # Test URL expansion and click tracking
    print("5. URL EXPANSION AND CLICK TRACKING:")
    
    if shortened_urls:
        test_url = shortened_urls[0]
        
        # Simulate clicks with different request info
        click_scenarios = [
            {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0",
                "referrer": "https://www.google.com",
                "country": "US",
                "city": "New York"
            },
            {
                "ip_address": "10.0.0.50",
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) Safari/604.1",
                "referrer": "https://www.facebook.com",
                "country": "CA",
                "city": "Toronto"
            },
            {
                "ip_address": "172.16.0.25",
                "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/89.0",
                "country": "UK",
                "city": "London"
            }
        ]
        
        for i, request_info in enumerate(click_scenarios):
            try:
                original_url, is_safe = service.expand_url(
                    test_url.short_code,
                    track_click=True,
                    request_info=request_info
                )
                print(f"   Click {i+1}: {test_url.short_code} -> {original_url} (safe: {is_safe})")
            except Exception as e:
                print(f"   ✗ Click {i+1} failed: {e}")
        
        print(f"   Total clicks tracked: {test_url.click_count}")
    
    print()
    
    # Test analytics
    print("6. ANALYTICS TEST:")
    
    if shortened_urls:
        test_url = shortened_urls[0]
        analytics = service.analytics.get_analytics(test_url.short_code)
        
        print(f"   Analytics for {test_url.short_code}:")
        print(f"     Total clicks: {analytics.total_clicks}")
        print(f"     Clicks by country: {dict(analytics.clicks_by_country)}")
        print(f"     Clicks by device: {dict(analytics.clicks_by_device)}")
        print(f"     Clicks by browser: {dict(analytics.clicks_by_browser)}")
        
        # Test click trends
        trends = service.analytics.get_click_trends(test_url.short_code, days=7)
        print(f"     Recent trends: {len(trends)} days with data")
    
    print()
    
    # Test API endpoints
    print("7. API ENDPOINTS TEST:")
    
    # Test shorten endpoint
    shorten_request = {
        "url": "https://www.api-test.com",
        "custom_code": "API123",
        "title": "API Test URL"
    }
    
    shorten_response = api.shorten_endpoint(shorten_request, premium_user.api_key)
    print(f"   Shorten API: {shorten_response.get('status', 'unknown')}")
    
    if shorten_response.get('status') == 'success':
        print(f"     Short URL: {shorten_response['short_url']}")
        
        # Test expand endpoint
        expand_response = api.expand_endpoint(
            shorten_response['short_code'],
            {"ip_address": "127.0.0.1", "user_agent": "API Test"}
        )
        print(f"   Expand API: {expand_response.get('status', 'unknown')}")
        
        # Test analytics endpoint
        analytics_response = api.analytics_endpoint(
            shorten_response['short_code'],
            premium_user.api_key
        )
        print(f"   Analytics API: {analytics_response.get('status', 'unknown')}")
    
    print()
    
    # Test bulk operations
    print("8. BULK OPERATIONS TEST:")
    
    bulk_urls = [
        "https://www.bulk1.com",
        "https://www.bulk2.com",
        "https://www.bulk3.com",
        "invalid-url",  # This should fail
        "https://www.bulk4.com"
    ]
    
    bulk_request = {"urls": bulk_urls}
    bulk_response = api.bulk_shorten_endpoint(bulk_request, premium_user.api_key)
    
    if bulk_response.get('status') == 'success':
        print("   Bulk shorten results:")
        for result in bulk_response['results']:
            status = "✓" if result['success'] else "✗"
            print(f"     {status} {result['original_url']}")
    
    print()
    
    # Test URL management
    print("9. URL MANAGEMENT TEST:")
    
    if shortened_urls:
        test_url = shortened_urls[0]
        
        # Update URL
        success = service.update_url(
            test_url.short_code,
            premium_user.user_id,
            title="Updated Title",
            description="Updated Description"
        )
        print(f"   Update URL: {'✓ Success' if success else '✗ Failed'}")
        
        # Get user URLs
        user_urls = service.get_user_urls(premium_user.user_id, limit=5)
        print(f"   User URLs: {len(user_urls)} found")
        
        for url in user_urls[:3]:  # Show first 3
            print(f"     {url.short_code}: {url.title or 'No title'} ({url.click_count} clicks)")
    
    print()
    
    # Test custom domain
    print("10. CUSTOM DOMAIN TEST:")
    
    try:
        custom_domain = service.add_custom_domain("my-brand.com", premium_user.user_id)
        print(f"   ✓ Custom domain added: {custom_domain.domain}")
        print(f"   Verification token: {custom_domain.verification_token[:16]}...")
        
        # Verify domain
        verified = service.verify_custom_domain(
            custom_domain.domain,
            custom_domain.verification_token
        )
        print(f"   Domain verification: {'✓ Success' if verified else '✗ Failed'}")
        
    except Exception as e:
        print(f"   ✗ Custom domain failed: {e}")
    
    print()
    
    # Test security features
    print("11. SECURITY FEATURES TEST:")
    
    # Test URL validation
    suspicious_urls = [
        "http://malware.com/virus.exe",
        "https://phishing.com/fake-bank",
        "https://192.168.1.1/admin",
        "https://bit.ly/suspicious123"
    ]
    
    for url in suspicious_urls:
        try:
            service.shorten_url(url, premium_user.user_id)
            print(f"   ✗ Suspicious URL allowed: {url}")
        except Exception as e:
            print(f"   ✓ Suspicious URL blocked: {url}")
    
    print()
    
    # Test expiration
    print("12. URL EXPIRATION TEST:")
    
    # Create URL with short expiration
    try:
        expires_soon = service.shorten_url(
            "https://www.expires-soon.com",
            premium_user.user_id,
            expires_at=datetime.now() + timedelta(seconds=1)
        )
        print(f"   ✓ Created expiring URL: {expires_soon.short_code}")
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Try to access expired URL
        try:
            service.expand_url(expires_soon.short_code)
            print("   ✗ Expired URL should not be accessible")
        except Exception as e:
            print(f"   ✓ Expired URL correctly blocked: {e}")
            
    except Exception as e:
        print(f"   ✗ Expiration test failed: {e}")
    
    print()
    
    # Show comprehensive statistics
    print("13. COMPREHENSIVE STATISTICS:")
    
    # Global stats
    global_stats = service.get_analytics_summary()
    print(f"   Global Statistics:")
    print(f"     Total URLs: {global_stats['total_urls']}")
    print(f"     Total clicks: {global_stats['total_clicks']}")
    print(f"     Average clicks per URL: {global_stats['average_clicks_per_url']:.2f}")
    print(f"     Active URLs: {global_stats['active_urls']}")
    
    # User stats
    user_stats = service.get_analytics_summary(premium_user.user_id)
    print(f"\n   Premium User Statistics:")
    print(f"     URLs created: {user_stats['total_urls']}")
    print(f"     Total clicks: {user_stats['total_clicks']}")
    
    # Top performing URLs
    top_urls = service.analytics.get_top_performing_urls(limit=3)
    print(f"\n   Top Performing URLs:")
    for short_code, clicks in top_urls:
        url_info = service.get_url_info(short_code)
        title = url_info.title if url_info else "No title"
        print(f"     {short_code}: {title} ({clicks} clicks)")
    
    print()
    
    # Show final system state
    print("14. FINAL SYSTEM STATE:")
    
    print(f"   Total users: {len(service.users)}")
    print(f"   Total URLs: {len(service.urls)}")
    print(f"   Total custom domains: {len(service.custom_domains)}")
    print(f"   Total click events: {len(service.analytics.click_events)}")
    
    # Show URL statuses
    status_counts = {}
    for url in service.urls.values():
        status_counts[url.status.value] = status_counts.get(url.status.value, 0) + 1
    
    print(f"   URL status breakdown:")
    for status, count in status_counts.items():
        print(f"     {status}: {count}")
    
    print()
    print("=== URL SHORTENER DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_url_shortener()
