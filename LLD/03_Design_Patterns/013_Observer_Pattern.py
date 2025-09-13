"""
OBSERVER PATTERN - Behavioral Design Pattern
============================================

Problem Statement:
Implement the Observer pattern to define a one-to-many dependency between objects
so that when one object changes state, all dependents are notified automatically:
- Subject-Observer relationship with loose coupling
- Event notification system with multiple subscribers
- Model-View architecture implementation
- Real-time data updates and synchronization
- Publisher-Subscriber messaging patterns

Learning Objectives:
- Understand Observer vs Publisher-Subscriber patterns
- Implement loose coupling between subjects and observers
- Design event-driven architectures
- Handle observer registration and notification
- Create reactive programming patterns
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Set
import threading
import time
from datetime import datetime
from enum import Enum
import weakref
import json


# ============================================================================
# OBSERVER INTERFACE
# ============================================================================

class Observer(ABC):
    """Abstract observer interface."""
    
    @abstractmethod
    def update(self, subject: 'Subject', event_data: Dict[str, Any]) -> None:
        """Called when subject state changes."""
        pass
    
    @abstractmethod
    def get_observer_id(self) -> str:
        """Get unique observer identifier."""
        pass


class Subject(ABC):
    """Abstract subject interface."""
    
    @abstractmethod
    def attach(self, observer: Observer) -> None:
        """Attach an observer."""
        pass
    
    @abstractmethod
    def detach(self, observer: Observer) -> None:
        """Detach an observer."""
        pass
    
    @abstractmethod
    def notify(self, event_data: Dict[str, Any]) -> None:
        """Notify all observers."""
        pass


# ============================================================================
# STOCK MARKET SYSTEM
# ============================================================================

class StockPrice(Subject):
    """Stock price subject that notifies observers of price changes."""
    
    def __init__(self, symbol: str, initial_price: float):
        self.symbol = symbol
        self._price = initial_price
        self._observers: Set[Observer] = set()
        self._price_history: List[Dict[str, Any]] = []
        self._notification_count = 0
        
        # Record initial price
        self._price_history.append({
            'price': initial_price,
            'timestamp': datetime.now().isoformat(),
            'change': 0.0,
            'change_percent': 0.0
        })
        
        print(f"StockPrice created for {symbol} at ${initial_price:.2f}")
    
    def attach(self, observer: Observer) -> None:
        """Attach observer to stock price updates."""
        self._observers.add(observer)
        print(f"Observer {observer.get_observer_id()} attached to {self.symbol}")
    
    def detach(self, observer: Observer) -> None:
        """Detach observer from stock price updates."""
        self._observers.discard(observer)
        print(f"Observer {observer.get_observer_id()} detached from {self.symbol}")
    
    def notify(self, event_data: Dict[str, Any]) -> None:
        """Notify all observers of price change."""
        self._notification_count += 1
        print(f"Notifying {len(self._observers)} observers of {self.symbol} price change")
        
        for observer in self._observers.copy():  # Copy to avoid modification during iteration
            try:
                observer.update(self, event_data)
            except Exception as e:
                print(f"Error notifying observer {observer.get_observer_id()}: {e}")
    
    @property
    def price(self) -> float:
        """Get current stock price."""
        return self._price
    
    @price.setter
    def price(self, new_price: float) -> None:
        """Set new stock price and notify observers."""
        if new_price <= 0:
            raise ValueError("Stock price must be positive")
        
        old_price = self._price
        self._price = new_price
        
        # Calculate change
        change = new_price - old_price
        change_percent = (change / old_price) * 100 if old_price > 0 else 0
        
        # Record price change
        price_record = {
            'price': new_price,
            'timestamp': datetime.now().isoformat(),
            'change': change,
            'change_percent': change_percent,
            'previous_price': old_price
        }
        self._price_history.append(price_record)
        
        # Notify observers
        event_data = {
            'symbol': self.symbol,
            'new_price': new_price,
            'old_price': old_price,
            'change': change,
            'change_percent': change_percent,
            'timestamp': price_record['timestamp']
        }
        
        self.notify(event_data)
    
    def get_price_history(self) -> List[Dict[str, Any]]:
        """Get price history."""
        return self._price_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get stock statistics."""
        if not self._price_history:
            return {}
        
        prices = [record['price'] for record in self._price_history]
        
        return {
            'symbol': self.symbol,
            'current_price': self._price,
            'min_price': min(prices),
            'max_price': max(prices),
            'price_changes': len(self._price_history) - 1,
            'observers_count': len(self._observers),
            'notifications_sent': self._notification_count
        }


class StockDisplay(Observer):
    """Display observer that shows current stock prices."""
    
    def __init__(self, display_name: str):
        self.display_name = display_name
        self.displayed_stocks: Dict[str, Dict[str, Any]] = {}
        self.update_count = 0
        
    def update(self, subject: StockPrice, event_data: Dict[str, Any]) -> None:
        """Update display with new stock price."""
        self.update_count += 1
        symbol = event_data['symbol']
        
        self.displayed_stocks[symbol] = {
            'price': event_data['new_price'],
            'change': event_data['change'],
            'change_percent': event_data['change_percent'],
            'last_updated': event_data['timestamp']
        }
        
        # Display the update
        change_indicator = "↑" if event_data['change'] > 0 else "↓" if event_data['change'] < 0 else "→"
        print(f"[{self.display_name}] {symbol}: ${event_data['new_price']:.2f} "
              f"{change_indicator} {event_data['change']:+.2f} ({event_data['change_percent']:+.1f}%)")
    
    def get_observer_id(self) -> str:
        return f"StockDisplay_{self.display_name}"
    
    def show_portfolio(self) -> None:
        """Show all displayed stocks."""
        print(f"\n=== {self.display_name} Portfolio ===")
        for symbol, data in self.displayed_stocks.items():
            print(f"{symbol}: ${data['price']:.2f} ({data['change_percent']:+.1f}%)")
        print(f"Total updates received: {self.update_count}")


class TradingBot(Observer):
    """Trading bot that makes decisions based on price changes."""
    
    def __init__(self, bot_name: str, buy_threshold: float = -5.0, sell_threshold: float = 10.0):
        self.bot_name = bot_name
        self.buy_threshold = buy_threshold  # Buy when price drops by this percentage
        self.sell_threshold = sell_threshold  # Sell when price rises by this percentage
        self.portfolio: Dict[str, int] = {}  # symbol -> quantity
        self.cash = 10000.0  # Starting cash
        self.transactions: List[Dict[str, Any]] = []
        
    def update(self, subject: StockPrice, event_data: Dict[str, Any]) -> None:
        """Make trading decisions based on price changes."""
        symbol = event_data['symbol']
        price = event_data['new_price']
        change_percent = event_data['change_percent']
        
        # Buy decision
        if change_percent <= self.buy_threshold and self.cash >= price:
            quantity = int(self.cash // price)  # Buy as many as possible
            if quantity > 0:
                cost = quantity * price
                self.cash -= cost
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                
                transaction = {
                    'action': 'BUY',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'total': cost,
                    'timestamp': event_data['timestamp']
                }
                self.transactions.append(transaction)
                
                print(f"[{self.bot_name}] BUY {quantity} shares of {symbol} at ${price:.2f} "
                      f"(Total: ${cost:.2f})")
        
        # Sell decision
        elif change_percent >= self.sell_threshold and symbol in self.portfolio:
            quantity = self.portfolio[symbol]
            if quantity > 0:
                revenue = quantity * price
                self.cash += revenue
                del self.portfolio[symbol]
                
                transaction = {
                    'action': 'SELL',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'total': revenue,
                    'timestamp': event_data['timestamp']
                }
                self.transactions.append(transaction)
                
                print(f"[{self.bot_name}] SELL {quantity} shares of {symbol} at ${price:.2f} "
                      f"(Total: ${revenue:.2f})")
    
    def get_observer_id(self) -> str:
        return f"TradingBot_{self.bot_name}"
    
    def get_portfolio_value(self, stock_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        portfolio_value = sum(
            quantity * stock_prices.get(symbol, 0)
            for symbol, quantity in self.portfolio.items()
        )
        return self.cash + portfolio_value
    
    def show_status(self) -> None:
        """Show bot status."""
        print(f"\n=== {self.bot_name} Status ===")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Holdings: {self.portfolio}")
        print(f"Transactions: {len(self.transactions)}")


# ============================================================================
# NEWS SYSTEM
# ============================================================================

class NewsEventType(Enum):
    BREAKING_NEWS = "breaking_news"
    MARKET_UPDATE = "market_update"
    COMPANY_ANNOUNCEMENT = "company_announcement"
    ECONOMIC_INDICATOR = "economic_indicator"


class NewsAgency(Subject):
    """News agency that publishes news to subscribers."""
    
    def __init__(self, agency_name: str):
        self.agency_name = agency_name
        self._observers: Dict[NewsEventType, Set[Observer]] = {
            event_type: set() for event_type in NewsEventType
        }
        self._all_observers: Set[Observer] = set()
        self.published_news: List[Dict[str, Any]] = []
        
    def attach(self, observer: Observer, event_types: List[NewsEventType] = None) -> None:
        """Attach observer to specific news types or all news."""
        if event_types is None:
            # Subscribe to all news types
            self._all_observers.add(observer)
            print(f"Observer {observer.get_observer_id()} subscribed to all news from {self.agency_name}")
        else:
            # Subscribe to specific news types
            for event_type in event_types:
                self._observers[event_type].add(observer)
            print(f"Observer {observer.get_observer_id()} subscribed to {[t.value for t in event_types]} from {self.agency_name}")
    
    def detach(self, observer: Observer, event_types: List[NewsEventType] = None) -> None:
        """Detach observer from specific news types or all news."""
        if event_types is None:
            # Unsubscribe from all news types
            self._all_observers.discard(observer)
            for observers_set in self._observers.values():
                observers_set.discard(observer)
            print(f"Observer {observer.get_observer_id()} unsubscribed from all news")
        else:
            # Unsubscribe from specific news types
            for event_type in event_types:
                self._observers[event_type].discard(observer)
            print(f"Observer {observer.get_observer_id()} unsubscribed from {[t.value for t in event_types]}")
    
    def notify(self, event_data: Dict[str, Any]) -> None:
        """Notify observers based on news type."""
        event_type = event_data.get('event_type')
        
        # Notify all subscribers
        all_notified = set()
        
        # Notify subscribers to all news
        for observer in self._all_observers.copy():
            try:
                observer.update(self, event_data)
                all_notified.add(observer)
            except Exception as e:
                print(f"Error notifying observer {observer.get_observer_id()}: {e}")
        
        # Notify subscribers to specific news type
        if event_type in self._observers:
            for observer in self._observers[event_type].copy():
                if observer not in all_notified:  # Avoid duplicate notifications
                    try:
                        observer.update(self, event_data)
                        all_notified.add(observer)
                    except Exception as e:
                        print(f"Error notifying observer {observer.get_observer_id()}: {e}")
        
        print(f"News '{event_data.get('headline', 'Unknown')}' sent to {len(all_notified)} subscribers")
    
    def publish_news(self, headline: str, content: str, event_type: NewsEventType,
                    tags: List[str] = None) -> None:
        """Publish news article."""
        news_article = {
            'headline': headline,
            'content': content,
            'event_type': event_type,
            'tags': tags or [],
            'agency': self.agency_name,
            'published_at': datetime.now().isoformat(),
            'article_id': len(self.published_news) + 1
        }
        
        self.published_news.append(news_article)
        
        # Notify observers
        self.notify(news_article)
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        type_subscriptions = {
            event_type.value: len(observers)
            for event_type, observers in self._observers.items()
        }
        
        return {
            'agency_name': self.agency_name,
            'all_news_subscribers': len(self._all_observers),
            'type_specific_subscribers': type_subscriptions,
            'total_articles_published': len(self.published_news)
        }


class NewsReader(Observer):
    """News reader that receives and processes news."""
    
    def __init__(self, reader_name: str, interests: List[str] = None):
        self.reader_name = reader_name
        self.interests = interests or []  # Keywords of interest
        self.received_news: List[Dict[str, Any]] = []
        self.filtered_news: List[Dict[str, Any]] = []
        
    def update(self, subject: NewsAgency, event_data: Dict[str, Any]) -> None:
        """Receive and process news update."""
        self.received_news.append(event_data)
        
        # Filter news based on interests
        if self._is_interested(event_data):
            self.filtered_news.append(event_data)
            print(f"[{self.reader_name}] 📰 {event_data['headline']}")
        else:
            print(f"[{self.reader_name}] 📄 Received: {event_data['headline']} (filtered out)")
    
    def _is_interested(self, news_article: Dict[str, Any]) -> bool:
        """Check if news article matches reader's interests."""
        if not self.interests:
            return True  # Interested in all news
        
        # Check headline and content for interest keywords
        text_to_check = (news_article.get('headline', '') + ' ' + 
                        news_article.get('content', '')).lower()
        
        return any(interest.lower() in text_to_check for interest in self.interests)
    
    def get_observer_id(self) -> str:
        return f"NewsReader_{self.reader_name}"
    
    def show_news_summary(self) -> None:
        """Show summary of received news."""
        print(f"\n=== {self.reader_name} News Summary ===")
        print(f"Total news received: {len(self.received_news)}")
        print(f"Relevant news: {len(self.filtered_news)}")
        
        if self.filtered_news:
            print("Recent relevant headlines:")
            for article in self.filtered_news[-3:]:  # Show last 3
                print(f"  • {article['headline']}")


# ============================================================================
# MODEL-VIEW ARCHITECTURE
# ============================================================================

class UserModel(Subject):
    """User model that notifies views of data changes."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._data = {
            'name': '',
            'email': '',
            'age': 0,
            'preferences': {},
            'last_login': None
        }
        self._observers: Set[Observer] = set()
        self._change_history: List[Dict[str, Any]] = []
        
    def attach(self, observer: Observer) -> None:
        """Attach view observer."""
        self._observers.add(observer)
        print(f"View {observer.get_observer_id()} attached to user {self.user_id}")
    
    def detach(self, observer: Observer) -> None:
        """Detach view observer."""
        self._observers.discard(observer)
        print(f"View {observer.get_observer_id()} detached from user {self.user_id}")
    
    def notify(self, event_data: Dict[str, Any]) -> None:
        """Notify all views of data changes."""
        for observer in self._observers.copy():
            try:
                observer.update(self, event_data)
            except Exception as e:
                print(f"Error notifying view {observer.get_observer_id()}: {e}")
    
    def update_field(self, field: str, value: Any) -> None:
        """Update a specific field and notify observers."""
        if field not in self._data:
            raise ValueError(f"Unknown field: {field}")
        
        old_value = self._data[field]
        self._data[field] = value
        
        # Record change
        change_record = {
            'field': field,
            'old_value': old_value,
            'new_value': value,
            'timestamp': datetime.now().isoformat()
        }
        self._change_history.append(change_record)
        
        # Notify observers
        event_data = {
            'user_id': self.user_id,
            'field_changed': field,
            'old_value': old_value,
            'new_value': value,
            'timestamp': change_record['timestamp']
        }
        
        self.notify(event_data)
    
    def get_data(self) -> Dict[str, Any]:
        """Get user data."""
        return self._data.copy()
    
    def get_field(self, field: str) -> Any:
        """Get specific field value."""
        return self._data.get(field)


class UserProfileView(Observer):
    """View that displays user profile information."""
    
    def __init__(self, view_name: str):
        self.view_name = view_name
        self.displayed_data: Dict[str, Any] = {}
        self.update_count = 0
        
    def update(self, subject: UserModel, event_data: Dict[str, Any]) -> None:
        """Update view when model changes."""
        self.update_count += 1
        field = event_data['field_changed']
        new_value = event_data['new_value']
        
        self.displayed_data[field] = new_value
        
        print(f"[{self.view_name}] Updated {field}: {new_value}")
        
        # Trigger view refresh
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        """Refresh the display (simulated)."""
        print(f"[{self.view_name}] Display refreshed")
    
    def get_observer_id(self) -> str:
        return f"UserProfileView_{self.view_name}"
    
    def show_profile(self) -> None:
        """Show current profile display."""
        print(f"\n=== {self.view_name} ===")
        for field, value in self.displayed_data.items():
            print(f"{field.title()}: {value}")
        print(f"Updates received: {self.update_count}")


# ============================================================================
# EVENT-DRIVEN SYSTEM WITH WEAK REFERENCES
# ============================================================================

class WeakObserverSubject(Subject):
    """Subject that uses weak references to prevent memory leaks."""
    
    def __init__(self, name: str):
        self.name = name
        self._observers: Set[weakref.ReferenceType] = set()
        self._notification_count = 0
        
    def attach(self, observer: Observer) -> None:
        """Attach observer using weak reference."""
        weak_ref = weakref.ref(observer, self._cleanup_observer)
        self._observers.add(weak_ref)
        print(f"Observer {observer.get_observer_id()} attached with weak reference")
    
    def detach(self, observer: Observer) -> None:
        """Detach observer."""
        # Find and remove the weak reference
        to_remove = None
        for weak_ref in self._observers:
            if weak_ref() is observer:
                to_remove = weak_ref
                break
        
        if to_remove:
            self._observers.remove(to_remove)
            print(f"Observer {observer.get_observer_id()} detached")
    
    def _cleanup_observer(self, weak_ref: weakref.ReferenceType) -> None:
        """Callback to clean up dead weak references."""
        self._observers.discard(weak_ref)
        print("Cleaned up dead observer reference")
    
    def notify(self, event_data: Dict[str, Any]) -> None:
        """Notify all live observers."""
        self._notification_count += 1
        live_observers = []
        
        # Collect live observers and clean up dead ones
        dead_refs = set()
        for weak_ref in self._observers:
            observer = weak_ref()
            if observer is not None:
                live_observers.append(observer)
            else:
                dead_refs.add(weak_ref)
        
        # Remove dead references
        self._observers -= dead_refs
        
        # Notify live observers
        for observer in live_observers:
            try:
                observer.update(self, event_data)
            except Exception as e:
                print(f"Error notifying observer: {e}")
        
        print(f"Notified {len(live_observers)} live observers")
    
    def trigger_event(self, event_name: str, data: Dict[str, Any] = None) -> None:
        """Trigger an event."""
        event_data = {
            'event_name': event_name,
            'source': self.name,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.notify(event_data)
    
    def get_observer_count(self) -> int:
        """Get count of live observers."""
        live_count = sum(1 for ref in self._observers if ref() is not None)
        return live_count


class EventLogger(Observer):
    """Observer that logs all events."""
    
    def __init__(self, logger_name: str):
        self.logger_name = logger_name
        self.logged_events: List[Dict[str, Any]] = []
        
    def update(self, subject: Subject, event_data: Dict[str, Any]) -> None:
        """Log the event."""
        log_entry = {
            'logger': self.logger_name,
            'subject': getattr(subject, 'name', str(type(subject).__name__)),
            'event_data': event_data,
            'logged_at': datetime.now().isoformat()
        }
        
        self.logged_events.append(log_entry)
        print(f"[{self.logger_name}] Logged event: {event_data.get('event_name', 'unknown')}")
    
    def get_observer_id(self) -> str:
        return f"EventLogger_{self.logger_name}"
    
    def show_log_summary(self) -> None:
        """Show log summary."""
        print(f"\n=== {self.logger_name} Log Summary ===")
        print(f"Total events logged: {len(self.logged_events)}")
        
        if self.logged_events:
            print("Recent events:")
            for entry in self.logged_events[-3:]:
                event_name = entry['event_data'].get('event_name', 'unknown')
                print(f"  • {event_name} from {entry['subject']}")


def demonstrate_observer_pattern():
    """
    Demonstrate Observer pattern implementations.
    """
    print("=== OBSERVER PATTERN DEMONSTRATION ===\n")
    
    # 1. Stock Market System
    print("1. STOCK MARKET OBSERVER SYSTEM:")
    
    # Create stocks
    apple_stock = StockPrice("AAPL", 150.00)
    google_stock = StockPrice("GOOGL", 2800.00)
    
    # Create observers
    main_display = StockDisplay("Main Display")
    mobile_display = StockDisplay("Mobile App")
    trading_bot = TradingBot("AlgoBot1", buy_threshold=-3.0, sell_threshold=8.0)
    
    # Attach observers
    apple_stock.attach(main_display)
    apple_stock.attach(mobile_display)
    apple_stock.attach(trading_bot)
    
    google_stock.attach(main_display)
    google_stock.attach(trading_bot)
    
    print(f"\n   Simulating stock price changes:")
    
    # Simulate price changes
    apple_stock.price = 145.50  # -3% drop, should trigger bot buy
    time.sleep(0.1)
    apple_stock.price = 158.25  # +8.5% rise, should trigger bot sell
    time.sleep(0.1)
    google_stock.price = 2750.00  # -1.8% drop
    time.sleep(0.1)
    google_stock.price = 2912.00  # +5.9% rise
    
    # Show results
    main_display.show_portfolio()
    trading_bot.show_status()
    
    # Show stock statistics
    print(f"\n   Stock Statistics:")
    apple_stats = apple_stock.get_statistics()
    print(f"   AAPL: {apple_stats['price_changes']} changes, {apple_stats['observers_count']} observers")
    
    print()
    
    # 2. News Agency System
    print("2. NEWS AGENCY OBSERVER SYSTEM:")
    
    # Create news agency
    reuters = NewsAgency("Reuters")
    
    # Create news readers with different interests
    tech_reader = NewsReader("TechEnthusiast", ["technology", "AI", "software"])
    finance_reader = NewsReader("FinanceExpert", ["market", "economy", "trading"])
    general_reader = NewsReader("GeneralReader")  # Interested in all news
    
    # Subscribe readers to different news types
    reuters.attach(tech_reader, [NewsEventType.COMPANY_ANNOUNCEMENT])
    reuters.attach(finance_reader, [NewsEventType.MARKET_UPDATE, NewsEventType.ECONOMIC_INDICATOR])
    reuters.attach(general_reader)  # Subscribe to all news
    
    print(f"\n   Publishing news articles:")
    
    # Publish different types of news
    reuters.publish_news(
        "Apple Announces New AI Technology",
        "Apple has unveiled revolutionary AI technology that will change computing forever.",
        NewsEventType.COMPANY_ANNOUNCEMENT,
        ["technology", "AI", "Apple"]
    )
    
    reuters.publish_news(
        "Stock Market Reaches Record High",
        "The stock market closed at an all-time high today driven by strong earnings.",
        NewsEventType.MARKET_UPDATE,
        ["market", "stocks", "economy"]
    )
    
    reuters.publish_news(
        "Federal Reserve Announces Interest Rate Decision",
        "The Fed has decided to maintain current interest rates at 5.25%.",
        NewsEventType.ECONOMIC_INDICATOR,
        ["economy", "interest rates", "Fed"]
    )
    
    reuters.publish_news(
        "Major Earthquake Hits California",
        "A 6.5 magnitude earthquake struck Northern California this morning.",
        NewsEventType.BREAKING_NEWS,
        ["earthquake", "California", "emergency"]
    )
    
    # Show reader summaries
    tech_reader.show_news_summary()
    finance_reader.show_news_summary()
    general_reader.show_news_summary()
    
    # Show subscription statistics
    print(f"\n   Subscription Statistics:")
    stats = reuters.get_subscription_stats()
    print(f"   {stats['agency_name']}: {stats['all_news_subscribers']} all-news subscribers")
    print(f"   Type-specific: {stats['type_specific_subscribers']}")
    
    print()
    
    # 3. Model-View Architecture
    print("3. MODEL-VIEW ARCHITECTURE:")
    
    # Create user model
    user = UserModel("user123")
    
    # Create views
    profile_view = UserProfileView("Profile Page")
    dashboard_view = UserProfileView("Dashboard")
    mobile_view = UserProfileView("Mobile App")
    
    # Attach views to model
    user.attach(profile_view)
    user.attach(dashboard_view)
    user.attach(mobile_view)
    
    print(f"\n   Updating user model:")
    
    # Update user data - all views will be notified
    user.update_field('name', 'John Doe')
    user.update_field('email', 'john.doe@example.com')
    user.update_field('age', 30)
    user.update_field('preferences', {'theme': 'dark', 'notifications': True})
    
    # Show view states
    profile_view.show_profile()
    
    # Detach one view and update again
    print(f"\n   Detaching mobile view and updating:")
    user.detach(mobile_view)
    user.update_field('age', 31)
    
    print(f"   Profile view updates: {profile_view.update_count}")
    print(f"   Dashboard view updates: {dashboard_view.update_count}")
    print(f"   Mobile view updates: {mobile_view.update_count}")
    
    print()
    
    # 4. Weak Reference Observer System
    print("4. WEAK REFERENCE OBSERVER SYSTEM:")
    
    # Create subject with weak references
    event_system = WeakObserverSubject("EventSystem")
    
    # Create observers
    logger1 = EventLogger("SystemLogger")
    logger2 = EventLogger("AuditLogger")
    
    # Attach observers
    event_system.attach(logger1)
    event_system.attach(logger2)
    
    print(f"   Initial observer count: {event_system.get_observer_count()}")
    
    # Trigger some events
    event_system.trigger_event("user_login", {"user_id": "123", "ip": "192.168.1.1"})
    event_system.trigger_event("file_uploaded", {"filename": "document.pdf", "size": 1024})
    
    # Delete one observer to test weak references
    print(f"\n   Deleting one observer:")
    del logger2  # This should be cleaned up automatically
    
    # Trigger another event
    event_system.trigger_event("system_shutdown", {"reason": "maintenance"})
    
    print(f"   Observer count after cleanup: {event_system.get_observer_count()}")
    
    # Show logger summary
    logger1.show_log_summary()
    
    print()
    
    # 5. Thread-Safe Observer System
    print("5. THREAD-SAFE OBSERVER SYSTEM:")
    
    class ThreadSafeStock(StockPrice):
        """Thread-safe version of stock price."""
        
        def __init__(self, symbol: str, initial_price: float):
            super().__init__(symbol, initial_price)
            self._lock = threading.Lock()
        
        def attach(self, observer: Observer) -> None:
            with self._lock:
                super().attach(observer)
        
        def detach(self, observer: Observer) -> None:
            with self._lock:
                super().detach(observer)
        
        def notify(self, event_data: Dict[str, Any]) -> None:
            with self._lock:
                super().notify(event_data)
    
    # Create thread-safe stock
    thread_safe_stock = ThreadSafeStock("TSLA", 800.00)
    thread_display = StockDisplay("Thread Display")
    thread_safe_stock.attach(thread_display)
    
    # Simulate concurrent price updates
    def update_price(stock, prices):
        for price in prices:
            stock.price = price
            time.sleep(0.01)
    
    import threading
    
    # Create threads for concurrent updates
    thread1 = threading.Thread(target=update_price, args=(thread_safe_stock, [810, 820, 815]))
    thread2 = threading.Thread(target=update_price, args=(thread_safe_stock, [825, 830, 828]))
    
    print(f"   Starting concurrent price updates:")
    thread1.start()
    thread2.start()
    
    # Wait for threads to complete
    thread1.join()
    thread2.join()
    
    print(f"   Final TSLA price: ${thread_safe_stock.price:.2f}")
    print(f"   Thread display updates: {thread_display.update_count}")
    
    print()
    
    # 6. Observer Pattern Benefits
    print("6. OBSERVER PATTERN BENEFITS:")
    print("   ✓ Loose Coupling: Subjects and observers are loosely coupled")
    print("   ✓ Dynamic Relationships: Observers can be added/removed at runtime")
    print("   ✓ Broadcast Communication: One-to-many notification")
    print("   ✓ Event-Driven Architecture: Supports reactive programming")
    print("   ✓ Separation of Concerns: Business logic separated from presentation")
    print("   ✓ Extensibility: New observers can be added without changing subjects")
    print("   ✓ Consistency: All observers get consistent updates")
    print("   ✓ Memory Management: Weak references prevent memory leaks")
    print()
    
    print("=== OBSERVER PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_observer_pattern()
