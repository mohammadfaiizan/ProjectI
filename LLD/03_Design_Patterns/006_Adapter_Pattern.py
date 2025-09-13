"""
ADAPTER PATTERN - Structural Design Pattern
===========================================

Problem Statement:
Implement the Adapter pattern to allow incompatible interfaces to work together:
- Object adapter using composition
- Class adapter using inheritance
- Two-way adapters for bidirectional compatibility
- Pluggable adapters for different implementations
- Adapter chains for complex transformations

Learning Objectives:
- Understand when to use Adapter pattern
- Implement object and class adapters
- Handle interface incompatibilities
- Design flexible adapter hierarchies
- Integrate legacy systems with new interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Protocol
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum


# ============================================================================
# TARGET INTERFACES (What the client expects)
# ============================================================================

class MediaPlayer(ABC):
    """Target interface for media players."""
    
    @abstractmethod
    def play(self, filename: str) -> bool:
        """Play media file."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop playback."""
        pass
    
    @abstractmethod
    def get_duration(self) -> float:
        """Get media duration in seconds."""
        pass
    
    @abstractmethod
    def get_current_position(self) -> float:
        """Get current playback position."""
        pass


class DataProcessor(ABC):
    """Target interface for data processing."""
    
    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and return result."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported data formats."""
        pass


class PaymentProcessor(ABC):
    """Target interface for payment processing."""
    
    @abstractmethod
    def process_payment(self, amount: float, currency: str, 
                       payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment and return result."""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: float) -> Dict[str, Any]:
        """Process refund."""
        pass
    
    @abstractmethod
    def get_transaction_status(self, transaction_id: str) -> str:
        """Get transaction status."""
        pass


# ============================================================================
# ADAPTEES (Existing classes with incompatible interfaces)
# ============================================================================

class LegacyAudioPlayer:
    """Legacy audio player with incompatible interface."""
    
    def __init__(self):
        self.current_file = ""
        self.is_playing = False
        self.position = 0.0
        self.total_duration = 0.0
    
    def load_audio_file(self, filepath: str) -> bool:
        """Load audio file (legacy method name)."""
        print(f"LegacyAudioPlayer: Loading audio file {filepath}")
        self.current_file = filepath
        self.total_duration = 180.0  # Simulated duration
        return True
    
    def start_playback(self) -> None:
        """Start audio playback (legacy method name)."""
        if self.current_file:
            print(f"LegacyAudioPlayer: Starting playback of {self.current_file}")
            self.is_playing = True
        else:
            print("LegacyAudioPlayer: No file loaded")
    
    def halt_playback(self) -> None:
        """Stop audio playback (legacy method name)."""
        print("LegacyAudioPlayer: Stopping playback")
        self.is_playing = False
        self.position = 0.0
    
    def get_track_length(self) -> float:
        """Get track length (legacy method name)."""
        return self.total_duration
    
    def get_playback_position(self) -> float:
        """Get current position (legacy method name)."""
        return self.position
    
    def set_playback_position(self, position: float) -> None:
        """Set playback position."""
        self.position = min(position, self.total_duration)


class ThirdPartyVideoPlayer:
    """Third-party video player with different interface."""
    
    def __init__(self):
        self.video_file = None
        self.state = "stopped"
        self.duration_ms = 0
        self.position_ms = 0
    
    def initialize_video(self, video_path: str) -> int:
        """Initialize video (returns status code)."""
        print(f"ThirdPartyVideoPlayer: Initializing video {video_path}")
        self.video_file = video_path
        self.duration_ms = 240000  # 4 minutes in milliseconds
        return 0  # Success code
    
    def begin_playback(self) -> int:
        """Begin video playback."""
        if self.video_file:
            print(f"ThirdPartyVideoPlayer: Beginning playback of {self.video_file}")
            self.state = "playing"
            return 0
        return -1  # Error code
    
    def terminate_playback(self) -> int:
        """Terminate video playback."""
        print("ThirdPartyVideoPlayer: Terminating playback")
        self.state = "stopped"
        self.position_ms = 0
        return 0
    
    def get_duration_milliseconds(self) -> int:
        """Get duration in milliseconds."""
        return self.duration_ms
    
    def get_position_milliseconds(self) -> int:
        """Get position in milliseconds."""
        return self.position_ms


class XMLDataService:
    """Service that works with XML data."""
    
    def __init__(self):
        self.xml_cache = {}
    
    def parse_xml_string(self, xml_string: str) -> ET.Element:
        """Parse XML string to element tree."""
        try:
            root = ET.fromstring(xml_string)
            print("XMLDataService: Successfully parsed XML")
            return root
        except ET.ParseError as e:
            print(f"XMLDataService: XML parsing error: {e}")
            return None
    
    def xml_to_dict(self, xml_element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if xml_element.attrib:
            result['@attributes'] = xml_element.attrib
        
        # Add text content
        if xml_element.text and xml_element.text.strip():
            result['text'] = xml_element.text.strip()
        
        # Add child elements
        for child in xml_element:
            child_dict = self.xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
        
        return result
    
    def validate_xml_schema(self, xml_element: ET.Element) -> bool:
        """Validate XML against schema (simplified)."""
        # Simplified validation - just check if it has required elements
        required_elements = ['id', 'name']
        element_tags = [child.tag for child in xml_element]
        
        for required in required_elements:
            if required not in element_tags:
                print(f"XMLDataService: Missing required element: {required}")
                return False
        
        print("XMLDataService: XML validation passed")
        return True


class LegacyPaymentGateway:
    """Legacy payment gateway with old interface."""
    
    def __init__(self, gateway_id: str):
        self.gateway_id = gateway_id
        self.transactions = {}
        self.transaction_counter = 1000
    
    def charge_credit_card(self, card_number: str, expiry: str, cvv: str,
                          amount_cents: int) -> Dict[str, Any]:
        """Charge credit card (amount in cents)."""
        transaction_id = f"TXN_{self.transaction_counter}"
        self.transaction_counter += 1
        
        # Simulate payment processing
        success = len(card_number) == 16 and len(cvv) == 3
        
        result = {
            'transaction_id': transaction_id,
            'success': success,
            'amount_cents': amount_cents,
            'gateway_response': 'APPROVED' if success else 'DECLINED',
            'timestamp': datetime.now().isoformat()
        }
        
        self.transactions[transaction_id] = result
        print(f"LegacyPaymentGateway: Processed charge for ${amount_cents/100:.2f}")
        
        return result
    
    def void_transaction(self, transaction_id: str, amount_cents: int) -> Dict[str, Any]:
        """Void a transaction (legacy refund method)."""
        if transaction_id in self.transactions:
            void_result = {
                'void_id': f"VOID_{self.transaction_counter}",
                'original_transaction': transaction_id,
                'voided_amount_cents': amount_cents,
                'status': 'VOIDED',
                'timestamp': datetime.now().isoformat()
            }
            self.transaction_counter += 1
            print(f"LegacyPaymentGateway: Voided transaction {transaction_id}")
            return void_result
        else:
            return {'error': 'Transaction not found'}
    
    def get_transaction_details(self, transaction_id: str) -> Dict[str, Any]:
        """Get transaction details."""
        return self.transactions.get(transaction_id, {'error': 'Transaction not found'})


# ============================================================================
# OBJECT ADAPTERS (Using Composition)
# ============================================================================

class AudioPlayerAdapter(MediaPlayer):
    """Adapter for legacy audio player using composition."""
    
    def __init__(self, legacy_player: LegacyAudioPlayer):
        self.legacy_player = legacy_player
        print("AudioPlayerAdapter: Created adapter for LegacyAudioPlayer")
    
    def play(self, filename: str) -> bool:
        """Adapt play method to legacy interface."""
        success = self.legacy_player.load_audio_file(filename)
        if success:
            self.legacy_player.start_playback()
        return success
    
    def stop(self) -> bool:
        """Adapt stop method to legacy interface."""
        self.legacy_player.halt_playback()
        return True
    
    def get_duration(self) -> float:
        """Adapt duration method to legacy interface."""
        return self.legacy_player.get_track_length()
    
    def get_current_position(self) -> float:
        """Adapt position method to legacy interface."""
        return self.legacy_player.get_playback_position()


class VideoPlayerAdapter(MediaPlayer):
    """Adapter for third-party video player using composition."""
    
    def __init__(self, video_player: ThirdPartyVideoPlayer):
        self.video_player = video_player
        print("VideoPlayerAdapter: Created adapter for ThirdPartyVideoPlayer")
    
    def play(self, filename: str) -> bool:
        """Adapt play method to third-party interface."""
        init_result = self.video_player.initialize_video(filename)
        if init_result == 0:  # Success
            play_result = self.video_player.begin_playback()
            return play_result == 0
        return False
    
    def stop(self) -> bool:
        """Adapt stop method to third-party interface."""
        result = self.video_player.terminate_playback()
        return result == 0
    
    def get_duration(self) -> float:
        """Adapt duration method (convert from milliseconds to seconds)."""
        duration_ms = self.video_player.get_duration_milliseconds()
        return duration_ms / 1000.0
    
    def get_current_position(self) -> float:
        """Adapt position method (convert from milliseconds to seconds)."""
        position_ms = self.video_player.get_position_milliseconds()
        return position_ms / 1000.0


class XMLDataAdapter(DataProcessor):
    """Adapter for XML data service to work with JSON-like interface."""
    
    def __init__(self, xml_service: XMLDataService):
        self.xml_service = xml_service
        print("XMLDataAdapter: Created adapter for XMLDataService")
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data by converting JSON to XML and back."""
        try:
            # Convert input dict to XML string
            xml_string = self._dict_to_xml(data)
            
            # Parse XML using the service
            xml_element = self.xml_service.parse_xml_string(xml_string)
            if xml_element is None:
                return {'error': 'Failed to parse XML'}
            
            # Convert back to dictionary
            result = self.xml_service.xml_to_dict(xml_element)
            
            # Add processing metadata
            result['_processed_by'] = 'XMLDataAdapter'
            result['_processed_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {'error': f'Processing failed: {str(e)}'}
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data by converting to XML and using XML validation."""
        try:
            xml_string = self._dict_to_xml(data)
            xml_element = self.xml_service.parse_xml_string(xml_string)
            if xml_element is None:
                return False
            
            return self.xml_service.validate_xml_schema(xml_element)
        except Exception:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Return supported formats."""
        return ['json', 'xml', 'dict']
    
    def _dict_to_xml(self, data: Dict[str, Any], root_name: str = 'root') -> str:
        """Convert dictionary to XML string."""
        root = ET.Element(root_name)
        self._dict_to_xml_element(data, root)
        return ET.tostring(root, encoding='unicode')
    
    def _dict_to_xml_element(self, data: Any, parent: ET.Element) -> None:
        """Recursively convert dictionary to XML elements."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key.startswith('@'):
                    # Handle attributes
                    continue
                elif isinstance(value, (dict, list)):
                    child = ET.SubElement(parent, str(key))
                    self._dict_to_xml_element(value, child)
                else:
                    child = ET.SubElement(parent, str(key))
                    child.text = str(value)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, 'item')
                self._dict_to_xml_element(item, child)
        else:
            parent.text = str(data)


class PaymentGatewayAdapter(PaymentProcessor):
    """Adapter for legacy payment gateway."""
    
    def __init__(self, legacy_gateway: LegacyPaymentGateway):
        self.legacy_gateway = legacy_gateway
        print(f"PaymentGatewayAdapter: Created adapter for gateway {legacy_gateway.gateway_id}")
    
    def process_payment(self, amount: float, currency: str,
                       payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt payment processing to legacy interface."""
        try:
            # Convert amount to cents (legacy expects cents)
            amount_cents = int(amount * 100)
            
            # Extract card details
            card_number = payment_details.get('card_number', '')
            expiry = payment_details.get('expiry', '')
            cvv = payment_details.get('cvv', '')
            
            # Process using legacy gateway
            result = self.legacy_gateway.charge_credit_card(
                card_number, expiry, cvv, amount_cents
            )
            
            # Adapt result to modern format
            return {
                'transaction_id': result['transaction_id'],
                'status': 'success' if result['success'] else 'failed',
                'amount': amount,
                'currency': currency,
                'gateway_response': result['gateway_response'],
                'timestamp': result['timestamp']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict[str, Any]:
        """Adapt refund processing to legacy void method."""
        try:
            amount_cents = int(amount * 100)
            result = self.legacy_gateway.void_transaction(transaction_id, amount_cents)
            
            if 'error' in result:
                return {
                    'status': 'failed',
                    'error_message': result['error']
                }
            
            return {
                'refund_id': result['void_id'],
                'original_transaction': result['original_transaction'],
                'status': 'success',
                'amount': amount,
                'timestamp': result['timestamp']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def get_transaction_status(self, transaction_id: str) -> str:
        """Get transaction status from legacy gateway."""
        details = self.legacy_gateway.get_transaction_details(transaction_id)
        
        if 'error' in details:
            return 'not_found'
        
        if details.get('success'):
            return 'completed'
        else:
            return 'failed'


# ============================================================================
# TWO-WAY ADAPTER
# ============================================================================

class TwoWayMediaAdapter(MediaPlayer):
    """Two-way adapter that can work with both audio and video players."""
    
    def __init__(self):
        self.audio_player: Optional[LegacyAudioPlayer] = None
        self.video_player: Optional[ThirdPartyVideoPlayer] = None
        self.current_player_type = None
        print("TwoWayMediaAdapter: Created two-way media adapter")
    
    def set_audio_player(self, audio_player: LegacyAudioPlayer) -> None:
        """Set audio player adaptee."""
        self.audio_player = audio_player
        print("TwoWayMediaAdapter: Audio player set")
    
    def set_video_player(self, video_player: ThirdPartyVideoPlayer) -> None:
        """Set video player adaptee."""
        self.video_player = video_player
        print("TwoWayMediaAdapter: Video player set")
    
    def play(self, filename: str) -> bool:
        """Play file using appropriate player based on extension."""
        file_extension = filename.split('.')[-1].lower()
        
        if file_extension in ['mp3', 'wav', 'flac'] and self.audio_player:
            self.current_player_type = 'audio'
            success = self.audio_player.load_audio_file(filename)
            if success:
                self.audio_player.start_playback()
            return success
            
        elif file_extension in ['mp4', 'avi', 'mkv'] and self.video_player:
            self.current_player_type = 'video'
            init_result = self.video_player.initialize_video(filename)
            if init_result == 0:
                play_result = self.video_player.begin_playback()
                return play_result == 0
            return False
        
        else:
            print(f"TwoWayMediaAdapter: Unsupported file type or no player available: {filename}")
            return False
    
    def stop(self) -> bool:
        """Stop current playback."""
        if self.current_player_type == 'audio' and self.audio_player:
            self.audio_player.halt_playback()
            return True
        elif self.current_player_type == 'video' and self.video_player:
            result = self.video_player.terminate_playback()
            return result == 0
        return False
    
    def get_duration(self) -> float:
        """Get duration from current player."""
        if self.current_player_type == 'audio' and self.audio_player:
            return self.audio_player.get_track_length()
        elif self.current_player_type == 'video' and self.video_player:
            return self.video_player.get_duration_milliseconds() / 1000.0
        return 0.0
    
    def get_current_position(self) -> float:
        """Get position from current player."""
        if self.current_player_type == 'audio' and self.audio_player:
            return self.audio_player.get_playback_position()
        elif self.current_player_type == 'video' and self.video_player:
            return self.video_player.get_position_milliseconds() / 1000.0
        return 0.0


# ============================================================================
# PLUGGABLE ADAPTER SYSTEM
# ============================================================================

class AdapterRegistry:
    """Registry for managing different adapters."""
    
    def __init__(self):
        self.media_adapters: Dict[str, MediaPlayer] = {}
        self.data_adapters: Dict[str, DataProcessor] = {}
        self.payment_adapters: Dict[str, PaymentProcessor] = {}
    
    def register_media_adapter(self, name: str, adapter: MediaPlayer) -> None:
        """Register a media adapter."""
        self.media_adapters[name] = adapter
        print(f"AdapterRegistry: Registered media adapter '{name}'")
    
    def register_data_adapter(self, name: str, adapter: DataProcessor) -> None:
        """Register a data adapter."""
        self.data_adapters[name] = adapter
        print(f"AdapterRegistry: Registered data adapter '{name}'")
    
    def register_payment_adapter(self, name: str, adapter: PaymentProcessor) -> None:
        """Register a payment adapter."""
        self.payment_adapters[name] = adapter
        print(f"AdapterRegistry: Registered payment adapter '{name}'")
    
    def get_media_adapter(self, name: str) -> Optional[MediaPlayer]:
        """Get media adapter by name."""
        return self.media_adapters.get(name)
    
    def get_data_adapter(self, name: str) -> Optional[DataProcessor]:
        """Get data adapter by name."""
        return self.data_adapters.get(name)
    
    def get_payment_adapter(self, name: str) -> Optional[PaymentProcessor]:
        """Get payment adapter by name."""
        return self.payment_adapters.get(name)
    
    def list_adapters(self) -> Dict[str, List[str]]:
        """List all registered adapters."""
        return {
            'media': list(self.media_adapters.keys()),
            'data': list(self.data_adapters.keys()),
            'payment': list(self.payment_adapters.keys())
        }


# ============================================================================
# ADAPTER CHAIN
# ============================================================================

class ChainedDataAdapter(DataProcessor):
    """Adapter that chains multiple data processors."""
    
    def __init__(self):
        self.processors: List[DataProcessor] = []
        print("ChainedDataAdapter: Created chained data adapter")
    
    def add_processor(self, processor: DataProcessor) -> None:
        """Add a processor to the chain."""
        self.processors.append(processor)
        print(f"ChainedDataAdapter: Added processor {processor.__class__.__name__}")
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the entire chain."""
        current_data = data.copy()
        processing_history = []
        
        for i, processor in enumerate(self.processors):
            try:
                print(f"ChainedDataAdapter: Processing with {processor.__class__.__name__}")
                result = processor.process_data(current_data)
                
                if 'error' in result:
                    return {
                        'error': f'Chain failed at step {i+1}: {result["error"]}',
                        'processing_history': processing_history
                    }
                
                current_data = result
                processing_history.append({
                    'step': i + 1,
                    'processor': processor.__class__.__name__,
                    'status': 'success'
                })
                
            except Exception as e:
                return {
                    'error': f'Chain failed at step {i+1}: {str(e)}',
                    'processing_history': processing_history
                }
        
        # Add chain metadata
        current_data['_chain_processing_history'] = processing_history
        current_data['_chain_length'] = len(self.processors)
        
        return current_data
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data using all processors in chain."""
        for processor in self.processors:
            if not processor.validate_data(data):
                return False
        return True
    
    def get_supported_formats(self) -> List[str]:
        """Get union of all supported formats."""
        all_formats = set()
        for processor in self.processors:
            all_formats.update(processor.get_supported_formats())
        return list(all_formats)


def demonstrate_adapter_pattern():
    """
    Demonstrate Adapter pattern implementations.
    """
    print("=== ADAPTER PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Object Adapters
    print("1. BASIC OBJECT ADAPTERS:")
    
    # Create legacy systems
    legacy_audio = LegacyAudioPlayer()
    third_party_video = ThirdPartyVideoPlayer()
    
    # Create adapters
    audio_adapter = AudioPlayerAdapter(legacy_audio)
    video_adapter = VideoPlayerAdapter(third_party_video)
    
    # Use through common interface
    media_players = [audio_adapter, video_adapter]
    test_files = ["song.mp3", "movie.mp4"]
    
    for i, player in enumerate(media_players):
        filename = test_files[i]
        print(f"\n   Testing {player.__class__.__name__} with {filename}:")
        
        success = player.play(filename)
        print(f"     Play result: {success}")
        
        if success:
            duration = player.get_duration()
            position = player.get_current_position()
            print(f"     Duration: {duration:.1f}s")
            print(f"     Position: {position:.1f}s")
            
            player.stop()
            print(f"     Stopped playback")
    
    print()
    
    # 2. Data Processing Adapter
    print("2. DATA PROCESSING ADAPTER:")
    
    xml_service = XMLDataService()
    xml_adapter = XMLDataAdapter(xml_service)
    
    # Test data processing
    test_data = {
        'id': '12345',
        'name': 'John Doe',
        'email': 'john@example.com',
        'address': {
            'street': '123 Main St',
            'city': 'Anytown',
            'zip': '12345'
        },
        'tags': ['customer', 'premium']
    }
    
    print("   Processing test data through XML adapter:")
    print(f"   Input data keys: {list(test_data.keys())}")
    
    # Validate data
    is_valid = xml_adapter.validate_data(test_data)
    print(f"   Data validation: {is_valid}")
    
    # Process data
    processed_data = xml_adapter.process_data(test_data)
    
    if 'error' not in processed_data:
        print(f"   Processing successful")
        print(f"   Output data keys: {list(processed_data.keys())}")
        print(f"   Processed by: {processed_data.get('_processed_by')}")
        print(f"   Supported formats: {xml_adapter.get_supported_formats()}")
    else:
        print(f"   Processing failed: {processed_data['error']}")
    
    print()
    
    # 3. Payment Gateway Adapter
    print("3. PAYMENT GATEWAY ADAPTER:")
    
    legacy_gateway = LegacyPaymentGateway("LEGACY_GW_001")
    payment_adapter = PaymentGatewayAdapter(legacy_gateway)
    
    # Test payment processing
    payment_details = {
        'card_number': '1234567890123456',
        'expiry': '12/25',
        'cvv': '123',
        'cardholder_name': 'John Doe'
    }
    
    print("   Processing payment through adapter:")
    payment_result = payment_adapter.process_payment(99.99, 'USD', payment_details)
    
    print(f"   Payment status: {payment_result['status']}")
    if payment_result['status'] == 'success':
        transaction_id = payment_result['transaction_id']
        print(f"   Transaction ID: {transaction_id}")
        print(f"   Amount: ${payment_result['amount']}")
        print(f"   Gateway response: {payment_result['gateway_response']}")
        
        # Test transaction status
        status = payment_adapter.get_transaction_status(transaction_id)
        print(f"   Transaction status: {status}")
        
        # Test refund
        print("\n   Processing refund:")
        refund_result = payment_adapter.refund_payment(transaction_id, 99.99)
        print(f"   Refund status: {refund_result['status']}")
        if refund_result['status'] == 'success':
            print(f"   Refund ID: {refund_result['refund_id']}")
    
    print()
    
    # 4. Two-Way Adapter
    print("4. TWO-WAY ADAPTER:")
    
    two_way_adapter = TwoWayMediaAdapter()
    two_way_adapter.set_audio_player(legacy_audio)
    two_way_adapter.set_video_player(third_party_video)
    
    # Test with different file types
    test_files = ["music.mp3", "video.mp4", "document.pdf"]
    
    for filename in test_files:
        print(f"\n   Testing two-way adapter with {filename}:")
        success = two_way_adapter.play(filename)
        
        if success:
            duration = two_way_adapter.get_duration()
            print(f"     Successfully playing, duration: {duration:.1f}s")
            two_way_adapter.stop()
            print(f"     Stopped playback")
        else:
            print(f"     Could not play file (unsupported or no player)")
    
    print()
    
    # 5. Adapter Registry
    print("5. ADAPTER REGISTRY:")
    
    registry = AdapterRegistry()
    
    # Register adapters
    registry.register_media_adapter("audio", audio_adapter)
    registry.register_media_adapter("video", video_adapter)
    registry.register_media_adapter("universal", two_way_adapter)
    registry.register_data_adapter("xml", xml_adapter)
    registry.register_payment_adapter("legacy", payment_adapter)
    
    # List registered adapters
    adapters = registry.list_adapters()
    print("   Registered adapters:")
    for category, adapter_list in adapters.items():
        print(f"     {category}: {adapter_list}")
    
    # Use adapters from registry
    print("\n   Using adapters from registry:")
    
    # Get and use media adapter
    audio_from_registry = registry.get_media_adapter("audio")
    if audio_from_registry:
        success = audio_from_registry.play("test.mp3")
        print(f"     Audio adapter from registry: {success}")
        audio_from_registry.stop()
    
    # Get and use data adapter
    data_from_registry = registry.get_data_adapter("xml")
    if data_from_registry:
        formats = data_from_registry.get_supported_formats()
        print(f"     Data adapter formats: {formats}")
    
    print()
    
    # 6. Chained Adapters
    print("6. CHAINED ADAPTERS:")
    
    # Create multiple data processors for chaining
    class JSONProcessor(DataProcessor):
        """Simple JSON processor."""
        
        def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
            result = data.copy()
            result['_processed_by_json'] = True
            result['_json_timestamp'] = datetime.now().isoformat()
            return result
        
        def validate_data(self, data: Dict[str, Any]) -> bool:
            return isinstance(data, dict)
        
        def get_supported_formats(self) -> List[str]:
            return ['json']
    
    class ValidationProcessor(DataProcessor):
        """Data validation processor."""
        
        def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
            if not self.validate_data(data):
                return {'error': 'Validation failed'}
            
            result = data.copy()
            result['_validation_passed'] = True
            result['_validation_timestamp'] = datetime.now().isoformat()
            return result
        
        def validate_data(self, data: Dict[str, Any]) -> bool:
            required_fields = ['id', 'name']
            return all(field in data for field in required_fields)
        
        def get_supported_formats(self) -> List[str]:
            return ['json', 'xml']
    
    # Create chained adapter
    chained_adapter = ChainedDataAdapter()
    chained_adapter.add_processor(ValidationProcessor())
    chained_adapter.add_processor(JSONProcessor())
    chained_adapter.add_processor(xml_adapter)
    
    # Test chained processing
    chain_test_data = {
        'id': 'CHAIN001',
        'name': 'Chain Test',
        'description': 'Testing chained adapters'
    }
    
    print("   Testing chained data processing:")
    chain_result = chained_adapter.process_data(chain_test_data)
    
    if 'error' not in chain_result:
        print(f"   Chain processing successful")
        print(f"   Chain length: {chain_result.get('_chain_length')}")
        print(f"   Processing history: {len(chain_result.get('_chain_processing_history', []))} steps")
        
        # Show processing steps
        for step in chain_result.get('_chain_processing_history', []):
            print(f"     Step {step['step']}: {step['processor']} - {step['status']}")
    else:
        print(f"   Chain processing failed: {chain_result['error']}")
    
    # Test chain validation
    chain_valid = chained_adapter.validate_data(chain_test_data)
    print(f"   Chain validation: {chain_valid}")
    
    # Test with invalid data
    invalid_data = {'description': 'Missing required fields'}
    print("\n   Testing chain with invalid data:")
    invalid_result = chained_adapter.process_data(invalid_data)
    if 'error' in invalid_result:
        print(f"   Chain correctly rejected invalid data: {invalid_result['error']}")
    
    print()
    
    # 7. Adapter Pattern Benefits
    print("7. ADAPTER PATTERN BENEFITS:")
    print("   ✓ Interface Compatibility: Makes incompatible interfaces work together")
    print("   ✓ Legacy Integration: Allows integration of legacy systems")
    print("   ✓ Third-party Integration: Adapts third-party libraries to your interface")
    print("   ✓ Separation of Concerns: Keeps adaptation logic separate")
    print("   ✓ Flexibility: Multiple adapters for different implementations")
    print("   ✓ Reusability: Adapters can be reused across different contexts")
    print("   ✓ Two-way Communication: Bidirectional adapters for complex scenarios")
    print("   ✓ Chain Processing: Multiple adapters can be chained together")
    print()
    
    print("=== ADAPTER PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_adapter_pattern()
