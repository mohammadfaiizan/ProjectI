#!/usr/bin/env python3
"""
Multimodal RAG Systems: Cross-Modal Information Retrieval and Processing
=======================================================================

WHAT IS THE PROBLEM?
==================
Traditional RAG systems only work with text:
- Cannot process images, videos, audio, or other media types
- Miss visual information and context in documents
- Cannot understand charts, graphs, diagrams, and infographics
- Cannot correlate information across different modalities
- Limited to text-only knowledge bases and responses
- Cannot handle real-world mixed-media content

Example: Medical Diagnosis Complexity
TEXT-ONLY RAG (Traditional):
- Query: "What does this skin condition look like?"
- System: Can only describe in text, cannot see actual image
- Misses: Visual patterns, color changes, texture details
- Result: Incomplete diagnosis without visual analysis

REAL WORLD EXAMPLE:
=================
How does Google Lens work?

GOOGLE LENS MULTIMODAL SYSTEM:
1. VISUAL RECOGNITION: Identifies objects, text, landmarks in images
2. CROSS-MODAL SEARCH: Combines visual and text information
3. CONTEXTUAL UNDERSTANDING: Relates visual content to knowledge base
4. MIXED MEDIA RESULTS: Returns text, images, and interactive content
5. REAL-WORLD INTEGRATION: Works with camera, photos, and live video
6. MULTILINGUAL SUPPORT: Text extraction and translation across languages
7. ACTION INTEGRATION: Enables shopping, navigation, and learning

BENEFITS OF MULTIMODAL RAG:
- Complete understanding of mixed-media content
- Visual question answering and image analysis
- Document processing with charts and diagrams
- Cross-modal information correlation and verification
- Rich, media-enhanced responses and explanations
- Real-world applicability across all content types

THE MULTIMODAL ADVANTAGE:
=======================
TRADITIONAL RAG: Text query → Text documents → Text response
MULTIMODAL RAG: Any modality → Cross-modal retrieval → Rich multimedia response

MULTIMODAL COMPONENTS:
====================
1. VISION PROCESSING: Image and video understanding
2. AUDIO PROCESSING: Speech and sound analysis
3. TEXT PROCESSING: Traditional NLP and document understanding
4. CROSS-MODAL EMBEDDING: Unified representation across modalities
5. MULTIMODAL RETRIEVAL: Search across different media types
6. FUSION ENGINES: Combining information from multiple modalities
7. RICH RESPONSE GENERATION: Creating multimedia answers

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI to work with complete real-world information
- Provides visual understanding and analysis capabilities
- Supports complex document types with mixed content
- Powers next-generation search and knowledge systems
- Critical for applications requiring visual intelligence
- Bridges the gap between AI and human multimodal perception
"""

import asyncio
import time
import json
import uuid
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import re
import math
import numpy as np
from datetime import datetime
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ModalityType(Enum):
    """Types of modalities supported"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"    # PDF, Word, etc.
    CHART = "chart"         # Data visualizations
    DIAGRAM = "diagram"     # Technical diagrams
    TABLE = "table"         # Structured data

class ProcessingStatus(Enum):
    """Status of multimodal processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class EmbeddingSpace(Enum):
    """Types of embedding spaces"""
    VISUAL = "visual"           # Image/video embeddings
    TEXTUAL = "textual"         # Text embeddings
    AUDIO = "audio"             # Audio embeddings
    MULTIMODAL = "multimodal"   # Cross-modal unified space
    SEMANTIC = "semantic"       # High-level concept space

@dataclass
class MultimodalDocument:
    """Document containing multiple modalities"""
    doc_id: str
    title: str
    
    # Content by modality
    text_content: Optional[str] = None
    image_content: List[Dict[str, Any]] = field(default_factory=list)  # Images with metadata
    audio_content: List[Dict[str, Any]] = field(default_factory=list)  # Audio with metadata
    video_content: List[Dict[str, Any]] = field(default_factory=list)  # Video with metadata
    
    # Processed information
    extracted_text: str = ""           # OCR and speech-to-text results
    visual_descriptions: List[str] = field(default_factory=list)
    detected_objects: List[Dict] = field(default_factory=list)
    chart_data: List[Dict] = field(default_factory=list)
    
    # Embeddings
    embeddings: Dict[EmbeddingSpace, List[float]] = field(default_factory=dict)
    
    # Metadata
    source: str = ""
    creation_time: datetime = field(default_factory=datetime.now)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())
    
    def get_primary_modality(self) -> ModalityType:
        """Determine primary modality of document"""
        if self.text_content:
            return ModalityType.TEXT
        elif self.image_content:
            return ModalityType.IMAGE
        elif self.video_content:
            return ModalityType.VIDEO
        elif self.audio_content:
            return ModalityType.AUDIO
        else:
            return ModalityType.TEXT
    
    def get_available_modalities(self) -> List[ModalityType]:
        """Get list of available modalities in document"""
        modalities = []
        
        if self.text_content or self.extracted_text:
            modalities.append(ModalityType.TEXT)
        if self.image_content:
            modalities.append(ModalityType.IMAGE)
        if self.audio_content:
            modalities.append(ModalityType.AUDIO)
        if self.video_content:
            modalities.append(ModalityType.VIDEO)
        if self.chart_data:
            modalities.append(ModalityType.CHART)
        
        return modalities
    
    def get_content_summary(self) -> str:
        """Get summary of all content"""
        summary_parts = []
        
        if self.text_content:
            summary_parts.append(f"Text: {self.text_content[:100]}...")
        
        if self.extracted_text:
            summary_parts.append(f"Extracted: {self.extracted_text[:100]}...")
        
        if self.visual_descriptions:
            summary_parts.append(f"Visual: {'; '.join(self.visual_descriptions[:2])}")
        
        if self.detected_objects:
            objects = [obj.get('label', 'object') for obj in self.detected_objects[:3]]
            summary_parts.append(f"Objects: {', '.join(objects)}")
        
        return " | ".join(summary_parts)

@dataclass
class MultimodalQuery:
    """Query with multiple modalities"""
    query_id: str
    
    # Query content by modality
    text_query: Optional[str] = None
    image_query: Optional[Dict[str, Any]] = None
    audio_query: Optional[Dict[str, Any]] = None
    
    # Query intent and context
    intent: str = "search"  # search, analyze, compare, explain
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing preferences
    preferred_modalities: List[ModalityType] = field(default_factory=list)
    response_format: str = "mixed"  # text, visual, audio, mixed
    
    # Embeddings
    query_embeddings: Dict[EmbeddingSpace, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())
    
    def get_query_modalities(self) -> List[ModalityType]:
        """Get modalities present in query"""
        modalities = []
        
        if self.text_query:
            modalities.append(ModalityType.TEXT)
        if self.image_query:
            modalities.append(ModalityType.IMAGE)
        if self.audio_query:
            modalities.append(ModalityType.AUDIO)
        
        return modalities

class VisionProcessor:
    """Processes images and extracts visual information"""
    
    def __init__(self):
        self.supported_formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
        self.logger = logging.getLogger("VisionProcessor")
    
    async def process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single image and extract information"""
        
        try:
            # Simulate image processing (in real implementation, use computer vision models)
            image_info = {
                'width': image_data.get('width', 800),
                'height': image_data.get('height', 600),
                'format': image_data.get('format', 'jpg'),
                'size_bytes': image_data.get('size', 1024000)
            }
            
            # Simulate object detection
            detected_objects = await self._detect_objects(image_data)
            
            # Simulate OCR (text extraction)
            extracted_text = await self._extract_text_ocr(image_data)
            
            # Simulate image captioning
            description = await self._generate_image_description(image_data)
            
            # Simulate visual embedding
            visual_embedding = await self._compute_visual_embedding(image_data)
            
            # Analyze if image contains charts/diagrams
            chart_analysis = await self._analyze_charts_diagrams(image_data)
            
            processing_result = {
                'image_info': image_info,
                'detected_objects': detected_objects,
                'extracted_text': extracted_text,
                'description': description,
                'visual_embedding': visual_embedding,
                'chart_analysis': chart_analysis,
                'confidence_score': 0.85,
                'processing_time': 0.5
            }
            
            self.logger.debug(f"Processed image: {len(detected_objects)} objects, {len(extracted_text)} chars text")
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return {
                'error': str(e),
                'confidence_score': 0.0,
                'processing_time': 0.0
            }
    
    async def _detect_objects(self, image_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate object detection in image"""
        
        # Simulate different types of objects based on image context
        possible_objects = [
            {'label': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]},
            {'label': 'car', 'confidence': 0.88, 'bbox': [300, 150, 500, 250]},
            {'label': 'building', 'confidence': 0.92, 'bbox': [50, 50, 400, 200]},
            {'label': 'text', 'confidence': 0.85, 'bbox': [20, 20, 600, 100]},
            {'label': 'chart', 'confidence': 0.78, 'bbox': [100, 200, 500, 400]},
            {'label': 'logo', 'confidence': 0.90, 'bbox': [10, 10, 80, 80]},
            {'label': 'table', 'confidence': 0.82, 'bbox': [50, 300, 550, 500]}
        ]
        
        # Return random subset based on image content hint
        import random
        num_objects = random.randint(1, 4)
        return random.sample(possible_objects, num_objects)
    
    async def _extract_text_ocr(self, image_data: Dict[str, Any]) -> str:
        """Simulate OCR text extraction"""
        
        # Simulate OCR results based on image context
        possible_texts = [
            "Annual Revenue Growth Chart 2024",
            "Company Performance Metrics Q3 2024",
            "Market Analysis: Technology Sector Trends",
            "Product Specifications and Features",
            "Financial Summary Report",
            "Customer Satisfaction Survey Results",
            "Strategic Planning Overview",
            "Technical Architecture Diagram"
        ]
        
        import random
        
        # Simulate presence of text in image
        if random.random() > 0.3:  # 70% chance of text
            return random.choice(possible_texts)
        
        return ""
    
    async def _generate_image_description(self, image_data: Dict[str, Any]) -> str:
        """Simulate image captioning/description"""
        
        descriptions = [
            "A business chart showing upward trending data with colorful bars and graphs",
            "Professional office environment with people working at computers",
            "Technical diagram illustrating system architecture with connected components",
            "Product showcase featuring multiple items arranged on a clean background",
            "Corporate presentation slide with text and visual elements",
            "Data visualization displaying key performance indicators and metrics",
            "Team meeting scenario with people around a conference table",
            "Modern building exterior with glass facade and architectural details"
        ]
        
        import random
        return random.choice(descriptions)
    
    async def _compute_visual_embedding(self, image_data: Dict[str, Any]) -> List[float]:
        """Simulate visual feature embedding"""
        
        # Simulate visual embedding (in real implementation, use vision transformer or CNN)
        import numpy as np
        
        np.random.seed(hash(str(image_data)) % 2**32)
        embedding = np.random.normal(0, 1, 512)
        
        # Add some structure based on image characteristics
        image_type = image_data.get('type', 'general')
        
        if image_type == 'chart':
            embedding[:50] += 0.5  # Chart-specific features
        elif image_type == 'document':
            embedding[50:100] += 0.5  # Document-specific features
        elif image_type == 'photo':
            embedding[100:150] += 0.5  # Photo-specific features
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    async def _analyze_charts_diagrams(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze charts and diagrams in image"""
        
        # Simulate chart/diagram analysis
        chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'flowchart', 'diagram', 'table']
        
        import random
        
        # Simulate chart detection
        if random.random() > 0.6:  # 40% chance of chart
            chart_type = random.choice(chart_types)
            
            return {
                'contains_chart': True,
                'chart_type': chart_type,
                'data_points': random.randint(3, 20),
                'title': f"Detected {chart_type.replace('_', ' ')}",
                'confidence': random.uniform(0.7, 0.95)
            }
        
        return {
            'contains_chart': False,
            'confidence': 0.0
        }

class AudioProcessor:
    """Processes audio and extracts information"""
    
    def __init__(self):
        self.supported_formats = ['mp3', 'wav', 'flac', 'm4a', 'ogg']
        self.logger = logging.getLogger("AudioProcessor")
    
    async def process_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio and extract information"""
        
        try:
            # Simulate audio processing
            audio_info = {
                'duration': audio_data.get('duration', 120.0),  # seconds
                'sample_rate': audio_data.get('sample_rate', 44100),
                'channels': audio_data.get('channels', 2),
                'format': audio_data.get('format', 'mp3'),
                'size_bytes': audio_data.get('size', 5000000)
            }
            
            # Simulate speech-to-text
            transcribed_text = await self._speech_to_text(audio_data)
            
            # Simulate audio classification
            audio_classification = await self._classify_audio(audio_data)
            
            # Simulate speaker identification
            speaker_info = await self._identify_speakers(audio_data)
            
            # Simulate audio embedding
            audio_embedding = await self._compute_audio_embedding(audio_data)
            
            processing_result = {
                'audio_info': audio_info,
                'transcribed_text': transcribed_text,
                'classification': audio_classification,
                'speaker_info': speaker_info,
                'audio_embedding': audio_embedding,
                'confidence_score': 0.82,
                'processing_time': 2.1
            }
            
            self.logger.debug(f"Processed audio: {len(transcribed_text)} chars transcribed")
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return {
                'error': str(e),
                'confidence_score': 0.0,
                'processing_time': 0.0
            }
    
    async def _speech_to_text(self, audio_data: Dict[str, Any]) -> str:
        """Simulate speech-to-text conversion"""
        
        sample_transcriptions = [
            "Welcome to our quarterly business review meeting. Today we'll be discussing market performance and growth strategies.",
            "The artificial intelligence market is expected to grow significantly over the next five years with major opportunities in healthcare and finance.",
            "Our product development team has made excellent progress on the new features including improved user interface and enhanced security.",
            "Customer satisfaction scores have increased by 15% this quarter thanks to our improved support processes and faster response times.",
            "The financial analysis shows strong revenue growth across all major product categories with particular strength in international markets.",
            "Technical implementation of the new system architecture is proceeding on schedule with successful integration testing completed.",
            "Market research indicates growing demand for sustainable technology solutions among enterprise customers.",
            "Our competitive analysis reveals significant advantages in pricing and feature completeness compared to major competitors."
        ]
        
        import random
        return random.choice(sample_transcriptions)
    
    async def _classify_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate audio classification"""
        
        classifications = [
            {'type': 'speech', 'confidence': 0.95},
            {'type': 'music', 'confidence': 0.88},
            {'type': 'presentation', 'confidence': 0.92},
            {'type': 'meeting', 'confidence': 0.87},
            {'type': 'interview', 'confidence': 0.90},
            {'type': 'podcast', 'confidence': 0.85}
        ]
        
        import random
        return random.choice(classifications)
    
    async def _identify_speakers(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate speaker identification"""
        
        import random
        
        num_speakers = random.randint(1, 4)
        
        speakers = []
        for i in range(num_speakers):
            speakers.append({
                'speaker_id': f"speaker_{i+1}",
                'confidence': random.uniform(0.7, 0.95),
                'segments': random.randint(2, 8)
            })
        
        return {
            'num_speakers': num_speakers,
            'speakers': speakers
        }
    
    async def _compute_audio_embedding(self, audio_data: Dict[str, Any]) -> List[float]:
        """Simulate audio feature embedding"""
        
        import numpy as np
        
        np.random.seed(hash(str(audio_data)) % 2**32)
        embedding = np.random.normal(0, 1, 256)
        
        # Add structure based on audio type
        audio_type = audio_data.get('type', 'speech')
        
        if audio_type == 'speech':
            embedding[:32] += 0.5
        elif audio_type == 'music':
            embedding[32:64] += 0.5
        elif audio_type == 'presentation':
            embedding[64:96] += 0.5
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()

class CrossModalEmbedding:
    """Creates unified embeddings across different modalities"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.modality_weights = {
            ModalityType.TEXT: 1.0,
            ModalityType.IMAGE: 0.8,
            ModalityType.AUDIO: 0.7,
            ModalityType.VIDEO: 0.9
        }
        
        self.logger = logging.getLogger("CrossModalEmbedding")
    
    async def create_multimodal_embedding(self, document: MultimodalDocument) -> List[float]:
        """Create unified embedding from multiple modalities"""
        
        try:
            embeddings = []
            weights = []
            
            # Collect embeddings from different modalities
            for embedding_space, embedding in document.embeddings.items():
                if embedding:
                    embeddings.append(np.array(embedding))
                    
                    # Determine weight based on modality and quality
                    if embedding_space == EmbeddingSpace.TEXTUAL:
                        weight = self.modality_weights[ModalityType.TEXT]
                    elif embedding_space == EmbeddingSpace.VISUAL:
                        weight = self.modality_weights[ModalityType.IMAGE]
                    elif embedding_space == EmbeddingSpace.AUDIO:
                        weight = self.modality_weights[ModalityType.AUDIO]
                    else:
                        weight = 0.5
                    
                    weights.append(weight)
            
            if not embeddings:
                # Create default embedding
                return np.zeros(self.embedding_dim).tolist()
            
            # Ensure all embeddings have same dimension
            normalized_embeddings = []
            for emb in embeddings:
                if len(emb) != self.embedding_dim:
                    # Resize embedding to target dimension
                    if len(emb) > self.embedding_dim:
                        emb = emb[:self.embedding_dim]
                    else:
                        emb = np.pad(emb, (0, self.embedding_dim - len(emb)))
                
                # Normalize embedding
                if np.linalg.norm(emb) > 0:
                    emb = emb / np.linalg.norm(emb)
                
                normalized_embeddings.append(emb)
            
            # Weighted average of embeddings
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            multimodal_embedding = np.zeros(self.embedding_dim)
            for emb, weight in zip(normalized_embeddings, weights):
                multimodal_embedding += weight * emb
            
            # Final normalization
            if np.linalg.norm(multimodal_embedding) > 0:
                multimodal_embedding = multimodal_embedding / np.linalg.norm(multimodal_embedding)
            
            return multimodal_embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Multimodal embedding creation failed: {e}")
            return np.zeros(self.embedding_dim).tolist()
    
    async def compute_cross_modal_similarity(self, embedding1: List[float], 
                                           embedding2: List[float]) -> float:
        """Compute similarity between cross-modal embeddings"""
        
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Ensure same dimension
            min_dim = min(len(vec1), len(vec2))
            vec1 = vec1[:min_dim]
            vec2 = vec2[:min_dim]
            
            # Cosine similarity
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return float(similarity)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Cross-modal similarity computation failed: {e}")
            return 0.0

class MultimodalRetriever:
    """Retrieves documents across multiple modalities"""
    
    def __init__(self):
        self.documents: Dict[str, MultimodalDocument] = {}
        self.cross_modal_embedder = CrossModalEmbedding()
        
        # Indexes by modality
        self.text_index: Dict[str, List[str]] = defaultdict(list)
        self.visual_index: Dict[str, List[str]] = defaultdict(list)
        self.audio_index: Dict[str, List[str]] = defaultdict(list)
        
        self.logger = logging.getLogger("MultimodalRetriever")
    
    async def add_document(self, document: MultimodalDocument) -> None:
        """Add multimodal document to retriever"""
        
        try:
            # Store document
            self.documents[document.doc_id] = document
            
            # Create cross-modal embedding if not exists
            if EmbeddingSpace.MULTIMODAL not in document.embeddings:
                multimodal_embedding = await self.cross_modal_embedder.create_multimodal_embedding(document)
                document.embeddings[EmbeddingSpace.MULTIMODAL] = multimodal_embedding
            
            # Update indexes
            await self._update_indexes(document)
            
            self.logger.debug(f"Added multimodal document: {document.doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add document {document.doc_id}: {e}")
    
    async def multimodal_search(self, query: MultimodalQuery, 
                              top_k: int = 10) -> List[Tuple[MultimodalDocument, float]]:
        """Search across multiple modalities"""
        
        try:
            # Process query embeddings if not provided
            if not query.query_embeddings:
                query.query_embeddings = await self._process_query_embeddings(query)
            
            # Score all documents
            doc_scores = []
            
            for doc_id, document in self.documents.items():
                score = await self._calculate_multimodal_score(query, document)
                if score > 0:
                    doc_scores.append((document, score))
            
            # Sort by score and return top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:top_k]
            
        except Exception as e:
            self.logger.error(f"Multimodal search failed: {e}")
            return []
    
    async def visual_search(self, image_query: Dict[str, Any], 
                          top_k: int = 10) -> List[Tuple[MultimodalDocument, float]]:
        """Search using image query"""
        
        try:
            # Process image query
            vision_processor = VisionProcessor()
            image_result = await vision_processor.process_image(image_query)
            
            if 'visual_embedding' not in image_result:
                return []
            
            query_embedding = image_result['visual_embedding']
            
            # Find visually similar documents
            doc_scores = []
            
            for doc_id, document in self.documents.items():
                if EmbeddingSpace.VISUAL in document.embeddings:
                    similarity = await self.cross_modal_embedder.compute_cross_modal_similarity(
                        query_embedding, document.embeddings[EmbeddingSpace.VISUAL]
                    )
                    
                    if similarity > 0.1:  # Minimum threshold
                        doc_scores.append((document, similarity))
            
            # Sort and return
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:top_k]
            
        except Exception as e:
            self.logger.error(f"Visual search failed: {e}")
            return []
    
    async def _process_query_embeddings(self, query: MultimodalQuery) -> Dict[EmbeddingSpace, List[float]]:
        """Process query to create embeddings"""
        
        embeddings = {}
        
        # Text query embedding
        if query.text_query:
            text_embedding = await self._create_text_embedding(query.text_query)
            embeddings[EmbeddingSpace.TEXTUAL] = text_embedding
        
        # Image query embedding
        if query.image_query:
            vision_processor = VisionProcessor()
            image_result = await vision_processor.process_image(query.image_query)
            if 'visual_embedding' in image_result:
                embeddings[EmbeddingSpace.VISUAL] = image_result['visual_embedding']
        
        # Audio query embedding
        if query.audio_query:
            audio_processor = AudioProcessor()
            audio_result = await audio_processor.process_audio(query.audio_query)
            if 'audio_embedding' in audio_result:
                embeddings[EmbeddingSpace.AUDIO] = audio_result['audio_embedding']
        
        return embeddings
    
    async def _create_text_embedding(self, text: str) -> List[float]:
        """Create text embedding (simulated)"""
        
        import numpy as np
        
        # Simulate text embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.normal(0, 1, 768)
        
        # Add semantic bias based on text content
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['chart', 'graph', 'data', 'analysis']):
            embedding[:50] += 0.3  # Data visualization bias
        
        if any(word in text_lower for word in ['image', 'photo', 'picture', 'visual']):
            embedding[50:100] += 0.3  # Visual content bias
        
        if any(word in text_lower for word in ['audio', 'sound', 'speech', 'voice']):
            embedding[100:150] += 0.3  # Audio content bias
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    async def _calculate_multimodal_score(self, query: MultimodalQuery, 
                                        document: MultimodalDocument) -> float:
        """Calculate relevance score across modalities"""
        
        total_score = 0.0
        weight_sum = 0.0
        
        # Text similarity
        if (EmbeddingSpace.TEXTUAL in query.query_embeddings and 
            EmbeddingSpace.TEXTUAL in document.embeddings):
            
            text_similarity = await self.cross_modal_embedder.compute_cross_modal_similarity(
                query.query_embeddings[EmbeddingSpace.TEXTUAL],
                document.embeddings[EmbeddingSpace.TEXTUAL]
            )
            
            total_score += text_similarity * 1.0
            weight_sum += 1.0
        
        # Visual similarity
        if (EmbeddingSpace.VISUAL in query.query_embeddings and 
            EmbeddingSpace.VISUAL in document.embeddings):
            
            visual_similarity = await self.cross_modal_embedder.compute_cross_modal_similarity(
                query.query_embeddings[EmbeddingSpace.VISUAL],
                document.embeddings[EmbeddingSpace.VISUAL]
            )
            
            total_score += visual_similarity * 0.8
            weight_sum += 0.8
        
        # Audio similarity
        if (EmbeddingSpace.AUDIO in query.query_embeddings and 
            EmbeddingSpace.AUDIO in document.embeddings):
            
            audio_similarity = await self.cross_modal_embedder.compute_cross_modal_similarity(
                query.query_embeddings[EmbeddingSpace.AUDIO],
                document.embeddings[EmbeddingSpace.AUDIO]
            )
            
            total_score += audio_similarity * 0.7
            weight_sum += 0.7
        
        # Cross-modal similarity (if available)
        if (EmbeddingSpace.MULTIMODAL in query.query_embeddings and 
            EmbeddingSpace.MULTIMODAL in document.embeddings):
            
            cross_modal_similarity = await self.cross_modal_embedder.compute_cross_modal_similarity(
                query.query_embeddings[EmbeddingSpace.MULTIMODAL],
                document.embeddings[EmbeddingSpace.MULTIMODAL]
            )
            
            total_score += cross_modal_similarity * 0.9
            weight_sum += 0.9
        
        # Normalize score
        if weight_sum > 0:
            return total_score / weight_sum
        
        return 0.0
    
    async def _update_indexes(self, document: MultimodalDocument) -> None:
        """Update modality-specific indexes"""
        
        # Text index
        if document.text_content or document.extracted_text:
            text_content = (document.text_content or "") + " " + document.extracted_text
            words = text_content.lower().split()
            
            for word in set(words):  # Unique words only
                if len(word) > 2:  # Skip very short words
                    self.text_index[word].append(document.doc_id)
        
        # Visual index (based on detected objects and descriptions)
        visual_terms = []
        
        for obj in document.detected_objects:
            if 'label' in obj:
                visual_terms.append(obj['label'].lower())
        
        for desc in document.visual_descriptions:
            visual_terms.extend(desc.lower().split())
        
        for term in set(visual_terms):
            if len(term) > 2:
                self.visual_index[term].append(document.doc_id)
        
        # Audio index (based on transcribed text)
        if hasattr(document, 'audio_transcripts'):
            for transcript in getattr(document, 'audio_transcripts', []):
                words = transcript.lower().split()
                for word in set(words):
                    if len(word) > 2:
                        self.audio_index[word].append(document.doc_id)

class MultimodalRAGSystem:
    """
    Complete Multimodal RAG System for cross-modal information processing
    
    EXAMPLE USAGE:
    =============
    # Create multimodal RAG system
    rag = MultimodalRAGSystem()
    await rag.initialize()
    
    # Add multimodal document
    doc = MultimodalDocument(
        doc_id="report_001",
        title="Q3 Financial Report",
        text_content="Our quarterly performance shows strong growth...",
        image_content=[{
            'type': 'chart',
            'description': 'Revenue growth chart',
            'data': 'base64_encoded_image_data'
        }]
    )
    
    await rag.add_document(doc)
    
    # Multimodal query - text + image
    query = MultimodalQuery(
        query_id="query_001",
        text_query="Show me financial charts with growth trends",
        image_query={'type': 'chart', 'description': 'sample chart'}
    )
    
    results = await rag.multimodal_search(query)
    
    # Visual search with image
    image_results = await rag.visual_search_by_image(image_data)
    """
    
    def __init__(self):
        # Core processors
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()
        self.cross_modal_embedder = CrossModalEmbedding()
        self.multimodal_retriever = MultimodalRetriever()
        
        # System statistics
        self.system_stats = {
            'documents_processed': 0,
            'images_processed': 0,
            'audio_files_processed': 0,
            'queries_processed': 0,
            'cross_modal_searches': 0,
            'average_processing_time': 0.0
        }
        
        self.logger = logging.getLogger("MultimodalRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize multimodal RAG system"""
        self.logger.info("Multimodal RAG system initialized")
    
    async def add_document(self, document: MultimodalDocument) -> Dict[str, Any]:
        """Add and process multimodal document"""
        
        start_time = time.time()
        
        try:
            # Process images
            for i, image_data in enumerate(document.image_content):
                image_result = await self.vision_processor.process_image(image_data)
                
                if 'error' not in image_result:
                    # Extract information
                    if image_result.get('extracted_text'):
                        document.extracted_text += " " + image_result['extracted_text']
                    
                    if image_result.get('description'):
                        document.visual_descriptions.append(image_result['description'])
                    
                    if image_result.get('detected_objects'):
                        document.detected_objects.extend(image_result['detected_objects'])
                    
                    if image_result.get('chart_analysis'):
                        document.chart_data.append(image_result['chart_analysis'])
                    
                    # Store visual embedding
                    if image_result.get('visual_embedding'):
                        document.embeddings[EmbeddingSpace.VISUAL] = image_result['visual_embedding']
                    
                    self.system_stats['images_processed'] += 1
            
            # Process audio
            for audio_data in document.audio_content:
                audio_result = await self.audio_processor.process_audio(audio_data)
                
                if 'error' not in audio_result:
                    # Extract transcribed text
                    if audio_result.get('transcribed_text'):
                        document.extracted_text += " " + audio_result['transcribed_text']
                    
                    # Store audio embedding
                    if audio_result.get('audio_embedding'):
                        document.embeddings[EmbeddingSpace.AUDIO] = audio_result['audio_embedding']
                    
                    self.system_stats['audio_files_processed'] += 1
            
            # Process text
            if document.text_content:
                text_embedding = await self.multimodal_retriever._create_text_embedding(document.text_content)
                document.embeddings[EmbeddingSpace.TEXTUAL] = text_embedding
            
            # Create cross-modal embedding
            multimodal_embedding = await self.cross_modal_embedder.create_multimodal_embedding(document)
            document.embeddings[EmbeddingSpace.MULTIMODAL] = multimodal_embedding
            
            # Add to retriever
            await self.multimodal_retriever.add_document(document)
            
            # Update status
            document.processing_status = ProcessingStatus.COMPLETED
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.system_stats['documents_processed'] += 1
            self._update_processing_stats(processing_time)
            
            result = {
                'success': True,
                'document_id': document.doc_id,
                'processing_time': processing_time,
                'modalities_processed': len(document.get_available_modalities()),
                'images_processed': len(document.image_content),
                'audio_processed': len(document.audio_content),
                'extracted_text_length': len(document.extracted_text),
                'objects_detected': len(document.detected_objects),
                'visual_descriptions': len(document.visual_descriptions)
            }
            
            self.logger.info(f"Processed multimodal document: {document.doc_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            
            self.logger.error(f"Failed to process document {document.doc_id}: {e}")
            
            return {
                'success': False,
                'document_id': document.doc_id,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def multimodal_search(self, query: MultimodalQuery, 
                              top_k: int = 10) -> Dict[str, Any]:
        """Perform multimodal search"""
        
        start_time = time.time()
        self.system_stats['queries_processed'] += 1
        
        try:
            # Determine if this is cross-modal search
            query_modalities = query.get_query_modalities()
            is_cross_modal = len(query_modalities) > 1
            
            if is_cross_modal:
                self.system_stats['cross_modal_searches'] += 1
            
            # Perform search
            search_results = await self.multimodal_retriever.multimodal_search(query, top_k)
            
            # Format results
            formatted_results = []
            for document, score in search_results:
                result_item = {
                    'document_id': document.doc_id,
                    'title': document.title,
                    'relevance_score': score,
                    'available_modalities': [m.value for m in document.get_available_modalities()],
                    'content_summary': document.get_content_summary(),
                    'confidence_scores': document.confidence_scores,
                    'has_images': len(document.image_content) > 0,
                    'has_audio': len(document.audio_content) > 0,
                    'detected_objects': len(document.detected_objects),
                    'visual_descriptions': document.visual_descriptions[:2],  # Top 2
                    'chart_data': len(document.chart_data) > 0
                }
                formatted_results.append(result_item)
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'query_id': query.query_id,
                'query_modalities': [m.value for m in query_modalities],
                'is_cross_modal': is_cross_modal,
                'results_found': len(formatted_results),
                'results': formatted_results,
                'processing_time': processing_time,
                'search_metadata': {
                    'intent': query.intent,
                    'preferred_modalities': [m.value for m in query.preferred_modalities],
                    'response_format': query.response_format
                }
            }
            
            self.logger.info(f"Multimodal search completed: {len(formatted_results)} results in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multimodal search failed: {e}")
            
            return {
                'success': False,
                'query_id': query.query_id,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def visual_search_by_image(self, image_data: Dict[str, Any], 
                                   top_k: int = 10) -> Dict[str, Any]:
        """Search using image as query"""
        
        start_time = time.time()
        
        try:
            # Perform visual search
            search_results = await self.multimodal_retriever.visual_search(image_data, top_k)
            
            # Format results
            formatted_results = []
            for document, similarity in search_results:
                result_item = {
                    'document_id': document.doc_id,
                    'title': document.title,
                    'visual_similarity': similarity,
                    'visual_descriptions': document.visual_descriptions,
                    'detected_objects': [obj.get('label', 'object') for obj in document.detected_objects[:5]],
                    'has_charts': len(document.chart_data) > 0,
                    'content_preview': document.get_content_summary()[:100]
                }
                formatted_results.append(result_item)
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'search_type': 'visual_similarity',
                'results_found': len(formatted_results),
                'results': formatted_results,
                'processing_time': processing_time
            }
            
            self.logger.info(f"Visual search completed: {len(formatted_results)} results in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Visual search failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def analyze_document_content(self, document_id: str) -> Dict[str, Any]:
        """Analyze multimodal content of specific document"""
        
        if document_id not in self.multimodal_retriever.documents:
            return {
                'success': False,
                'error': 'Document not found'
            }
        
        document = self.multimodal_retriever.documents[document_id]
        
        analysis = {
            'success': True,
            'document_id': document_id,
            'title': document.title,
            'modalities_analysis': {
                'text': {
                    'has_text': bool(document.text_content),
                    'extracted_text_length': len(document.extracted_text),
                    'text_preview': (document.text_content or document.extracted_text)[:200]
                },
                'visual': {
                    'image_count': len(document.image_content),
                    'objects_detected': len(document.detected_objects),
                    'object_types': list(set(obj.get('label', 'unknown') for obj in document.detected_objects)),
                    'visual_descriptions': document.visual_descriptions,
                    'chart_count': len(document.chart_data)
                },
                'audio': {
                    'audio_count': len(document.audio_content),
                    'has_transcription': bool(document.extracted_text)
                }
            },
            'embeddings_available': list(document.embeddings.keys()),
            'processing_status': document.processing_status.value,
            'confidence_scores': document.confidence_scores
        }
        
        return analysis
    
    def _update_processing_stats(self, processing_time: float) -> None:
        """Update processing time statistics"""
        
        docs_processed = self.system_stats['documents_processed']
        current_avg = self.system_stats['average_processing_time']
        
        self.system_stats['average_processing_time'] = (
            (current_avg * (docs_processed - 1) + processing_time) / docs_processed
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            'system_stats': self.system_stats,
            'document_count': len(self.multimodal_retriever.documents),
            'modality_support': {
                'text': True,
                'image': True,
                'audio': True,
                'video': False,  # Not implemented in demo
                'cross_modal': True
            },
            'processing_capabilities': {
                'ocr': True,
                'object_detection': True,
                'image_captioning': True,
                'speech_to_text': True,
                'chart_analysis': True,
                'cross_modal_embedding': True
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_vision_processing():
    """Demo: Image processing and analysis"""
    print("\nDEMO 1: VISION PROCESSING")
    print("=" * 50)
    
    processor = VisionProcessor()
    
    # Test different types of images
    test_images = [
        {
            'type': 'chart',
            'description': 'Business chart with revenue data',
            'width': 800,
            'height': 600,
            'format': 'png'
        },
        {
            'type': 'document',
            'description': 'Scanned document with text',
            'width': 1200,
            'height': 800,
            'format': 'jpg'
        },
        {
            'type': 'photo',
            'description': 'Office environment photo',
            'width': 1024,
            'height': 768,
            'format': 'jpg'
        }
    ]
    
    print("Processing different types of images:")
    
    for i, image_data in enumerate(test_images, 1):
        print(f"\n--- Image {i}: {image_data['type']} ---")
        print(f"Description: {image_data['description']}")
        
        result = await processor.process_image(image_data)
        
        if 'error' not in result:
            print(f"Objects detected: {len(result['detected_objects'])}")
            for obj in result['detected_objects'][:3]:
                print(f"  - {obj['label']}: {obj['confidence']:.2f}")
            
            print(f"Extracted text: {result['extracted_text'] or 'None'}")
            print(f"Image description: {result['description']}")
            
            if result['chart_analysis']['contains_chart']:
                chart = result['chart_analysis']
                print(f"Chart detected: {chart['chart_type']} ({chart['confidence']:.2f})")
            
            print(f"Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"Processing failed: {result['error']}")

async def demo_audio_processing():
    """Demo: Audio processing and analysis"""
    print("\nDEMO 2: AUDIO PROCESSING")
    print("=" * 50)
    
    processor = AudioProcessor()
    
    # Test different types of audio
    test_audio = [
        {
            'type': 'speech',
            'description': 'Business meeting recording',
            'duration': 180.0,
            'format': 'mp3',
            'channels': 1
        },
        {
            'type': 'presentation',
            'description': 'Conference presentation',
            'duration': 300.0,
            'format': 'wav',
            'channels': 2
        },
        {
            'type': 'interview',
            'description': 'Customer interview',
            'duration': 120.0,
            'format': 'flac',
            'channels': 1
        }
    ]
    
    print("Processing different types of audio:")
    
    for i, audio_data in enumerate(test_audio, 1):
        print(f"\n--- Audio {i}: {audio_data['type']} ---")
        print(f"Description: {audio_data['description']}")
        print(f"Duration: {audio_data['duration']}s")
        
        result = await processor.process_audio(audio_data)
        
        if 'error' not in result:
            print(f"Transcribed text: {result['transcribed_text'][:100]}...")
            
            classification = result['classification']
            print(f"Classification: {classification['type']} ({classification['confidence']:.2f})")
            
            speaker_info = result['speaker_info']
            print(f"Speakers detected: {speaker_info['num_speakers']}")
            
            print(f"Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"Processing failed: {result['error']}")

async def demo_multimodal_documents():
    """Demo: Creating and processing multimodal documents"""
    print("\nDEMO 3: MULTIMODAL DOCUMENTS")
    print("=" * 50)
    
    rag_system = MultimodalRAGSystem()
    await rag_system.initialize()
    
    # Create sample multimodal documents
    documents = [
        MultimodalDocument(
            doc_id="business_report",
            title="Q3 Business Performance Report",
            text_content="Our Q3 performance shows strong growth across all sectors with revenue up 25% year-over-year.",
            image_content=[
                {
                    'type': 'chart',
                    'description': 'Revenue growth chart showing quarterly progression',
                    'format': 'png'
                },
                {
                    'type': 'chart',
                    'description': 'Market share analysis pie chart',
                    'format': 'jpg'
                }
            ]
        ),
        MultimodalDocument(
            doc_id="product_demo",
            title="Product Demonstration Video",
            text_content="Complete walkthrough of our new software platform features and capabilities.",
            image_content=[
                {
                    'type': 'document',
                    'description': 'Software interface screenshot',
                    'format': 'png'
                }
            ],
            audio_content=[
                {
                    'type': 'presentation',
                    'description': 'Narrator explaining product features',
                    'duration': 240.0,
                    'format': 'mp3'
                }
            ]
        ),
        MultimodalDocument(
            doc_id="technical_manual",
            title="System Architecture Documentation",
            text_content="Technical overview of our distributed system architecture and implementation details.",
            image_content=[
                {
                    'type': 'diagram',
                    'description': 'System architecture flowchart diagram',
                    'format': 'svg'
                },
                {
                    'type': 'chart',
                    'description': 'Performance metrics graphs',
                    'format': 'png'
                }
            ]
        )
    ]
    
    print("Processing multimodal documents:")
    
    for i, document in enumerate(documents, 1):
        print(f"\n--- Document {i}: {document.title} ---")
        print(f"Modalities: {[m.value for m in document.get_available_modalities()]}")
        
        result = await rag_system.add_document(document)
        
        if result['success']:
            print(f"✓ Processing successful")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Images processed: {result['images_processed']}")
            print(f"  Audio processed: {result['audio_processed']}")
            print(f"  Extracted text length: {result['extracted_text_length']} chars")
            print(f"  Objects detected: {result['objects_detected']}")
            print(f"  Visual descriptions: {result['visual_descriptions']}")
        else:
            print(f"✗ Processing failed: {result['error']}")

async def demo_multimodal_search():
    """Demo: Multimodal search capabilities"""
    print("\nDEMO 4: MULTIMODAL SEARCH")
    print("=" * 50)
    
    rag_system = MultimodalRAGSystem()
    await rag_system.initialize()
    
    # Add some documents first (simplified for demo)
    sample_docs = [
        MultimodalDocument(
            doc_id="financial_analysis",
            title="Financial Analysis Report",
            text_content="Comprehensive financial analysis with market trends and growth projections for technology sector.",
            image_content=[{'type': 'chart', 'description': 'Financial performance charts'}]
        ),
        MultimodalDocument(
            doc_id="product_showcase",
            title="Product Feature Showcase", 
            text_content="Detailed presentation of new product features and user interface improvements.",
            image_content=[{'type': 'document', 'description': 'Product interface screenshots'}],
            audio_content=[{'type': 'presentation', 'description': 'Product demo narration'}]
        ),
        MultimodalDocument(
            doc_id="tech_architecture",
            title="Technical Architecture Guide",
            text_content="System architecture documentation with detailed diagrams and performance metrics.",
            image_content=[{'type': 'diagram', 'description': 'Architecture flowcharts'}]
        )
    ]
    
    # Process documents
    print("Adding sample documents...")
    for doc in sample_docs:
        await rag_system.add_document(doc)
    
    # Test different types of multimodal queries
    test_queries = [
        MultimodalQuery(
            query_id="text_only",
            text_query="financial analysis charts and market trends",
            intent="search"
        ),
        MultimodalQuery(
            query_id="text_visual",
            text_query="system architecture diagrams",
            image_query={'type': 'diagram', 'description': 'technical diagram'},
            intent="analyze"
        ),
        MultimodalQuery(
            query_id="mixed_media",
            text_query="product demonstration with audio and visuals",
            preferred_modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO],
            response_format="mixed"
        )
    ]
    
    print(f"\nTesting multimodal search queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query.query_id} ---")
        print(f"Text query: {query.text_query}")
        print(f"Query modalities: {[m.value for m in query.get_query_modalities()]}")
        print(f"Intent: {query.intent}")
        
        result = await rag_system.multimodal_search(query, top_k=3)
        
        if result['success']:
            print(f"Results found: {result['results_found']}")
            print(f"Cross-modal search: {result['is_cross_modal']}")
            print(f"Processing time: {result['processing_time']:.3f}s")
            
            for j, doc_result in enumerate(result['results'], 1):
                print(f"  {j}. {doc_result['title']}")
                print(f"     Relevance: {doc_result['relevance_score']:.3f}")
                print(f"     Modalities: {doc_result['available_modalities']}")
                print(f"     Has images: {doc_result['has_images']}, Has audio: {doc_result['has_audio']}")
        else:
            print(f"Search failed: {result['error']}")

async def demo_visual_search():
    """Demo: Visual similarity search"""
    print("\nDEMO 5: VISUAL SEARCH")
    print("=" * 50)
    
    rag_system = MultimodalRAGSystem()
    await rag_system.initialize()
    
    # Add documents with visual content
    visual_docs = [
        MultimodalDocument(
            doc_id="chart_collection",
            title="Business Charts Collection",
            text_content="Collection of business performance charts and graphs",
            image_content=[
                {'type': 'chart', 'description': 'Revenue growth bar chart'},
                {'type': 'chart', 'description': 'Market share pie chart'}
            ]
        ),
        MultimodalDocument(
            doc_id="diagram_library",
            title="Technical Diagrams Library",
            text_content="Technical architecture and workflow diagrams",
            image_content=[
                {'type': 'diagram', 'description': 'System architecture flowchart'},
                {'type': 'diagram', 'description': 'Data flow diagram'}
            ]
        ),
        MultimodalDocument(
            doc_id="photo_gallery",
            title="Corporate Photo Gallery", 
            text_content="Office photos and team meeting pictures",
            image_content=[
                {'type': 'photo', 'description': 'Office environment photo'},
                {'type': 'photo', 'description': 'Team meeting photo'}
            ]
        )
    ]
    
    print("Adding documents with visual content...")
    for doc in visual_docs:
        await rag_system.add_document(doc)
    
    # Test visual search with different image types
    test_images = [
        {
            'type': 'chart',
            'description': 'Business performance chart',
            'format': 'png',
            'query_description': 'Finding similar charts'
        },
        {
            'type': 'diagram',
            'description': 'Technical architecture diagram',
            'format': 'svg',
            'query_description': 'Finding similar diagrams'
        },
        {
            'type': 'photo',
            'description': 'Office workspace photo',
            'format': 'jpg',
            'query_description': 'Finding similar photos'
        }
    ]
    
    print(f"\nTesting visual similarity search:")
    
    for i, image_query in enumerate(test_images, 1):
        print(f"\n--- Visual Search {i}: {image_query['query_description']} ---")
        print(f"Query image type: {image_query['type']}")
        print(f"Query description: {image_query['description']}")
        
        result = await rag_system.visual_search_by_image(image_query, top_k=3)
        
        if result['success']:
            print(f"Visually similar documents found: {result['results_found']}")
            print(f"Processing time: {result['processing_time']:.3f}s")
            
            for j, doc_result in enumerate(result['results'], 1):
                print(f"  {j}. {doc_result['title']}")
                print(f"     Visual similarity: {doc_result['visual_similarity']:.3f}")
                print(f"     Detected objects: {doc_result['detected_objects'][:3]}")
                print(f"     Has charts: {doc_result['has_charts']}")
        else:
            print(f"Visual search failed: {result['error']}")

async def demo_system_analytics():
    """Demo: Multimodal system analytics"""
    print("\nDEMO 6: SYSTEM ANALYTICS")
    print("=" * 50)
    
    rag_system = MultimodalRAGSystem()
    await rag_system.initialize()
    
    # Create comprehensive test dataset
    test_documents = []
    
    # Mixed content documents
    for i in range(5):
        doc = MultimodalDocument(
            doc_id=f"mixed_doc_{i}",
            title=f"Mixed Media Document {i+1}",
            text_content=f"Document {i+1} contains comprehensive analysis with visual and audio elements.",
            image_content=[
                {'type': 'chart', 'description': f'Chart {i+1}'},
                {'type': 'diagram', 'description': f'Diagram {i+1}'}
            ],
            audio_content=[
                {'type': 'presentation', 'description': f'Audio {i+1}', 'duration': 120.0}
            ] if i % 2 == 0 else []  # Every other document has audio
        )
        test_documents.append(doc)
    
    print("Processing comprehensive multimodal dataset...")
    
    # Process all documents
    processing_results = []
    for doc in test_documents:
        result = await rag_system.add_document(doc)
        processing_results.append(result)
        print(f"  ✓ Processed: {doc.title}")
    
    # Run multiple search queries
    search_queries = [
        MultimodalQuery(
            query_id=f"query_{i}",
            text_query=f"analysis charts data visualization document {i}",
            intent="search"
        ) for i in range(3)
    ]
    
    # Add cross-modal queries
    cross_modal_queries = [
        MultimodalQuery(
            query_id="cross_modal_1",
            text_query="comprehensive analysis with visual elements",
            image_query={'type': 'chart', 'description': 'data chart'},
            intent="analyze"
        ),
        MultimodalQuery(
            query_id="cross_modal_2",
            text_query="technical documentation with diagrams",
            image_query={'type': 'diagram', 'description': 'technical diagram'},
            intent="explain"
        )
    ]
    
    all_queries = search_queries + cross_modal_queries
    
    print(f"\nProcessing {len(all_queries)} search queries...")
    
    search_results = []
    for query in all_queries:
        result = await rag_system.multimodal_search(query)
        search_results.append(result)
    
    # Perform visual searches
    visual_queries = [
        {'type': 'chart', 'description': 'business chart'},
        {'type': 'diagram', 'description': 'technical diagram'}
    ]
    
    visual_results = []
    for visual_query in visual_queries:
        result = await rag_system.visual_search_by_image(visual_query)
        visual_results.append(result)
    
    # Get comprehensive analytics
    stats = rag_system.get_system_statistics()
    
    print(f"\nMULTIMODAL RAG SYSTEM ANALYTICS")
    print("=" * 40)
    
    print(f"\nProcessing Statistics:")
    system_stats = stats['system_stats']
    print(f"  Documents processed: {system_stats['documents_processed']}")
    print(f"  Images processed: {system_stats['images_processed']}")
    print(f"  Audio files processed: {system_stats['audio_files_processed']}")
    print(f"  Queries processed: {system_stats['queries_processed']}")
    print(f"  Cross-modal searches: {system_stats['cross_modal_searches']}")
    print(f"  Average processing time: {system_stats['average_processing_time']:.3f}s")
    
    print(f"\nSystem Capabilities:")
    capabilities = stats['processing_capabilities']
    for capability, enabled in capabilities.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {capability.upper()}")
    
    print(f"\nModality Support:")
    modality_support = stats['modality_support']
    for modality, supported in modality_support.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {modality.title()}")
    
    print(f"\nSearch Performance Analysis:")
    successful_searches = [r for r in search_results if r['success']]
    if successful_searches:
        avg_search_time = sum(r['processing_time'] for r in successful_searches) / len(successful_searches)
        avg_results_found = sum(r['results_found'] for r in successful_searches) / len(successful_searches)
        cross_modal_count = sum(1 for r in successful_searches if r['is_cross_modal'])
        
        print(f"  Search success rate: {len(successful_searches)}/{len(search_results)} ({len(successful_searches)/len(search_results)*100:.1f}%)")
        print(f"  Average search time: {avg_search_time:.3f}s")
        print(f"  Average results found: {avg_results_found:.1f}")
        print(f"  Cross-modal searches: {cross_modal_count}/{len(successful_searches)} ({cross_modal_count/len(successful_searches)*100:.1f}%)")
    
    print(f"\nDocument Analysis:")
    successful_processing = [r for r in processing_results if r['success']]
    if successful_processing:
        avg_processing_time = sum(r['processing_time'] for r in successful_processing) / len(successful_processing)
        total_images = sum(r['images_processed'] for r in successful_processing)
        total_audio = sum(r['audio_processed'] for r in successful_processing)
        
        print(f"  Processing success rate: {len(successful_processing)}/{len(processing_results)} ({len(successful_processing)/len(processing_results)*100:.1f}%)")
        print(f"  Average document processing time: {avg_processing_time:.3f}s")
        print(f"  Total images processed: {total_images}")
        print(f"  Total audio files processed: {total_audio}")

async def main():
    """
    Demonstrate Multimodal RAG Systems for cross-modal information processing
    
    WHAT YOU'LL LEARN:
    ================
    1. How to process and understand images, audio, and mixed media content
    2. How to create unified embeddings across different modalities
    3. How to implement cross-modal search and retrieval
    4. How to build systems that work with real-world multimedia content
    5. How to create next-generation AI that understands the complete picture
    
    REAL WORLD APPLICATIONS:
    =======================
    - Medical diagnosis with image analysis and patient records
    - Educational content with multimedia learning materials
    - Business intelligence with charts, reports, and presentations
    - Legal document analysis with scanned documents and audio transcripts
    - E-commerce with product images, descriptions, and videos
    - Scientific research with data visualizations and lab recordings
    """
    
    print("MULTIMODAL RAG SYSTEMS DEMONSTRATION")
    print("Building AI systems that understand text, images, audio, and more!")
    
    await demo_vision_processing()
    await demo_audio_processing()
    await demo_multimodal_documents()
    await demo_multimodal_search()
    await demo_visual_search()
    await demo_system_analytics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Vision processing enables image understanding and analysis")
    print("✓ Audio processing provides speech-to-text and audio classification")
    print("✓ Cross-modal embeddings unify different types of content")
    print("✓ Multimodal search finds relevant content across all media types")
    print("✓ Visual search enables image-to-image similarity matching")
    print("✓ Complete systems handle real-world multimedia content effectively")
    print("\nTHE POWER OF MULTIMODAL RAG:")
    print("- Enables AI to work with complete real-world information")
    print("- Provides visual understanding and analysis capabilities")
    print("- Supports complex documents with mixed content types")
    print("- Powers next-generation search and knowledge systems")

if __name__ == "__main__":
    asyncio.run(main())
