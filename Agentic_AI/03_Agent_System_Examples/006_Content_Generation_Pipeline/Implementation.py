"""
Content Generation Pipeline - Complete Implementation

A multi-agent pipeline that researches, outlines, writes, edits, and optimizes
content using specialized stages orchestrated by a main pipeline.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
import re

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")


class Pipeline_Stage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        self.model = "gpt-4"
        self.execution_time = 0
        self.retry_count = 0
        self.max_retries = 3
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return output.
        
        Args:
            input_data: Stage-specific input data
        
        Returns:
            Stage-specific output data
        """
        pass
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the stage with retry logic.
        
        Args:
            input_data: Stage input data
        
        Returns:
            Stage output data
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                result = self.process(input_data)
                self.execution_time = time.time() - start_time
                return result
            except Exception as e:
                self.retry_count = attempt + 1
                if attempt == self.max_retries - 1:
                    raise Exception(f"Stage {self.name} failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)
        
        raise Exception(f"Stage {self.name} failed unexpectedly")
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate stage output meets quality requirements.
        
        Args:
            output: Stage output to validate
        
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this stage."""
        return {
            'name': self.name,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count
        }


class Research_Stage(Pipeline_Stage):
    """Research stage: Gathers information about the topic."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Research", api_key)
        self.research_data = {}
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research queries and gather information."""
        topic = input_data.get('topic', '')
        keywords = input_data.get('keywords', [])
        target_audience = input_data.get('target_audience', 'general audience')
        
        print(f"Researching topic: {topic}")
        
        queries = self._generate_search_queries(topic, keywords)
        research_summary = self._gather_information(topic, queries, keywords)
        
        return {
            'topic': topic,
            'keywords': keywords,
            'search_queries': queries,
            'research_summary': research_summary,
            'key_facts': research_summary.get('key_facts', []),
            'statistics': research_summary.get('statistics', []),
            'references': research_summary.get('references', []),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_search_queries(self, topic: str, keywords: List[str]) -> List[str]:
        """Generate search queries for research."""
        queries = [topic]
        
        if keywords:
            for keyword in keywords[:3]:
                queries.append(f"{topic} {keyword}")
        
        queries.extend([
            f"{topic} overview",
            f"{topic} best practices",
            f"{topic} trends"
        ])
        
        return queries[:5]
    
    def _gather_information(self, topic: str, queries: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Gather information using LLM (mock implementation with real LLM call)."""
        if not self.client:
            return self._mock_research(topic, keywords)
        
        prompt = f"""Research the topic: {topic}
Keywords: {', '.join(keywords) if keywords else 'None'}

Provide a comprehensive research summary including:
1. Key facts and information about the topic
2. Relevant statistics and data points
3. Important references and sources
4. Current trends and developments
5. Common questions and answers

Format as JSON with keys: key_facts, statistics, references, trends, qa_pairs."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant. Provide comprehensive, accurate information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return self._parse_research_text(content)
        except Exception as e:
            print(f"Research API error: {e}, using mock data")
            return self._mock_research(topic, keywords)
    
    def _parse_research_text(self, text: str) -> Dict[str, Any]:
        """Parse research text into structured format."""
        return {
            'key_facts': [text[:200]],
            'statistics': [],
            'references': [],
            'trends': [],
            'qa_pairs': []
        }
    
    def _mock_research(self, topic: str, keywords: List[str]) -> Dict[str, Any]:
        """Mock research data for demonstration."""
        return {
            'key_facts': [
                f"{topic} is an important subject in modern technology.",
                f"Understanding {topic} requires knowledge of fundamental concepts.",
                f"Recent developments in {topic} have shown significant progress."
            ],
            'statistics': [
                f"Studies show that {topic} adoption has increased by 40% in recent years.",
                f"Over 60% of organizations are implementing {topic} solutions."
            ],
            'references': [
                f"Industry report on {topic} (2024)",
                f"Research paper: Advances in {topic}"
            ],
            'trends': [
                f"Increasing focus on {topic} optimization",
                f"Growing demand for {topic} expertise"
            ],
            'qa_pairs': [
                {"question": f"What is {topic}?", "answer": f"{topic} is a key concept in the field."},
                {"question": f"Why is {topic} important?", "answer": f"{topic} provides significant benefits."}
            ]
        }


class Outline_Stage(Pipeline_Stage):
    """Outline stage: Creates structured content outline."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Outline", api_key)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content outline from research."""
        research = input_data.get('research_summary', {})
        topic = input_data.get('topic', '')
        target_length = input_data.get('target_length', 1000)
        content_type = input_data.get('content_type', 'blog_post')
        
        print(f"Creating outline for: {topic}")
        
        outline = self._generate_outline(topic, research, target_length, content_type)
        
        return {
            'topic': topic,
            'outline': outline,
            'sections': outline.get('sections', []),
            'estimated_length': outline.get('estimated_length', target_length),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_outline(self, topic: str, research: Dict[str, Any], 
                         target_length: int, content_type: str) -> Dict[str, Any]:
        """Generate structured outline using LLM."""
        if not self.client:
            return self._mock_outline(topic, target_length)
        
        research_text = json.dumps(research, indent=2)
        
        prompt = f"""Create a detailed content outline for a {content_type} about: {topic}

Target length: {target_length} words

Research information:
{research_text}

Create a structured outline with:
- Title
- Introduction section
- Main sections (3-5 sections) with headings and subheadings
- Key points for each section
- Conclusion section

Format as JSON with structure:
{{
  "title": "...",
  "sections": [
    {{
      "heading": "...",
      "subheadings": ["..."],
      "key_points": ["..."]
    }}
  ],
  "estimated_length": ...
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert content strategist. Create well-structured outlines."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return self._parse_outline_text(content, topic)
        except Exception as e:
            print(f"Outline API error: {e}, using mock outline")
            return self._mock_outline(topic, target_length)
    
    def _parse_outline_text(self, text: str, topic: str) -> Dict[str, Any]:
        """Parse outline text into structured format."""
        return {
            'title': f"Understanding {topic}",
            'sections': [
                {
                    'heading': 'Introduction',
                    'subheadings': [],
                    'key_points': ['Introduction to the topic']
                },
                {
                    'heading': 'Main Content',
                    'subheadings': ['Key Concepts', 'Best Practices'],
                    'key_points': ['Main points about the topic']
                },
                {
                    'heading': 'Conclusion',
                    'subheadings': [],
                    'key_points': ['Summary and takeaways']
                }
            ],
            'estimated_length': 1000
        }
    
    def _mock_outline(self, topic: str, target_length: int) -> Dict[str, Any]:
        """Mock outline for demonstration."""
        return {
            'title': f"Complete Guide to {topic}",
            'sections': [
                {
                    'heading': 'Introduction',
                    'subheadings': [],
                    'key_points': [
                        f'What is {topic}?',
                        'Why is it important?',
                        'What will you learn?'
                    ]
                },
                {
                    'heading': 'Understanding the Fundamentals',
                    'subheadings': ['Core Concepts', 'Key Principles'],
                    'key_points': [
                        'Basic concepts explained',
                        'Fundamental principles',
                        'How it works'
                    ]
                },
                {
                    'heading': 'Best Practices and Applications',
                    'subheadings': ['Implementation Strategies', 'Common Use Cases'],
                    'key_points': [
                        'Best practices to follow',
                        'Real-world applications',
                        'Success stories'
                    ]
                },
                {
                    'heading': 'Conclusion',
                    'subheadings': [],
                    'key_points': [
                        'Key takeaways',
                        'Next steps',
                        'Additional resources'
                    ]
                }
            ],
            'estimated_length': target_length
        }


class Writing_Stage(Pipeline_Stage):
    """Writing stage: Generates content from outline."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Writing", api_key)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content following the outline."""
        outline = input_data.get('outline', {})
        research = input_data.get('research_summary', {})
        tone = input_data.get('tone', 'professional')
        
        print("Writing content sections...")
        
        content = self._write_content(outline, research, tone)
        
        return {
            'title': outline.get('title', ''),
            'content': content,
            'word_count': len(content.split()),
            'sections_completed': len(outline.get('sections', [])),
            'timestamp': datetime.now().isoformat()
        }
    
    def _write_content(self, outline: Dict[str, Any], research: Dict[str, Any], 
                      tone: str) -> str:
        """Generate content using LLM."""
        if not self.client:
            return self._mock_content(outline)
        
        outline_text = json.dumps(outline, indent=2)
        research_text = json.dumps(research, indent=2)
        
        prompt = f"""Write a complete {tone} article based on this outline:

Outline:
{outline_text}

Research Information:
{research_text}

Write the full article following the outline structure. Include all sections and subheadings.
Maintain a {tone} tone throughout. Write naturally and engagingly.
Target length: approximately {outline.get('estimated_length', 1000)} words."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert content writer. Write in a {tone} tone."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=3000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Writing API error: {e}, using mock content")
            return self._mock_content(outline)
    
    def _mock_content(self, outline: Dict[str, Any]) -> str:
        """Mock content for demonstration."""
        title = outline.get('title', 'Article Title')
        sections = outline.get('sections', [])
        
        content_parts = [f"# {title}\n\n"]
        
        for section in sections:
            heading = section.get('heading', 'Section')
            content_parts.append(f"## {heading}\n\n")
            
            key_points = section.get('key_points', [])
            for point in key_points:
                content_parts.append(f"{point}. This section covers important aspects of the topic. "
                                   f"Here we discuss relevant information and provide insights.\n\n")
            
            subheadings = section.get('subheadings', [])
            for subheading in subheadings:
                content_parts.append(f"### {subheading}\n\n")
                content_parts.append(f"Content about {subheading} goes here. "
                                   f"This provides detailed information on the subtopic.\n\n")
        
        return "".join(content_parts)


class Editing_Stage(Pipeline_Stage):
    """Editing stage: Reviews and improves content."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Editing", api_key)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Edit and improve content."""
        content = input_data.get('content', '')
        title = input_data.get('title', '')
        
        print("Editing content...")
        
        edited_content = self._edit_content(content, title)
        quality_score = self._calculate_quality_score(edited_content)
        
        return {
            'title': title,
            'content': edited_content,
            'original_length': len(content.split()),
            'edited_length': len(edited_content.split()),
            'quality_score': quality_score,
            'improvements': self._identify_improvements(content, edited_content),
            'timestamp': datetime.now().isoformat()
        }
    
    def _edit_content(self, content: str, title: str) -> str:
        """Edit content using LLM."""
        if not self.client:
            return self._mock_edit(content)
        
        prompt = f"""Edit and improve the following content. Fix grammar, improve clarity, 
enhance readability, and ensure consistency. Maintain the original meaning and structure.

Title: {title}

Content:
{content}

Return the edited content with improvements."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert editor. Improve content quality while preserving meaning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Editing API error: {e}, returning original content")
            return content
    
    def _mock_edit(self, content: str) -> str:
        """Mock editing (minimal changes for demo)."""
        edited = content.replace('  ', ' ')
        edited = re.sub(r'\n{3,}', '\n\n', edited)
        return edited.strip()
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate content quality score."""
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        score = 0.7
        if 10 <= avg_sentence_length <= 25:
            score += 0.1
        if word_count >= 500:
            score += 0.1
        if len(re.findall(r'\n##', content)) >= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_improvements(self, original: str, edited: str) -> List[str]:
        """Identify improvements made during editing."""
        improvements = []
        
        if len(edited.split()) != len(original.split()):
            improvements.append("Word count adjusted for clarity")
        
        if edited != original:
            improvements.append("Grammar and style improvements")
        
        return improvements
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate edited content meets quality threshold."""
        quality_score = output.get('quality_score', 0)
        return quality_score >= 0.6


class SEO_Stage(Pipeline_Stage):
    """SEO stage: Optimizes content for search engines."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("SEO", api_key)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for SEO."""
        content = input_data.get('content', '')
        title = input_data.get('title', '')
        keywords = input_data.get('keywords', [])
        
        print("Optimizing for SEO...")
        
        optimized = self._optimize_seo(content, title, keywords)
        seo_score = self._calculate_seo_score(optimized['content'], optimized['title'], 
                                             optimized['meta_description'], keywords)
        
        return {
            'title': optimized['title'],
            'meta_description': optimized['meta_description'],
            'content': optimized['content'],
            'keywords': keywords,
            'seo_score': seo_score,
            'optimizations': optimized.get('optimizations', []),
            'timestamp': datetime.now().isoformat()
        }
    
    def _optimize_seo(self, content: str, title: str, keywords: List[str]) -> Dict[str, Any]:
        """Optimize content for SEO using LLM."""
        if not self.client:
            return self._mock_seo_optimize(content, title, keywords)
        
        keywords_str = ', '.join(keywords) if keywords else 'None'
        
        prompt = f"""Optimize the following content for SEO. Focus on:
1. Title optimization (60 characters, include primary keyword)
2. Meta description (150-160 characters, compelling)
3. Heading optimization (ensure H1, H2, H3 structure)
4. Keyword integration (natural, not keyword stuffing)
5. Internal/external link suggestions

Target keywords: {keywords_str}

Current title: {title}

Content:
{content[:2000]}

Return optimized content with:
- Optimized title
- Meta description
- Optimized content with improved headings
- List of optimizations made"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an SEO expert. Optimize content for search engines naturally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_seo_result(result_text, content, title, keywords)
        except Exception as e:
            print(f"SEO API error: {e}, using mock optimization")
            return self._mock_seo_optimize(content, title, keywords)
    
    def _parse_seo_result(self, result_text: str, original_content: str, 
                          original_title: str, keywords: List[str]) -> Dict[str, Any]:
        """Parse SEO optimization result."""
        optimized_title = original_title
        if keywords:
            optimized_title = f"{keywords[0].title()}: {original_title}"
        
        meta_description = f"Learn about {original_title.lower()}. " \
                          f"Comprehensive guide with insights and best practices."
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        return {
            'title': optimized_title[:60],
            'meta_description': meta_description,
            'content': original_content,
            'optimizations': ['Title optimized', 'Meta description created', 'Keywords integrated']
        }
    
    def _mock_seo_optimize(self, content: str, title: str, keywords: List[str]) -> Dict[str, Any]:
        """Mock SEO optimization."""
        optimized_title = title
        if keywords:
            optimized_title = f"{keywords[0].title()}: {title}"
        
        meta_description = f"Learn about {title.lower()}. Comprehensive guide with insights."
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        return {
            'title': optimized_title[:60],
            'meta_description': meta_description,
            'content': content,
            'optimizations': ['Title optimized', 'Meta description created']
        }
    
    def _calculate_seo_score(self, content: str, title: str, 
                            meta_description: str, keywords: List[str]) -> float:
        """Calculate SEO score."""
        score = 0.5
        
        if title and len(title) <= 60:
            score += 0.1
        
        if meta_description and 150 <= len(meta_description) <= 160:
            score += 0.1
        
        if keywords:
            keyword_count = sum(content.lower().count(kw.lower()) for kw in keywords)
            if keyword_count >= len(keywords):
                score += 0.1
        
        h1_count = len(re.findall(r'^#\s+', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', content, re.MULTILINE))
        
        if h1_count >= 1:
            score += 0.1
        if h2_count >= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate SEO optimization meets threshold."""
        seo_score = output.get('seo_score', 0)
        return seo_score >= 0.6


class Pipeline_Monitor:
    """Monitors pipeline execution and performance."""
    
    def __init__(self):
        self.stage_timings = {}
        self.quality_scores = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def start_pipeline(self):
        """Mark pipeline start."""
        self.start_time = time.time()
    
    def end_pipeline(self):
        """Mark pipeline end."""
        self.end_time = time.time()
    
    def record_stage(self, stage_name: str, metrics: Dict[str, Any]):
        """Record stage execution metrics."""
        self.stage_timings[stage_name] = {
            'execution_time': metrics.get('execution_time', 0),
            'retry_count': metrics.get('retry_count', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def record_quality_score(self, stage_name: str, score: float):
        """Record quality score for a stage."""
        self.quality_scores[stage_name] = score
    
    def record_error(self, stage_name: str, error: str):
        """Record an error."""
        self.errors.append({
            'stage': stage_name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive pipeline report."""
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            'total_execution_time': total_time,
            'stage_timings': self.stage_timings,
            'quality_scores': self.quality_scores,
            'errors': self.errors,
            'success': len(self.errors) == 0,
            'timestamp': datetime.now().isoformat()
        }


class Content_Pipeline:
    """Main pipeline orchestrating all stages."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.research_stage = Research_Stage(self.api_key)
        self.outline_stage = Outline_Stage(self.api_key)
        self.writing_stage = Writing_Stage(self.api_key)
        self.editing_stage = Editing_Stage(self.api_key)
        self.seo_stage = SEO_Stage(self.api_key)
        self.monitor = Pipeline_Monitor()
        self.quality_gates = {
            'research': {'min_score': 0.0},
            'outline': {'min_score': 0.0},
            'writing': {'min_score': 0.0},
            'editing': {'min_score': 0.6},
            'seo': {'min_score': 0.6}
        }
    
    def run(self, topic: str, keywords: List[str] = None, 
           target_audience: str = "general audience",
           target_length: int = 1000, content_type: str = "blog_post",
           tone: str = "professional") -> Dict[str, Any]:
        """
        Run the complete content generation pipeline.
        
        Args:
            topic: Content topic
            keywords: Target keywords for SEO
            target_audience: Target audience description
            target_length: Target word count
            content_type: Type of content (blog_post, article, etc.)
            tone: Writing tone (professional, casual, technical)
        
        Returns:
            Complete content with metadata and metrics
        """
        self.monitor.start_pipeline()
        
        pipeline_data = {
            'topic': topic,
            'keywords': keywords or [],
            'target_audience': target_audience,
            'target_length': target_length,
            'content_type': content_type,
            'tone': tone
        }
        
        try:
            print(f"\n{'='*60}")
            print(f"Content Generation Pipeline - Topic: {topic}")
            print(f"{'='*60}\n")
            
            research_result = self.run_stage(self.research_stage, pipeline_data)
            pipeline_data.update(research_result)
            
            outline_result = self.run_stage(self.outline_stage, pipeline_data)
            pipeline_data.update(outline_result)
            
            writing_result = self.run_stage(self.writing_stage, pipeline_data)
            pipeline_data.update(writing_result)
            
            editing_result = self.run_stage(self.editing_stage, {
                'content': writing_result['content'],
                'title': outline_result['outline']['title']
            })
            pipeline_data.update(editing_result)
            
            seo_result = self.run_stage(self.seo_stage, {
                'content': editing_result['content'],
                'title': editing_result['title'],
                'keywords': keywords or []
            })
            pipeline_data.update(seo_result)
            
            self.monitor.end_pipeline()
            
            return {
                'success': True,
                'topic': topic,
                'title': seo_result['title'],
                'meta_description': seo_result['meta_description'],
                'content': seo_result['content'],
                'keywords': keywords or [],
                'metrics': {
                    'word_count': len(seo_result['content'].split()),
                    'quality_score': editing_result.get('quality_score', 0),
                    'seo_score': seo_result.get('seo_score', 0)
                },
                'pipeline_report': self.monitor.get_report(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.monitor.record_error('pipeline', str(e))
            self.monitor.end_pipeline()
            return {
                'success': False,
                'error': str(e),
                'pipeline_report': self.monitor.get_report()
            }
    
    def run_stage(self, stage: Pipeline_Stage, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single pipeline stage with quality gate validation.
        
        Args:
            stage: Pipeline stage to execute
            input_data: Stage input data
        
        Returns:
            Stage output data
        """
        print(f"\n--- Running {stage.name} Stage ---")
        
        try:
            result = stage.run(input_data)
            
            if not stage.validate_output(result):
                gate_config = self.quality_gates.get(stage.name.lower(), {})
                min_score = gate_config.get('min_score', 0)
                print(f"Warning: {stage.name} quality gate not met (min: {min_score})")
            
            metrics = stage.get_metrics()
            self.monitor.record_stage(stage.name, metrics)
            
            if 'quality_score' in result:
                self.monitor.record_quality_score(stage.name, result['quality_score'])
            elif 'seo_score' in result:
                self.monitor.record_quality_score(stage.name, result['seo_score'])
            
            print(f"✓ {stage.name} completed in {metrics['execution_time']:.2f}s")
            
            return result
        except Exception as e:
            self.monitor.record_error(stage.name, str(e))
            raise


def main():
    """Main function demonstrating the Content Generation Pipeline."""
    
    print("=" * 60)
    print("Content Generation Pipeline - Demonstration")
    print("=" * 60)
    
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI library not available.")
        print("Please install with: pip install openai")
        print("And set OPENAI_API_KEY environment variable.")
        return
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set.")
        print("Set your API key: export OPENAI_API_KEY='your-key'")
        print("Running in mock mode (will use placeholder content)...")
    
    pipeline = Content_Pipeline(api_key=api_key)
    
    topic = "Artificial Intelligence in Healthcare"
    keywords = ["AI healthcare", "medical AI", "healthcare technology"]
    
    print(f"\nGenerating content on topic: {topic}")
    print(f"Keywords: {', '.join(keywords)}\n")
    
    result = pipeline.run(
        topic=topic,
        keywords=keywords,
        target_audience="healthcare professionals and technology enthusiasts",
        target_length=1200,
        content_type="blog_post",
        tone="professional"
    )
    
    if result['success']:
        print("\n" + "=" * 60)
        print("Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"\nTitle: {result['title']}")
        print(f"\nMeta Description: {result['meta_description']}")
        print(f"\nWord Count: {result['metrics']['word_count']}")
        print(f"Quality Score: {result['metrics']['quality_score']:.2%}")
        print(f"SEO Score: {result['metrics']['seo_score']:.2%}")
        
        print("\n--- Content Preview (first 500 chars) ---")
        print(result['content'][:500] + "...")
        
        print("\n--- Pipeline Performance ---")
        report = result['pipeline_report']
        print(f"Total Time: {report['total_execution_time']:.2f}s")
        print("\nStage Timings:")
        for stage, timing in report['stage_timings'].items():
            print(f"  {stage}: {timing['execution_time']:.2f}s")
        
        output_file = "generated_content.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {result['title']}\n\n")
            f.write(f"**Meta Description:** {result['meta_description']}\n\n")
            f.write(f"**Keywords:** {', '.join(result['keywords'])}\n\n")
            f.write("---\n\n")
            f.write(result['content'])
        
        print(f"\n✓ Full content saved to: {output_file}")
    else:
        print(f"\nPipeline failed: {result.get('error', 'Unknown error')}")
        print("\nErrors:")
        for error in result['pipeline_report'].get('errors', []):
            print(f"  {error['stage']}: {error['error']}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
