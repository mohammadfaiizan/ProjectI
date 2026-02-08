"""
Agent module for Content Generation Pipeline.
Contains LangGraph-based content generation pipeline with state management.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from Tools import (
    Research_Topic,
    Check_Grammar,
    Analyze_SEO,
    Calculate_Readability,
    Content_Validator,
    SEO_Analyzer
)
from Config import Content_Config, SEO_Config, Quality_Config
import json


class Pipeline_State(TypedDict):
    """State schema for content generation pipeline."""
    topic: str
    audience: str
    tone: str
    research: Dict[str, Any]
    outline: str
    draft: str
    edited_draft: str
    seo_optimized: str
    quality_score: float
    revision_count: int
    final_content: str
    stage_history: List[str]
    target_keywords: List[str]
    target_word_count: int
    messages: Annotated[List[BaseMessage], add_messages]


class Content_Pipeline_Graph:
    """LangGraph-based content generation pipeline."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        content_config: Content_Config,
        seo_config: SEO_Config,
        quality_config: Quality_Config
    ):
        """
        Initialize content pipeline graph.
        
        Args:
            llm: Language model instance
            content_config: Content configuration
            seo_config: SEO configuration
            quality_config: Quality configuration
        """
        self.llm = llm
        self.content_config = content_config
        self.seo_config = seo_config
        self.quality_config = quality_config
        self.validator = Content_Validator(
            min_word_count=500,
            max_word_count=content_config.Get_Max_Word_Count()
        )
        self.seo_analyzer = SEO_Analyzer(
            min_keyword_density=seo_config.Get_Min_Keyword_Density()
        )
        self.graph = None
    
    def Build_Graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with nodes and edges.
        
        Returns:
            StateGraph instance
        """
        workflow = StateGraph(Pipeline_State)
        
        workflow.add_node("research", self._Research_Node)
        workflow.add_node("create_outline", self._Create_Outline_Node)
        workflow.add_node("write_draft", self._Write_Draft_Node)
        workflow.add_node("edit_draft", self._Edit_Draft_Node)
        workflow.add_node("quality_check", self._Quality_Check_Node)
        workflow.add_node("seo_optimize", self._SEO_Optimize_Node)
        workflow.add_node("finalize", self._Finalize_Node)
        
        workflow.set_entry_point("research")
        
        workflow.add_edge("research", "create_outline")
        workflow.add_edge("create_outline", "write_draft")
        workflow.add_edge("write_draft", "edit_draft")
        workflow.add_edge("edit_draft", "quality_check")
        
        workflow.add_conditional_edges(
            "quality_check",
            self._Should_Proceed_To_SEO,
            {
                "seo_optimize": "seo_optimize",
                "revise": "edit_draft",
                "accept": "seo_optimize"
            }
        )
        
        workflow.add_edge("seo_optimize", "finalize")
        workflow.add_edge("finalize", END)
        
        self.graph = workflow
        return workflow
    
    def _Research_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        Research node: Gather information about the topic.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with research data
        """
        topic = state["topic"]
        stage_history = state.get("stage_history", [])
        stage_history.append("Research")
        
        research_result = Research_Topic.invoke({"topic": topic})
        
        return {
            "research": research_result,
            "stage_history": stage_history
        }
    
    def _Create_Outline_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        Create outline node: Generate structured outline with sections.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with outline
        """
        topic = state["topic"]
        audience = state["audience"]
        tone = state["tone"]
        research = state["research"]
        target_word_count = state.get("target_word_count", 1500)
        stage_history = state.get("stage_history", [])
        stage_history.append("Create_Outline")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert content strategist. Create a detailed, structured outline "
                "for an article based on the research provided. The outline should include "
                "main sections, subsections, and key points to cover."
            )),
            ("human", (
                "Topic: {topic}\n"
                "Target Audience: {audience}\n"
                "Tone: {tone}\n"
                "Target Word Count: {target_word_count}\n\n"
                "Research Data:\n{research_data}\n\n"
                "Create a comprehensive outline with:\n"
                "- A compelling title\n"
                "- Introduction section\n"
                "- 3-5 main sections with subsections\n"
                "- Conclusion section\n"
                "Format the outline in markdown with proper heading hierarchy."
            ))
        ])
        
        research_str = json.dumps(research, indent=2)
        
        formatted_prompt = prompt.format_messages(
            topic=topic,
            audience=audience,
            tone=tone,
            target_word_count=target_word_count,
            research_data=research_str
        )
        
        response = self.llm.invoke(formatted_prompt)
        outline = response.content if hasattr(response, "content") else str(response)
        
        return {
            "outline": outline,
            "stage_history": stage_history
        }
    
    def _Write_Draft_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        Write draft node: Write content section by section from outline.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with draft content
        """
        topic = state["topic"]
        audience = state["audience"]
        tone = state["tone"]
        outline = state["outline"]
        research = state["research"]
        target_word_count = state.get("target_word_count", 1500)
        stage_history = state.get("stage_history", [])
        stage_history.append("Write_Draft")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert content writer. Write comprehensive, engaging content "
                "based on the provided outline and research. Ensure the content matches "
                "the specified audience level and tone. Write in markdown format with "
                "proper headings, paragraphs, and formatting."
            )),
            ("human", (
                "Topic: {topic}\n"
                "Target Audience: {audience}\n"
                "Tone: {tone}\n"
                "Target Word Count: {target_word_count}\n\n"
                "Outline:\n{outline}\n\n"
                "Research Data:\n{research_data}\n\n"
                "Write the full article following the outline. Include:\n"
                "- Engaging introduction\n"
                "- Detailed sections with examples and explanations\n"
                "- Clear conclusion\n"
                "Use markdown formatting with proper heading hierarchy."
            ))
        ])
        
        research_str = json.dumps(research, indent=2)
        
        formatted_prompt = prompt.format_messages(
            topic=topic,
            audience=audience,
            tone=tone,
            target_word_count=target_word_count,
            outline=outline,
            research_data=research_str
        )
        
        response = self.llm.invoke(formatted_prompt)
        draft = response.content if hasattr(response, "content") else str(response)
        
        return {
            "draft": draft,
            "stage_history": stage_history
        }
    
    def _Edit_Draft_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        Edit draft node: Review for clarity, grammar, coherence, and improve.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with edited draft
        """
        draft = state.get("draft", "")
        audience = state["audience"]
        tone = state["tone"]
        revision_count = state.get("revision_count", 0)
        stage_history = state.get("stage_history", [])
        
        if "Edit_Draft" not in stage_history:
            stage_history.append("Edit_Draft")
        
        grammar_result = Check_Grammar.invoke({"text": draft})
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert editor. Review and improve the provided content for "
                "clarity, grammar, coherence, and engagement. Maintain the original "
                "tone and audience level while enhancing readability and flow."
            )),
            ("human", (
                "Content to Edit:\n{draft}\n\n"
                "Target Audience: {audience}\n"
                "Tone: {tone}\n"
                "Grammar Issues Found: {grammar_issues}\n\n"
                "Please improve the content by:\n"
                "- Fixing grammar and spelling errors\n"
                "- Improving clarity and flow\n"
                "- Enhancing coherence between sections\n"
                "- Maintaining consistent tone\n"
                "- Ensuring engagement\n"
                "Return the improved content in markdown format."
            ))
        ])
        
        grammar_issues = json.dumps(grammar_result.get("errors", []), indent=2)
        
        formatted_prompt = prompt.format_messages(
            draft=draft,
            audience=audience,
            tone=tone,
            grammar_issues=grammar_issues
        )
        
        response = self.llm.invoke(formatted_prompt)
        edited_draft = response.content if hasattr(response, "content") else str(response)
        
        return {
            "edited_draft": edited_draft,
            "revision_count": revision_count + 1,
            "stage_history": stage_history
        }
    
    def _Quality_Check_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        Quality check node: Evaluate quality score (0-1).
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with quality score
        """
        edited_draft = state.get("edited_draft", state.get("draft", ""))
        stage_history = state.get("stage_history", [])
        stage_history.append("Quality_Check")
        
        quality_score = self.validator.Calculate_Quality_Score(edited_draft)
        
        return {
            "quality_score": quality_score,
            "stage_history": stage_history
        }
    
    def _Should_Proceed_To_SEO(
        self,
        state: Pipeline_State
    ) -> Literal["seo_optimize", "revise", "accept"]:
        """
        Conditional edge function: Determine next step based on quality score.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to proceed to
        """
        quality_score = state.get("quality_score", 0.0)
        revision_count = state.get("revision_count", 0)
        min_score = self.quality_config.Get_Min_Quality_Score()
        max_revisions = self.quality_config.Get_Max_Revision_Rounds()
        
        if quality_score >= min_score:
            return "seo_optimize"
        elif revision_count < max_revisions:
            return "revise"
        else:
            return "accept"
    
    def _SEO_Optimize_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        SEO optimize node: Optimize headings, add keywords, meta description.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with SEO-optimized content
        """
        edited_draft = state.get("edited_draft", state.get("draft", ""))
        target_keywords = state.get("target_keywords", [])
        topic = state["topic"]
        stage_history = state.get("stage_history", [])
        stage_history.append("SEO_Optimize")
        
        if not target_keywords:
            target_keywords = [topic]
        
        seo_analysis = self.seo_analyzer.Analyze(edited_draft, target_keywords)
        meta_tags = self.seo_analyzer.Suggest_Meta_Tags(edited_draft, target_keywords)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an SEO expert. Optimize the provided content for search engines "
                "by improving keyword usage, heading structure, and adding meta tags. "
                "Maintain content quality and readability while improving SEO."
            )),
            ("human", (
                "Content to Optimize:\n{content}\n\n"
                "Target Keywords: {keywords}\n"
                "SEO Analysis:\n{seo_analysis}\n"
                "Suggested Meta Tags:\n{meta_tags}\n\n"
                "Please optimize the content by:\n"
                "- Ensuring target keywords appear naturally throughout\n"
                "- Optimizing heading structure (H1, H2, H3)\n"
                "- Adding meta description at the top\n"
                "- Improving keyword density where appropriate\n"
                "- Maintaining content quality\n"
                "Return the optimized content in markdown format with meta tags."
            ))
        ])
        
        keywords_str = ", ".join(target_keywords)
        seo_analysis_str = json.dumps(seo_analysis, indent=2)
        meta_tags_str = json.dumps(meta_tags, indent=2)
        
        formatted_prompt = prompt.format_messages(
            content=edited_draft,
            keywords=keywords_str,
            seo_analysis=seo_analysis_str,
            meta_tags=meta_tags_str
        )
        
        response = self.llm.invoke(formatted_prompt)
        seo_optimized = response.content if hasattr(response, "content") else str(response)
        
        return {
            "seo_optimized": seo_optimized,
            "stage_history": stage_history
        }
    
    def _Finalize_Node(self, state: Pipeline_State) -> Dict[str, Any]:
        """
        Finalize node: Format final output with metadata.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with final content
        """
        seo_optimized = state.get("seo_optimized", "")
        quality_score = state.get("quality_score", 0.0)
        stage_history = state.get("stage_history", [])
        stage_history.append("Finalize")
        
        final_content = seo_optimized
        
        metadata = {
            "topic": state["topic"],
            "audience": state["audience"],
            "tone": state["tone"],
            "quality_score": quality_score,
            "revision_count": state.get("revision_count", 0),
            "stages_completed": stage_history
        }
        
        metadata_section = "\n\n---\n\n## Metadata\n\n"
        metadata_section += json.dumps(metadata, indent=2)
        
        final_content += metadata_section
        
        return {
            "final_content": final_content,
            "stage_history": stage_history
        }
    
    def Compile(self) -> Any:
        """
        Compile the graph for execution.
        
        Returns:
            Compiled graph
        """
        if self.graph is None:
            self.Build_Graph()
        
        return self.graph.compile()
    
    def Generate(
        self,
        topic: str,
        audience: str,
        tone: str,
        target_keywords: Optional[List[str]] = None,
        target_word_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate content using the pipeline.
        
        Args:
            topic: Content topic
            audience: Target audience level
            tone: Content tone
            target_keywords: Optional list of target keywords
            target_word_count: Optional target word count
            
        Returns:
            Dictionary containing generated content and metadata
        """
        if self.graph is None:
            self.Build_Graph()
        
        compiled_graph = self.Compile()
        
        if not self.content_config.Validate_Audience(audience):
            raise ValueError(f"Invalid audience: {audience}")
        
        if not self.content_config.Validate_Tone(tone):
            raise ValueError(f"Invalid tone: {tone}")
        
        if target_keywords is None:
            target_keywords = [topic]
        
        if target_word_count is None:
            target_word_count = self.content_config.Get_Max_Word_Count()
        
        initial_state = {
            "topic": topic,
            "audience": audience,
            "tone": tone,
            "research": {},
            "outline": "",
            "draft": "",
            "edited_draft": "",
            "seo_optimized": "",
            "quality_score": 0.0,
            "revision_count": 0,
            "final_content": "",
            "stage_history": [],
            "target_keywords": target_keywords,
            "target_word_count": target_word_count,
            "messages": []
        }
        
        result = compiled_graph.invoke(initial_state)
        
        return {
            "final_content": result["final_content"],
            "quality_score": result["quality_score"],
            "revision_count": result["revision_count"],
            "stage_history": result["stage_history"],
            "topic": result["topic"],
            "audience": result["audience"],
            "tone": result["tone"]
        }
