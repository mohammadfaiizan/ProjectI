"""
Agent module for Research Assistant system.
Defines the research state graph and agent workflow using LangGraph.
"""

from typing import TypedDict, List, Dict, Any, Literal, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from Config import LLM_Config, Search_Config, Report_Config
from Tools import Search_Web, Fetch_URL_Content, Summarize_Text, Citation_Tracker, Search_Result_Parser


class Research_State(TypedDict):
    """
    State dictionary for research graph.
    Tracks all information throughout the research process.
    """
    topic: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    extracted_content: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    report: str
    citations: Citation_Tracker
    current_step: str
    iteration_count: int
    min_sources_required: int


class Research_Graph:
    """
    Research graph implementing the research workflow.
    Coordinates search, content extraction, analysis, and report synthesis.
    """
    
    def __init__(
        self,
        llm_config: LLM_Config,
        search_config: Search_Config,
        report_config: Report_Config
    ):
        """
        Initialize research graph with configurations.
        
        Args:
            llm_config: LLM configuration
            search_config: Search configuration
            report_config: Report configuration
        """
        self.llm_config = llm_config
        self.search_config = search_config
        self.report_config = report_config
        self.llm = llm_config.get_llm()
        self.citation_tracker = Citation_Tracker(report_config.citation_style)
        self.result_parser = Search_Result_Parser(search_config.max_total_results)
        self.graph = None
    
    def Build_Graph(self) -> StateGraph:
        """
        Build the research state graph with all nodes and edges.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(Research_State)
        
        # Add nodes
        workflow.add_node("Generate_Queries", self._generate_queries_node)
        workflow.add_node("Execute_Search", self._execute_search_node)
        workflow.add_node("Extract_Content", self._extract_content_node)
        workflow.add_node("Analyze_Sources", self._analyze_sources_node)
        workflow.add_node("Synthesize_Report", self._synthesize_report_node)
        
        # Set entry point
        workflow.set_entry_point("Generate_Queries")
        
        # Add edges
        workflow.add_edge("Generate_Queries", "Execute_Search")
        workflow.add_edge("Execute_Search", "Extract_Content")
        workflow.add_edge("Extract_Content", "Analyze_Sources")
        workflow.add_conditional_edges(
            "Analyze_Sources",
            self._should_continue_research,
            {
                "continue": "Generate_Queries",
                "synthesize": "Synthesize_Report"
            }
        )
        workflow.add_edge("Synthesize_Report", END)
        
        self.graph = workflow
        return workflow
    
    def _generate_queries_node(self, state: Research_State) -> Research_State:
        """
        Generate diverse search queries from research topic.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with search queries
        """
        topic = state.get("topic", "")
        existing_queries = state.get("search_queries", [])
        iteration = state.get("iteration_count", 0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Generate diverse, specific search queries that will help gather comprehensive information on a research topic."),
            ("human", """Generate {num_queries} diverse search queries for researching: {topic}
            
            Requirements:
            - Each query should explore different aspects or angles of the topic
            - Queries should be specific enough to yield relevant results
            - Avoid redundant or overly similar queries
            - Focus on different subtopics, applications, or perspectives
            
            {existing_context}
            
            Return only the queries, one per line, without numbering.""")
        ])
        
        num_queries = 5 if iteration == 0 else 3
        existing_context = ""
        if existing_queries:
            existing_context = f"Previous queries: {', '.join(existing_queries[:3])}. Generate new queries that explore different aspects."
        
        chain = prompt | self.llm
        response = chain.invoke({
            "topic": topic,
            "num_queries": num_queries,
            "existing_context": existing_context
        })
        
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        queries = queries[:num_queries]
        
        # Combine with existing queries
        all_queries = existing_queries + queries
        
        state["search_queries"] = all_queries
        state["current_step"] = "Generate_Queries"
        state["iteration_count"] = iteration + 1
        
        return state
    
    def _execute_search_node(self, state: Research_State) -> Research_State:
        """
        Execute web searches for all generated queries.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with search results
        """
        queries = state.get("search_queries", [])
        existing_results = state.get("search_results", [])
        
        all_results = existing_results.copy()
        
        # Execute searches for new queries only
        new_queries = queries[len(existing_results):] if len(existing_results) > 0 else queries
        
        for query in new_queries:
            try:
                results = Search_Web.invoke({"query": query})
                all_results.extend(results)
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
        
        # Parse and clean results
        parsed_results = self.result_parser.parse_results(all_results)
        
        # Rank results by relevance to topic
        topic = state.get("topic", "")
        ranked_results = self.result_parser.rank_results(parsed_results, topic)
        
        # Limit to max results
        final_results = ranked_results[:self.search_config.max_total_results]
        
        state["search_results"] = final_results
        state["current_step"] = "Execute_Search"
        
        return state
    
    def _extract_content_node(self, state: Research_State) -> Research_State:
        """
        Fetch and extract content from top search results.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with extracted content
        """
        search_results = state.get("search_results", [])
        existing_content = state.get("extracted_content", [])
        
        # Determine how many URLs to fetch
        max_to_fetch = min(10, len(search_results))
        urls_to_fetch = [r["url"] for r in search_results[:max_to_fetch]]
        
        # Filter out already fetched URLs
        fetched_urls = {item["url"] for item in existing_content}
        new_urls = [url for url in urls_to_fetch if url not in fetched_urls]
        
        extracted = existing_content.copy()
        
        for url in new_urls:
            try:
                content = Fetch_URL_Content.invoke({"url": url})
                extracted.append({
                    "url": url,
                    "title": content.get("title", ""),
                    "content": content.get("content", ""),
                    "author": content.get("author", "Unknown"),
                    "date": content.get("date", "")
                })
            except Exception as e:
                print(f"Error fetching content from '{url}': {e}")
        
        state["extracted_content"] = extracted
        state["current_step"] = "Extract_Content"
        
        return state
    
    def _analyze_sources_node(self, state: Research_State) -> Research_State:
        """
        Analyze and summarize each source, extracting key findings.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with summaries
        """
        extracted_content = state.get("extracted_content", [])
        existing_summaries = state.get("summaries", [])
        
        # Analyze new sources only
        analyzed_urls = {s["url"] for s in existing_summaries}
        new_sources = [item for item in extracted_content if item["url"] not in analyzed_urls]
        
        summaries = existing_summaries.copy()
        
        for source in new_sources:
            try:
                # Summarize the content
                summary_text = Summarize_Text.invoke({
                    "text": source.get("content", ""),
                    "llm": self.llm
                })
                
                # Extract key findings
                findings_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a research analyst. Extract key findings, insights, and important points from research content."),
                    ("human", "Extract 3-5 key findings from this research summary:\n\n{summary}")
                ])
                
                findings_chain = findings_prompt | self.llm
                findings_response = findings_chain.invoke({"summary": summary_text})
                findings = findings_response.content.split("\n")[:5]
                
                # Add citation
                citation_num = self.citation_tracker.add_source(
                    title=source.get("title", ""),
                    url=source.get("url", ""),
                    author=source.get("author", "Unknown"),
                    date=source.get("date", "")
                )
                
                summaries.append({
                    "url": source["url"],
                    "title": source.get("title", ""),
                    "summary": summary_text,
                    "findings": findings,
                    "citation_number": citation_num
                })
            except Exception as e:
                print(f"Error analyzing source '{source.get('url', 'unknown')}': {e}")
        
        state["summaries"] = summaries
        state["citations"] = self.citation_tracker
        state["current_step"] = "Analyze_Sources"
        
        return state
    
    def _should_continue_research(self, state: Research_State) -> Literal["continue", "synthesize"]:
        """
        Determine if more research is needed or if we can synthesize the report.
        
        Args:
            state: Current research state
            
        Returns:
            "continue" if more research needed, "synthesize" otherwise
        """
        summaries = state.get("summaries", [])
        min_sources = state.get("min_sources_required", 5)
        iteration = state.get("iteration_count", 0)
        max_iterations = 3
        
        # Check if we have enough sources
        if len(summaries) >= min_sources:
            return "synthesize"
        
        # Check if we've exceeded max iterations
        if iteration >= max_iterations:
            return "synthesize"
        
        # Continue research if we need more sources
        return "continue"
    
    def _synthesize_report_node(self, state: Research_State) -> Research_State:
        """
        Synthesize all findings into a structured research report.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with final report
        """
        topic = state.get("topic", "")
        summaries = state.get("summaries", [])
        citation_tracker = state.get("citations", self.citation_tracker)
        
        # Prepare content for synthesis
        all_findings = []
        for summary in summaries:
            findings_text = "\n".join([f"- {f}" for f in summary.get("findings", [])])
            citation = citation_tracker.format_citation(summary["url"])
            all_findings.append(f"Source: {summary['title']} {citation}\n{summary['summary']}\n\nKey Findings:\n{findings_text}")
        
        findings_text = "\n\n---\n\n".join(all_findings)
        
        # Generate report sections
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research report writer. Create a comprehensive, well-structured research report based on the provided findings.
            
            Structure the report with:
            1. Executive Summary (if requested)
            2. Introduction
            3. Methodology (if requested)
            4. Main Content Sections (3-6 sections covering different aspects)
            5. Key Findings and Analysis
            6. Conclusion (if requested)
            
            Use proper citations throughout. Write in clear, professional language."""),
            ("human", """Create a comprehensive research report on: {topic}

            Available Findings:
            {findings}

            Requirements:
            - Include {min_sections} to {max_sections} main content sections
            - Use citations: {citation_format}
            - Write {max_words} words per section
            - Ensure each section has at least {min_sources} sources cited
            - Use markdown formatting: {use_markdown}
            
            Generate the complete report.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "topic": topic,
            "findings": findings_text[:8000],  # Limit input length
            "min_sections": self.report_config.min_sections,
            "max_sections": self.report_config.max_sections,
            "citation_format": self.report_config.citation_style,
            "max_words": self.report_config.max_words_per_section,
            "min_sources": self.report_config.min_sources_per_section,
            "use_markdown": self.report_config.enable_markdown_formatting
        })
        
        report = response.content
        
        # Add bibliography
        bibliography = citation_tracker.generate_bibliography()
        final_report = f"{report}\n\n{bibliography}"
        
        state["report"] = final_report
        state["current_step"] = "Synthesize_Report"
        
        return state
    
    def Compile(self, checkpointer=None):
        """
        Compile the research graph with optional checkpointer.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
            
        Returns:
            Compiled graph
        """
        if self.graph is None:
            self.Build_Graph()
        
        if checkpointer is None:
            checkpointer = MemorySaver()
        
        return self.graph.compile(checkpointer=checkpointer)
    
    def Run(self, topic: str, min_sources: int = 5) -> Dict[str, Any]:
        """
        Run the research graph for a given topic.
        
        Args:
            topic: Research topic
            min_sources: Minimum number of sources required
            
        Returns:
            Final state dictionary with research results
        """
        compiled_graph = self.Compile()
        
        initial_state = {
            "topic": topic,
            "search_queries": [],
            "search_results": [],
            "extracted_content": [],
            "summaries": [],
            "report": "",
            "citations": self.citation_tracker,
            "current_step": "start",
            "iteration_count": 0,
            "min_sources_required": min_sources
        }
        
        config = {"configurable": {"thread_id": "research-1"}}
        
        final_state = None
        for state in compiled_graph.stream(initial_state, config):
            final_state = state
            step = list(state.keys())[0] if state else None
            if step:
                print(f"Completed step: {step}")
        
        # Get final state
        if final_state:
            last_node = list(final_state.keys())[-1]
            return final_state[last_node]
        
        return initial_state
