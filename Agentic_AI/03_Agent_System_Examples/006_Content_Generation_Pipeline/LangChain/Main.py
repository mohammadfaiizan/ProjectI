"""
Main module for Content Generation Pipeline.
Contains setup functions and interactive demo.
"""

from Config import LLM_Config, Content_Config, SEO_Config, Quality_Config
from Agent import Content_Pipeline_Graph
from typing import Dict, Any, Optional


def Setup_Pipeline(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    max_word_count: int = 2000,
    min_quality_score: float = 0.7,
    max_revision_rounds: int = 3
) -> Content_Pipeline_Graph:
    """
    Setup and configure the content generation pipeline.
    
    Args:
        model_name: Name of the LLM model to use
        temperature: Temperature for LLM generation
        api_key: OpenAI API key (optional, uses env var if not provided)
        max_word_count: Maximum word count for generated content
        min_quality_score: Minimum quality score threshold
        max_revision_rounds: Maximum number of revision rounds
        
    Returns:
        Configured Content_Pipeline_Graph instance
    """
    llm_config = LLM_Config(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key
    )
    
    content_config = Content_Config(
        max_word_count=max_word_count
    )
    
    seo_config = SEO_Config(
        min_keyword_density=0.01,
        max_title_length=60,
        meta_description_length=160
    )
    
    quality_config = Quality_Config(
        min_quality_score=min_quality_score,
        max_revision_rounds=max_revision_rounds
    )
    
    llm = llm_config.Get_LLM()
    
    pipeline = Content_Pipeline_Graph(
        llm=llm,
        content_config=content_config,
        seo_config=seo_config,
        quality_config=quality_config
    )
    
    pipeline.Build_Graph()
    
    return pipeline


def Generate_Content(
    topic: str,
    audience: str,
    tone: str,
    pipeline: Optional[Content_Pipeline_Graph] = None,
    target_keywords: Optional[list] = None,
    target_word_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate content using the pipeline.
    
    Args:
        topic: Content topic
        audience: Target audience level
        tone: Content tone
        pipeline: Optional pre-configured pipeline (creates new one if not provided)
        target_keywords: Optional list of target keywords
        target_word_count: Optional target word count
        
    Returns:
        Dictionary containing generated content and metadata
    """
    if pipeline is None:
        pipeline = Setup_Pipeline()
    
    print(f"\nGenerating content for topic: {topic}")
    print(f"Audience: {audience}, Tone: {tone}")
    print("Processing through pipeline...\n")
    
    result = pipeline.Generate(
        topic=topic,
        audience=audience,
        tone=tone,
        target_keywords=target_keywords,
        target_word_count=target_word_count
    )
    
    return result


def Run_Demo() -> None:
    """
    Run interactive demo where user enters topic, audience, and tone.
    """
    print("=" * 70)
    print("Content Generation Pipeline - Interactive Demo")
    print("=" * 70)
    print("\nThis demo will guide you through generating content using the pipeline.")
    print("You'll be asked to provide:\n")
    print("1. Topic: What the content should be about")
    print("2. Audience: Target audience level")
    print("3. Tone: Desired tone for the content\n")
    
    try:
        pipeline = Setup_Pipeline()
        print("Pipeline initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Please ensure OPENAI_API_KEY is set in your environment.")
        return
    
    while True:
        print("-" * 70)
        topic = input("Enter topic (or 'quit' to exit): ").strip()
        
        if topic.lower() == "quit":
            print("\nExiting demo. Goodbye!")
            break
        
        if not topic:
            print("Topic cannot be empty. Please try again.")
            continue
        
        print("\nAvailable audience levels:")
        print("  - beginners")
        print("  - intermediate")
        print("  - advanced")
        print("  - developers")
        print("  - business leaders")
        print("  - general audience")
        
        audience = input("\nEnter target audience: ").strip().lower()
        
        if not audience:
            print("Audience cannot be empty. Please try again.")
            continue
        
        print("\nAvailable tone options:")
        print("  - professional")
        print("  - friendly")
        print("  - technical")
        print("  - casual")
        print("  - formal")
        print("  - conversational")
        
        tone = input("\nEnter tone: ").strip().lower()
        
        if not tone:
            print("Tone cannot be empty. Please try again.")
            continue
        
        use_keywords = input("\nDo you want to specify target keywords? (y/n): ").strip().lower()
        target_keywords = None
        
        if use_keywords == "y":
            keywords_input = input("Enter keywords (comma-separated): ").strip()
            if keywords_input:
                target_keywords = [k.strip() for k in keywords_input.split(",")]
        
        try:
            result = Generate_Content(
                topic=topic,
                audience=audience,
                tone=tone,
                pipeline=pipeline,
                target_keywords=target_keywords
            )
            
            print("\n" + "=" * 70)
            print("Content Generation Complete!")
            print("=" * 70)
            print(f"\nQuality Score: {result['quality_score']:.3f}")
            print(f"Revision Count: {result['revision_count']}")
            print(f"Stages Completed: {', '.join(result['stage_history'])}")
            print("\n" + "-" * 70)
            print("Generated Content:")
            print("-" * 70)
            print(result["final_content"])
            print("\n" + "=" * 70)
            
            save = input("\nDo you want to save this content to a file? (y/n): ").strip().lower()
            if save == "y":
                filename = input("Enter filename (default: generated_content.md): ").strip()
                if not filename:
                    filename = "generated_content.md"
                
                if not filename.endswith(".md"):
                    filename += ".md"
                
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(result["final_content"])
                    print(f"\nContent saved to {filename}")
                except Exception as e:
                    print(f"\nError saving file: {e}")
        
        except Exception as e:
            print(f"\nError generating content: {e}")
            print("Please check your inputs and try again.")
        
        continue_demo = input("\nGenerate another piece of content? (y/n): ").strip().lower()
        if continue_demo != "y":
            print("\nThank you for using the Content Generation Pipeline!")
            break


if __name__ == "__main__":
    Run_Demo()
