#!/usr/bin/env python3
"""
Quick test for GigaChat planner with structured output
"""

import os
import asyncio
from langchain_gigachat import GigaChat
from langchain.schema import HumanMessage
from src.models.planner_model import Plan

async def test_gigachat_planner():
    """Test GigaChat with structured output for planner"""
    
    # Initialize GigaChat
    gigachat = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        model="GigaChat-2-Max",
        verify_ssl_certs=False,
        temperature=0.1
    )
    
    print("🧪 Testing GigaChat with structured output...")
    
    # Test 1: Direct structured output
    print("\n1️⃣ Testing direct structured output with format_instructions...")
    try:
        structured_llm = gigachat.with_structured_output(Plan, method="format_instructions")
        
        messages = [
            HumanMessage(content="Создай план исследования квантовых компьютеров. Нужно изучить принципы работы, текущее состояние технологии и перспективы.")
        ]
        
        result = structured_llm.invoke(messages)
        print(f"✅ Success! Got Plan object: {type(result)}")
        print(f"   - Title: {result.title}")
        print(f"   - Has enough context: {result.has_enough_context}")
        print(f"   - Steps count: {len(result.steps)}")
        print(f"   - Locale: {result.locale}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Test with planner node logic
    print("\n\n2️⃣ Testing with planner node logic...")
    try:
        from src.graph.nodes import planner_node
        from src.config.config_loader import get_settings
        from langgraph.types import RunnableConfig
        
        # Create test state
        state = {
            "messages": [
                HumanMessage(content="Исследуй возможности применения ИИ в медицине")
            ],
            "research_topic": "ИИ в медицине",
            "plan_iterations": 0,
            "current_plan": None,
            "background_investigation_results": "Предварительное исследование показало широкие перспективы ИИ в медицине"
        }
        
        # Create config
        settings = get_settings()
        config = RunnableConfig(
            configurable={
                "locale": "ru-RU",
                "agents": settings.agents,
                "report_language": "ru"
            }
        )
        
        # Run planner node
        result = planner_node(state, config)
        print(f"✅ Planner node executed successfully!")
        print(f"   - Result type: {type(result)}")
        print(f"   - Goto: {result.goto if hasattr(result, 'goto') else 'N/A'}")
        
        # Check if plan was saved to state
        if hasattr(result, '__self__') and 'current_plan' in result.__self__:
            plan = result.__self__['current_plan']
            print(f"   - Plan saved: {plan is not None}")
            if plan:
                print(f"   - Plan title: {plan.get('title', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error in planner node: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gigachat_planner())