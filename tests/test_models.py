import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.providers import PlannerProvider

models_to_test = [
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/google/gemma-3-27b-it:free",
    "openrouter/nvidia/nemotron-nano-12b-v2-vl:free"
]

dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def run_test_model(model_name):
    print(f"\n--- Testing Model: {model_name} ---")
    provider = PlannerProvider(model=model_name)
    goal = "Open the browser, navigate to example.com, and extract the text."
    
    try:
        print("Testing hierarchical planning...")
        plan = provider.plan_hierarchical(goal, latest_screenshot_b64=dummy_b64)
        
        print("✅ SUCCESS: Successfully generated and parsed hierarchical plan.")
        print(f"Reasoning: {plan.reasoning}")
        print(f"Sub-tasks: {len(plan.sub_tasks)}")
        for st in plan.sub_tasks:
            print(f"  - {st.description} ({len(st.actions)} actions)")
        return True
    except Exception as e:
        print(f"ERROR: Failed to generate or parse plan. Exception: {str(e)}")
        return False

if __name__ == "__main__":
    for model in models_to_test:
        test_model(model)
