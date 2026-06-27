# test_llm_spike.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr.llm_restoration import GeezRestorationEngine

engine = GeezRestorationEngine(mode="ollama_fewshot")

# Test with a known damaged inscription
test_cases = [
    ("ሰ[MISSING]ም",        "Aksumite", "Lalibela"),
    ("ዓጼ [MISSING] ነጉሠ",   "4th century CE", "Aksum"),
    ("[MISSING]ሪ[MISSING]ም", "12th century CE", "Lalibela"),
]

print("Testing Ge'ez LLM Restoration Engine\n" + "="*50)
for damaged, period, location in test_cases:
    print(f"\nInput:    {damaged}")
    result = engine.restore(damaged, period, location)
    print(f"Restored: {result['restored_text']}")
    print(f"English:  {result['translation']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Reasoning: {result['reasoning']}")