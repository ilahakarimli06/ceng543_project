"""
Quick verification script to test BigBird integration into main.py pipeline
"""
import sys
sys.path.insert(0, '/home/ilaha/ceng543_project')

from src.models.sliding import load_model, generate_with_windows

print("=" * 60)
print("Testing BigBird Integration")
print("=" * 60)

# Test 1: Load BigBird model
print("\n1. Testing BigBird model loading...")
try:
    model, tok, metadata = load_model(
        "google/bigbird-pegasus-large-arxiv",
        attention_impl="default",
        use_bf16=True,
        use_compile=False  # Skip compile for quick test
    )
    print("✅ BigBird model loaded successfully!")
    print(f"   Model family: {metadata['model_family']}")
    print(f"   Attention backend: {metadata['attention_backend']}")
    print(f"   Max length: {metadata.get('max_length', 'N/A')}")
    print(f"   Block size: {metadata.get('block_size', 'N/A')}")
except Exception as e:
    print(f"❌ BigBird loading failed: {e}")
    sys.exit(1)

# Test 2: Test generation routing
print("\n2. Testing generation function routing...")
try:
    # Create dummy windows (token IDs)
    dummy_windows = [[1, 2, 3, 4, 5] * 100]  # Simple test window
    
    # This should automatically route to generate_with_windows_bigbird
    summary = generate_with_windows(
        model, tok, dummy_windows, 
        gen_max=50, 
        global_tokens=0,  # Should be ignored for BigBird
        batch_size=1
    )
    print("✅ Generation routing works!")
    print(f"   Generated summary length: {len(summary)} chars")
except Exception as e:
    print(f"❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify LED still works (no regression)
print("\n3. Testing LED model (regression check)...")
try:
    led_model, led_tok, led_metadata = load_model(
        "allenai/led-base-16384",
        attention_impl="default",
        use_bf16=True,
        use_compile=False
    )
    print("✅ LED model still loads successfully!")
    print(f"   Model family: {led_metadata['model_family']}")
    print(f"   Attention backend: {led_metadata['attention_backend']}")
except Exception as e:
    print(f"❌ LED loading failed (REGRESSION!): {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - BigBird integration successful!")
print("=" * 60)
print("\nUsage:")
print("  LED:     python main.py --config CONFIG_FILE")
print("  BigBird: python main.py --config CONFIG_FILE --model google/bigbird-pegasus-large-arxiv")
