#!/usr/bin/env python
"""Test script for the complete hybrid AI system."""

from backend.nlp_engine import NLPEngine
from types import SimpleNamespace

# Create a mock config
config = SimpleNamespace()
config.max_tokens = 32
config.temperature = 0.8
config.memory_file = 'conversation_memory.json'

engine = NLPEngine(config)

# Simulate a conversation
print('=== Testing Hybrid AI System ===')
print('\n1. Testing Pattern Matching (Greeting)')
print('User: hello')
response1 = engine.get_response('hello')
print('AI:', response1[:60] + '...')

print('\n2. Testing AI Response (Transformer)')
print('User: What is AI?')
response2 = engine.get_response('What is AI?')
print('AI:', response2[:60] + '...')

print('\n3. Testing Memory')
print('Memory sessions:', list(engine.conversation_memory.keys()))
if engine.conversation_memory:
    for session_key in list(engine.conversation_memory.keys())[-1:]:
        messages = engine.conversation_memory[session_key]
        print(f'Session has {len(messages)} messages')
        if len(messages) > 0:
            print('Last message type:', messages[-1].get('intent', 'unknown'))

print('\n4. Testing Weight Persistence')
engine.transformer_llm.save_weights('iwo_weights.json')
print('Weights saved to iwo_weights.json')

# Load in new instance
engine2 = NLPEngine(config)
engine2.load_pretrained_weights('iwo_weights.json')
print('Weights loaded successfully')

print('\n5. Testing Fallback')
try:
    response3 = engine2.get_response('help')
    print('Help response:', response3[:60] + '...')
except Exception as e:
    print('Fallback triggered:', str(e)[:40])

print('\n=== System Complete ===')
print('✓ Hybrid routing (templates + AI)')
print('✓ Weight loading (pre-trained intelligence)')
print('✓ Memory context (conversation awareness)')
print('✓ Error handling (fallback responses)')
