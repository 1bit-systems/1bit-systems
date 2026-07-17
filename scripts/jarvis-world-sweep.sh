#!/bin/bash
set -euo pipefail
#!/bin/sh
set -eu
# jarvis-world-sweep.sh — JARVIS world awareness sweep
#
# Records today's briefing into the awareness system.
# Run this in a 1bit session to have JARVIS do a full world sweep.
#
# Usage:
#   1bit -p "Run jarvis-world-sweep" --skill jarvis-world
#   # Or manually: source scripts/jarvis-world-sweep.sh

echo "🦅 JARVIS World Awareness Sweep"
echo "================================"
echo ""
echo "This script prepares JARVIS to do a world sweep."
echo ""
echo "To run the sweep, start a 1bit session:"
echo "  pi"
echo "  /skill:jarvis-world"
echo "  'JARVIS, run a full world sweep and report back'"
echo ""
echo "Or use one-shot:"
echo '  pi -p "JARVIS, run a full world sweep of AI/LLM engineering news and record discoveries" --skill jarvis-world'
echo ""
echo "The sweep queries:"
echo "  • arxiv / HuggingFace papers"
echo "  • GitHub trending AI repos"
echo "  • Tech blogs (Ollama, llama.cpp, etc.)"
echo "  • Inference optimization news"
echo "  • New model releases"
echo "  • GPU/NPU ecosystem developments"
echo ""
echo "Findings are recorded in ~/.1bit/agent/awareness.json"
echo "and broadcast to any running agents."
