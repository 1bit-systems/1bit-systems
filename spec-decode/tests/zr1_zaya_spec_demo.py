#!/usr/bin/env python3
"""
ZR1→Zaya Speculative Decode Demo

Runs two zaya_server instances:
  - Port 8081: ZR1-1.5B (draft, 26 tok/s on ZINC)
  - Port 8082: Zaya1-8B (target, faster inference)

Coordinator:
  1. Sends prompt to ZR1 → generates N draft tokens (autoregressive)
  2. Sends prompt + draft tokens to Zaya → verifies in one pass
  3. Rejection sampling: accept matching tokens, reject at first mismatch
  4. Reports: acceptance rate, speedup, tok/s

Usage:
  # Start servers (in separate terminals or background):
  ./build/zaya_server --model models/ZR1-1.5B.1bp --port 8081
  ./build/zaya_server --model models/ZAYA1-8B-Q4_K_M.gguf --port 8082
  
  # Run coordinator:
  python3 spec-decode/tests/zr1_zaya_spec_demo.py
"""

import requests
import json
import time
import sys
import argparse

DRAFT_URL = "http://127.0.0.1:8081"
TARGET_URL = "http://127.0.0.1:8082"

def generate(server_url, prompt, max_tokens=64, temperature=0.0):
    """Generate tokens from a server. Returns list of token IDs and text."""
    payload = {
        "model": "zaya",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Request token IDs in response
        "echo": True,
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                print(f"  Server returned {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.exceptions.ConnectionError:
            print(f"  Connection failed (attempt {attempt+1}/3). Is the server running?")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}")
            return None
    return None

def tokenize(server_url, text):
    """Tokenize text if server supports it."""
    try:
        resp = requests.post(
            f"{server_url}/v1/tokenize",
            json={"text": text},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("tokens", [])
    except:
        pass
    return None

def run_spec_decode(prompt, n_draft=5, n_rounds=10):
    """
    Run one round of speculative decoding:
    1. Generate n_draft draft tokens from ZR1
    2. Verify all at once with Zaya
    3. Accept matching prefix
    """
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'='*60}")
    
    stats = {
        "total_draft": 0,
        "total_accepted": 0,
        "total_verify_calls": 0,
        "total_tokens_generated": 0,
        "rounds": 0,
    }
    
    for round_idx in range(n_rounds):
        print(f"\n--- Round {round_idx + 1} ---")
        
        # Step 1: Generate draft tokens from ZR1
        t0 = time.time()
        draft_text = generate(DRAFT_URL, prompt, max_tokens=n_draft)
        t_draft = time.time() - t0
        
        if not draft_text:
            print("  ZR1 draft failed, stopping")
            break
        
        # Split into tokens (rough: use spaces/newlines)
        draft_parts = draft_text.strip().split()
        n_proposed = min(len(draft_parts), n_draft)
        if n_proposed == 0:
            print("  ZR1 produced no draft tokens, stopping")
            break
        
        draft_snippet = " ".join(draft_parts[:n_proposed])
        print(f"  ZR1 draft ({t_draft*1000:.0f}ms): \"{draft_snippet[:80]}\"")
        
        # Step 2: Verify draft tokens with Zaya
        verify_prompt = prompt + draft_snippet
        t0 = time.time()
        verify_text = generate(TARGET_URL, verify_prompt, max_tokens=n_draft)
        t_verify = time.time() - t0
        
        if not verify_text:
            print("  Zaya verification failed, stopping")
            break
        
        verify_parts = verify_text.strip().split()
        print(f"  Zaya verify ({t_verify*1000:.0f}ms): \"{verify_text[:80]}\"")
        
        # Step 3: Rejection sampling (greedy: compare first tokens)
        n_accepted = 0
        for i in range(min(n_proposed, len(verify_parts))):
            if draft_parts[i] == verify_parts[i]:
                n_accepted += 1
            else:
                break
        
        # Bonus token: Zaya's continuation after all accepted drafts
        n_bonus = 0
        if n_accepted == n_proposed and n_accepted > 0:
            n_bonus = 1  # One bonus token from Zaya
        
        tokens_this_round = n_accepted + (1 if n_accepted > 0 else 0)
        
        print(f"  Accepted: {n_accepted}/{n_proposed} draft tokens + {n_bonus} bonus")
        
        stats["total_draft"] += n_proposed
        stats["total_accepted"] += n_accepted
        stats["total_verify_calls"] += 1
        stats["total_tokens_generated"] += tokens_this_round
        stats["rounds"] += 1
        
        # Use verified text as next prompt (continuation)
        prompt = verify_prompt  # Continue from where we left off
        
        if n_accepted == 0:
            print("  Zero acceptance — ZR1 and Zaya disagree significantly")
    
    # Report
    if stats["rounds"] > 0:
        accept_rate = stats["total_accepted"] / stats["total_draft"] * 100 if stats["total_draft"] > 0 else 0
        print(f"\n{'='*60}")
        print(f"RESULTS ({stats['rounds']} rounds)")
        print(f"{'='*60}")
        print(f"  Draft tokens proposed:  {stats['total_draft']}")
        print(f"  Draft tokens accepted:  {stats['total_accepted']}")
        print(f"  Acceptance rate:        {accept_rate:.1f}%")
        print(f"  Verify calls:           {stats['total_verify_calls']}")
        print(f"  Total tokens generated: {stats['total_tokens_generated']}")
        
        # Theoretical speedup (assuming draft is faster per-token)
        # ZR1: 26 tok/s → 38ms/tok
        # Zaya: need actual perf
        draft_ms_per_tok = 38.5  # 26 tok/s
        target_ms_per_tok = 50.0  # placeholder for Zaya on ZINC
        
        no_spec_ms = stats["total_tokens_generated"] * target_ms_per_tok
        spec_ms = stats["rounds"] * (n_draft * draft_ms_per_tok + target_ms_per_tok)
        speedup = no_spec_ms / spec_ms if spec_ms > 0 else 0
        
        print(f"\n  Theoretical speedup:    {speedup:.2f}x")
        print(f"    (ZR1 draft @ 26 tok/s, target @ {1000/target_ms_per_tok:.0f} tok/s)")
        print(f"    Naive: {no_spec_ms:.0f}ms | Spec: {spec_ms:.0f}ms")
    else:
        print("\n  No rounds completed")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="ZR1→Zaya Speculative Decode Demo")
    parser.add_argument("--prompt", default="Write a short poem about artificial intelligence.",
                       help="Prompt to use")
    parser.add_argument("--draft-port", type=int, default=8081, help="ZR1 draft server port")
    parser.add_argument("--verify-port", type=int, default=8082, help="Zaya verify server port")
    parser.add_argument("--n-draft", type=int, default=5, help="Number of draft tokens per round")
    parser.add_argument("--n-rounds", type=int, default=5, help="Number of spec-decode rounds")
    
    args = parser.parse_args()
    
    global DRAFT_URL, TARGET_URL
    DRAFT_URL = f"http://127.0.0.1:{args.draft_port}"
    TARGET_URL = f"http://127.0.0.1:{args.verify_port}"
    
    print(f"ZR1→Zaya Speculative Decode Demo")
    print(f"  Draft server (ZR1-1.5B):  {DRAFT_URL}")
    print(f"  Verify server (Zaya1-8B): {TARGET_URL}")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Draft tokens/round: {args.n_draft}")
    print(f"  Rounds: {args.n_rounds}")
    
    # Health check
    for name, url in [("ZR1 draft", DRAFT_URL), ("Zaya verify", TARGET_URL)]:
        try:
            r = requests.get(f"{url}/v1/health", timeout=5)
            print(f"  {name}: {'✅' if r.status_code == 200 else '❌'} ({url})")
        except:
            print(f"  {name}: ❌ NOT REACHABLE ({url})")
            print(f"\n  Start servers:\n"
                  f"    ./build/zaya_server --model models/ZR1-1.5B.1bp --port {args.draft_port}\n"
                  f"    ./build/zaya_server --model models/ZAYA1-8B-Q4_K_M.gguf --port {args.verify_port}")
            sys.exit(1)
    
    run_spec_decode(args.prompt, args.n_draft, args.n_rounds)

if __name__ == "__main__":
    main()
