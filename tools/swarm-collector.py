#!/usr/bin/env python3
"""Swarm Result Collector — saves subagent results to swarm-results.json"""
import json, sys, os
RESULTS_FILE = 'swarm-results.json'

def save_result(result):
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    # Avoid duplicates
    for i, r in enumerate(results):
        if r.get('issue_number') == result.get('issue_number'):
            results[i] = result
            break
    else:
        results.append(result)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved result for #{result.get("issue_number")} ({len(results)} total)')

def generate_report():
    if not os.path.exists(RESULTS_FILE):
        print('No results yet')
        return
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    
    high = [r for r in results if r.get('fix_proposal',{}).get('confidence') == 'high']
    medium = [r for r in results if r.get('fix_proposal',{}).get('confidence') == 'medium']
    low = [r for r in results if r.get('fix_proposal',{}).get('confidence') == 'low']
    needshuman = [r for r in results if r.get('needs_human')]
    
    print(f'=== Swarm Report ===')
    print(f'Total results: {len(results)}')
    print(f'High confidence: {len(high)}')
    print(f'Medium confidence: {len(medium)}')  
    print(f'Low confidence: {len(low)}')
    print(f'Needs human: {len(needshuman)}')
    print()
    if high:
        print('High confidence fixes ready to apply:')
        for r in high:
            print(f'  #{r["issue_number"]}: {r["title"][:70]}')
        print()
    if needshuman:
        print('Needs human judgment:')
        for r in needshuman:
            print(f'  #{r["issue_number"]}: {r["title"][:70]}')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        generate_report()
    elif len(sys.argv) > 1 and sys.argv[1] == 'save':
        save_result(json.loads(sys.argv[2]))
