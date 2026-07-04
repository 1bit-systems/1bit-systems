#!/usr/bin/env python3
"""
issue-swarm.py — Issue-Solving Swarm Orchestrator

Splits repository issues across N Reasonix agents running in parallel.
Designed for powerful multi-core Ryzen machines — spawn as many agents
as your hardware can handle.

Four modes:
  plan    — Read issues JSON, group by area, partition if needed
  report  — Read swarm-results.json, generate swarm-report.md
  stats   — Show stats about the issue files
  extract — Dump a single issue as JSON (for manual subagent feeding)

Partitioning splits the work across multiple Reasonix sessions/machines:
  # On machine 1 (issues 0-49)
  python3 tools/issue-swarm.py plan --batch issues_batch1.json --concurrency 5 --partitions 4 --partition-id 0

  # On machine 2 (issues 50-99)
  python3 tools/issue-swarm.py plan --batch issues_batch1.json --concurrency 5 --partitions 4 --partition-id 1

Usage:
  python3 tools/issue-swarm.py plan --batch issues_batch1.json [--concurrency 5] [--limit 10]
  python3 tools/issue-swarm.py plan --remaining [--limit 5] [--area rocm]
  python3 tools/issue-swarm.py plan --all [--partitions 4 --partition-id 0]
  python3 tools/issue-swarm.py report [--results swarm-results.json]
  python3 tools/issue-swarm.py stats [--file issues_all.json]
  python3 tools/issue-swarm.py extract --number 2451
"""

import json
import sys
import os
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "swarm-results.json"

# Known label prefixes that map to source areas
AREA_PREFIXES = {
    "engine::llamacpp":     "src/cpp/backends/llamacpp/",
    "engine::sd":           "src/cpp/backends/sd/",
    "engine::whispercpp":   "src/cpp/backends/whisper/",
    "engine::vllm":         "src/cpp/backends/vllm/",
    "engine::flm":          "src/cpp/backends/flm/",
    "runtime::rocm":        "src/cpp/ (ROCm runtime)",
    "runtime::vulkan":      "src/cpp/ (Vulkan runtime)",
    "runtime::metal":       "src/cpp/ (Metal runtime)",
    "area::cli":            "src/cpp/cli/",
    "area::api":            "src/cpp/server/ (API routes)",
    "area::ci":             ".github/workflows/",
    "area::installer":      "CMakeLists.txt, setup.sh, packaging/",
    "web ui":               "src/app/src/",
    "cpp":                  "src/cpp/",
    "documentation":        "docs/, README.md",
}

# Priority ordering
PRIORITY_ORDER = {
    "priority:p0": 0,
    "priority:p1": 1,
    "bug": 2,
    "enhancement": 3,
    "documentation": 4,
}

def load_issues(path):
    """Load issues from a JSON file (list of issue objects)."""
    path = Path(path)
    if not path.exists():
        print(f"❌ File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    # Handle both list and dict with issues key
    if isinstance(data, dict) and "issues" in data:
        data = data["issues"]
    return data


def get_issue_area(issue):
    """Determine the primary area/label for an issue."""
    labels = [l["name"] for l in issue.get("labels", [])]
    
    # Check label prefixes in order
    for prefix, area in AREA_PREFIXES.items():
        if any(l.startswith(prefix) or l == prefix for l in labels):
            return area
    
    # Fall back to first label that isn't a meta-label
    meta_labels = {"bug", "enhancement", "documentation"}
    for l in labels:
        if l not in meta_labels:
            return l
    
    return "unclassified"


def get_issue_priority(issue):
    """Get priority score (lower = more important)."""
    labels = [l["name"] for l in issue.get("labels", [])]
    for p, score in PRIORITY_ORDER.items():
        if p in labels:
            return score
    return 99


def format_issue_brief(issue):
    """One-line summary of an issue."""
    labels = ", ".join(l["name"] for l in issue.get("labels", [])[:3])
    remaining = len(issue.get("labels", [])) - 3
    if remaining > 0:
        labels += f" +{remaining}"
    comments = len(issue.get("comments", []))
    return f"  #{issue['number']} {issue['title'][:80]:<80} [{labels}] ({comments} comments)"


def partition_issues(issues, n_partitions, partition_id):
    """Split a list of issues into N partitions, return partition `partition_id`.
    
    Groups related issues (same area) together within each partition
    so a single agent gets a coherent batch.
    """
    # First group by area
    by_area = defaultdict(list)
    for issue in issues:
        area = get_issue_area(issue)
        by_area[area].append(issue)
    
    # Sort areas by total count (largest first)
    sorted_areas = sorted(by_area.keys(), key=lambda a: len(by_area[a]), reverse=True)
    
    # Round-robin assign areas to partitions (greedy: assign each area to 
    # the partition with the fewest issues so far)
    partitions = [[] for _ in range(n_partitions)]
    partition_counts = [0] * n_partitions
    
    for area in sorted_areas:
        area_issues = by_area[area]
        # Assign this whole area to the least-loaded partition
        target = min(range(n_partitions), key=lambda i: partition_counts[i])
        partitions[target].extend(area_issues)
        partition_counts[target] += len(area_issues)
    
    return partitions[partition_id]


def cmd_plan(args):
    """Preprocess issues into a grouped plan for the swarm."""
    source = args.batch or args.remaining or args.all or args.file
    issues = load_issues(source) if source else load_issues("issues_all.json")
    
    # Filter by area if specified
    if args.area:
        issues = [i for i in issues if args.area.lower() in 
                  " ".join(l["name"] for l in i.get("labels", [])).lower()]
    
    # Partition if requested (splits the swarm across machines/sessions)
    if args.partitions and args.partitions > 1:
        if args.partition_id is None:
            print("❌ --partition-id required when --partitions > 1", file=sys.stderr)
            sys.exit(1)
        issues = partition_issues(issues, args.partitions, args.partition_id)
        print(f"\n  🔀 Partition {args.partition_id + 1}/{args.partitions}: {len(issues)} issues\n")
    
    # Group by area
    by_area = defaultdict(list)
    for issue in issues:
        area = get_issue_area(issue)
        by_area[area].append(issue)
    
    # Sort within each area by priority
    for area in by_area:
        by_area[area].sort(key=get_issue_priority)
    
    # Apply limit
    if args.limit and args.limit < len(issues):
        flat = []
        for area in sorted(by_area.keys()):
            flat.extend(by_area[area])
        flat = flat[:args.limit]
        by_area = defaultdict(list)
        for issue in flat:
            by_area[get_issue_area(issue)].append(issue)
    
    # Swarm identity
    swarm_id = args.swarm_id or os.environ.get("SWARM_ID", f"swarm-{datetime.utcnow().strftime('%H%M%S')}")
    machine = args.machine or os.environ.get("MACHINE_NAME", os.uname().nodename)
    
    # Build the plan
    plan = {
        "generated_at": datetime.utcnow().isoformat(),
        "swarm_id": swarm_id,
        "machine": machine,
        "total_issues": len(issues),
        "total_across_all_partitions": args.partitions or 1,
        "partition": args.partition_id if args.partitions else 0,
        "areas": {},
        "issues_ordered": [],
        "concurrency": args.concurrency,
        "source_file": str(source or "issues_all.json"),
    }
    
    print(f"\n{'='*70}")
    print(f"  🐝 SWARM: {swarm_id} on {machine}")
    print(f"  {plan['total_issues']} issues across {len(by_area)} areas | "
          f"concurrency={args.concurrency}")
    if args.partitions:
        print(f"  Partition {args.partition_id + 1}/{args.partitions}")
    print(f"{'='*70}\n")
    
    for area in sorted(by_area.keys()):
        area_issues = by_area[area]
        plan["areas"][area] = {
            "count": len(area_issues),
            "issues": [i["number"] for i in area_issues],
        }
        print(f"  📁 {area} ({len(area_issues)} issues)")
        for issue in area_issues[:5]:
            print(f"     #{issue['number']} — {issue['title'][:70]}")
        if len(area_issues) > 5:
            print(f"     ... and {len(area_issues) - 5} more")
        plan["issues_ordered"].extend(
            {"number": i["number"], "title": i["title"],
             "area": area, "labels": [l["name"] for l in i.get("labels", [])],
             "comments_count": len(i.get("comments", [])),
             "updated_at": i.get("updatedAt", "")}
            for i in area_issues
        )
        print()
    
    # Write plan with swarm-specific filename to avoid collisions
    plan_filename = f"swarm-plan-{swarm_id}.json"
    if args.partitions:
        plan_filename = f"swarm-plan-{swarm_id}-p{args.partition_id}.json"
    plan_path = REPO_ROOT / plan_filename
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    
    print(f"  ✅ Plan written to {plan_filename}")
    print(f"  💡 Copy this file to the target machine and run:")
    print(f"     /skill swarm-command --plan {plan_filename}")
    if args.partitions and args.partitions > 1:
        print(f"\n  💡 Partitions remaining to spawn on other machines:")
        for p in range(args.partitions):
            if p != args.partition_id:
                print(f"     partition {p + 1}/{args.partitions}: "
                      f"python3 tools/issue-swarm.py plan --file {source} "
                      f"--partitions {args.partitions} --partition-id {p} "
                      f"--swarm-id {swarm_id}")


def cmd_report(args):
    """Generate a consolidated report from swarm results."""
    results_path = Path(args.results) if args.results else DEFAULT_RESULTS
    
    if not results_path.exists():
        print(f"❌ No results file found at {results_path}", file=sys.stderr)
        print(f"   Run the swarm first, or specify --results <path>", file=sys.stderr)
        sys.exit(1)
    
    with open(results_path) as f:
        results = json.load(f)
    
    if not results:
        print("❌ Results file is empty")
        sys.exit(1)
    
    # Categorize
    high = [r for r in results if r.get("fix_proposal", {}).get("confidence") == "high"]
    medium = [r for r in results if r.get("fix_proposal", {}).get("confidence") == "medium"]
    low = [r for r in results if r.get("fix_proposal", {}).get("confidence") == "low"]
    needs_human = [r for r in results if r.get("needs_human")]
    
    report_lines = []
    report_lines.append("# 🐝 Swarm Report\n")
    report_lines.append(f"*Generated: {datetime.utcnow().isoformat()}*\n")
    report_lines.append(f"**Summary:** {len(results)} issues analyzed\n")
    report_lines.append(f"| Confidence | Count |")
    report_lines.append(f"|------------|-------|")
    report_lines.append(f"| 🟢 High (fix ready) | {len(high)} |")
    report_lines.append(f"| 🟡 Medium (needs review) | {len(medium)} |")
    report_lines.append(f"| 🔴 Low (investigation only) | {len(low)} |")
    report_lines.append(f"| 👤 Needs human judgment | {len(needs_human)} |")
    report_lines.append("")
    
    # High confidence results first
    if high:
        report_lines.append("---")
        report_lines.append("## 🟢 High Confidence — Fix-ready\n")
        for r in high:
            report_lines.append(f"### #{r['issue_number']}: {r['title']}")
            report_lines.append(f"**Area:** {r.get('area', 'unknown')}")
            report_lines.append(f"**Diagnosis:** {r['diagnosis']['summary']}")
            report_lines.append(f"**Fix:** {r['fix_proposal']['explanation']}")
            if r['fix_proposal'].get('edits'):
                report_lines.append(f"\n**Files to change:**")
                for edit in r['fix_proposal']['edits']:
                    report_lines.append(f"- `{edit['file']}`")
            if r['fix_proposal'].get('testing_notes'):
                report_lines.append(f"\n**Testing:** {r['fix_proposal']['testing_notes']}")
            report_lines.append("")
    
    # Medium confidence
    if medium:
        report_lines.append("---")
        report_lines.append("## 🟡 Medium Confidence — Needs Review\n")
        for r in medium:
            report_lines.append(f"### #{r['issue_number']}: {r['title']}")
            report_lines.append(f"**Area:** {r.get('area', 'unknown')}")
            report_lines.append(f"**Diagnosis:** {r['diagnosis']['summary']}")
            report_lines.append(f"**Fix:** {r['fix_proposal']['explanation']}")
            report_lines.append("")
    
    # Needs human
    if needs_human:
        report_lines.append("---")
        report_lines.append("## 👤 Needs Human Judgment\n")
        for r in needs_human:
            report_lines.append(f"### #{r['issue_number']}: {r['title']}")
            report_lines.append(f"**Area:** {r.get('area', 'unknown')}")
            report_lines.append(f"**Note:** {r.get('diagnosis', {}).get('summary', 'Requires human investigation')}")
            report_lines.append("")
    
    # Low confidence
    if low:
        report_lines.append("---")
        report_lines.append("## 🔴 Low Confidence\n")
        for r in low:
            report_lines.append(f"- #{r['issue_number']}: {r['title']} — {r.get('diagnosis', {}).get('summary', 'Uncertain')}")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    report_path = REPO_ROOT / "swarm-report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"✅ Report written to swarm-report.md")
    print(f"   {len(high)} fix-ready  |  {len(medium)} needs review  |  {len(needs_human)} needs human  |  {len(low)} low confidence")


def cmd_stats(args):
    """Show stats about the issue files."""
    for fname in [args.file or "issues_all.json", "issues_batch1.json", "issues_remaining.json"]:
        path = Path(fname)
        if not path.exists():
            continue
        issues = load_issues(str(path))
        
        by_area = Counter()
        by_label = Counter()
        bugs = 0
        enhancements = 0
        
        for issue in issues:
            area = get_issue_area(issue)
            by_area[area] += 1
            labels = [l["name"] for l in issue.get("labels", [])]
            for l in labels:
                by_label[l] += 1
            if "bug" in labels:
                bugs += 1
            if "enhancement" in labels:
                enhancements += 1
        
        print(f"\n{'='*60}")
        print(f"  📊 {fname} — {len(issues)} issues")
        print(f"{'='*60}")
        print(f"  Bugs: {bugs}  |  Enhancements: {enhancements}")
        print(f"\n  By area:")
        for area, count in by_area.most_common(10):
            print(f"    {area}: {count}")
        print(f"\n  Top labels:")
        for label, count in by_label.most_common(15):
            print(f"    {label}: {count}")


def cmd_extract(args):
    """Extract a single issue for feeding to a subagent."""
    issues = load_issues(args.file or "issues_all.json")
    
    if args.number:
        matches = [i for i in issues if i["number"] == args.number]
    elif args.search:
        query = args.search.lower()
        matches = [i for i in issues if query in i["title"].lower()]
    else:
        matches = issues[:1]
    
    if not matches:
        print(f"❌ No matching issues found", file=sys.stderr)
        sys.exit(1)
    
    issue = matches[0]
    print(json.dumps(issue, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Issue-Solving Swarm Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # plan
    p_plan = subparsers.add_parser("plan", help="Preprocess issues into a grouped plan")
    p_plan.add_argument("--batch", help="issues_batch1.json path")
    p_plan.add_argument("--remaining", help="issues_remaining.json path")
    p_plan.add_argument("--all", help="issues_all.json path")
    p_plan.add_argument("--file", help="Custom issues JSON file")
    p_plan.add_argument("--limit", type=int, default=0, help="Max issues to include")
    p_plan.add_argument("--area", help="Filter by area/label keyword")
    p_plan.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel subagents (Ryzen can handle more; default 5)")
    p_plan.add_argument("--partitions", type=int, default=0,
                        help="Split work across N machines/sessions (default: no split)")
    p_plan.add_argument("--partition-id", type=int, default=None,
                        help="Which partition this is (0-indexed, requires --partitions)")
    p_plan.add_argument("--swarm-id", default=None,
                        help="Unique swarm identifier (default: auto-generated)")
    p_plan.add_argument("--machine", default=None,
                        help="Machine name for the plan header")
    
    # report
    p_report = subparsers.add_parser("report", help="Generate report from results")
    p_report.add_argument("--results", default=str(DEFAULT_RESULTS), help="Results JSON file")
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Show issue file statistics")
    p_stats.add_argument("--file", help="Specific file to analyze")
    
    # extract
    p_extract = subparsers.add_parser("extract", help="Extract a single issue as JSON")
    p_extract.add_argument("--number", type=int, help="Issue number")
    p_extract.add_argument("--search", help="Search title text")
    p_extract.add_argument("--file", help="Issues JSON file (default: issues_all.json)")
    
    args = parser.parse_args()
    
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "extract":
        cmd_extract(args)


if __name__ == "__main__":
    main()
