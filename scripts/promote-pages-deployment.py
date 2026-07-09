#!/usr/bin/env python3
"""Cloudflare Pages deployment promotion script.

Usage: promote-pages-deployment.py <account_id> <api_token> <project_name>

Lists deployments, sets production_branch, deletes stale production deployment.
"""
import json
import os
import subprocess
import sys
import urllib.request


def api(method, url, token, data=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    account_id = sys.argv[1]
    token = sys.argv[2]
    project = sys.argv[3]
    base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects/{project}"

    # 1. Set production_branch
    print("=== Setting production_branch to main ===")
    result = api("PATCH", base, token, {"production_branch": "main"})
    print(f"Success: {result.get('success')}")

    # 2. List deployments
    print("=== Listing deployments ===")
    result = api("GET", f"{base}/deployments", token)
    if not result.get("success"):
        print("API error:", result.get("errors"))
        sys.exit(1)
    deploys = result.get("result", [])
    print(f"Total deployments: {len(deploys)}")

    # Find main branch deployments (branch is in metadata for ad_hoc deploys)
    main_deps = []
    for d in deploys:
        trigger = d.get("deployment_trigger", {}) or {}
        # Direct Upload deploys store branch in metadata.branch
        branch = (trigger.get("metadata", {}) or {}).get("branch", "") or ""
        if branch == "main":
            main_deps.append(d)
    print(f"Main branch deployments: {len(main_deps)}")

    if not main_deps:
        print("No main branch deployments found!")
        sys.exit(1)

    # Latest main branch deployment
    latest = max(main_deps, key=lambda d: d.get("created_on", ""))
    latest_id = latest["id"]
    print(f"Latest main deployment: {latest_id}")
    print(f"  URL: {latest.get('url', 'N/A')}")
    print(f"  Environment: {latest.get('environment', '?')}")

    # Current production deployment
    prod_id = None
    for dep in deploys:
        if dep.get("environment") == "production":
            prod_id = dep["id"]
            print(f"Current production: {prod_id}")
            break
    if not prod_id:
        print("Current production: NONE")

    # 3. Delete old production deployment if different from latest
    if prod_id and prod_id != latest_id:
        print(f"=== Deleting old production deployment {prod_id} ===")
        result = api("DELETE", f"{base}/deployments/{prod_id}", token)
        print(f"Deleted: {result.get('success')}")
    elif prod_id == latest_id:
        print("Latest deployment is already production. Nothing to do.")
    else:
        print("No production deployment to delete.")


if __name__ == "__main__":
    main()
