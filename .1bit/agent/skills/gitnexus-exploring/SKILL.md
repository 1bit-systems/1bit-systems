---
name: gitnexus-exploring
description: "Use when the user asks how code works, wants to understand architecture, trace execution flows, or explore unfamiliar parts of the codebase. Examples: \"How does X work?\", \"What calls this function?\", \"Show me the auth flow\""
---

# Exploring Codebases with GitNexus

## When to Use

- "How does authentication work?"
- "What's the project structure?"
- "Show me the main components"
- "Where is the database logic?"
- Understanding code you haven't seen before

## Workflow

```
1. gitnexus_list_repos()                                   → Discover indexed repos
2. gitnexus_query({query: "<what you want to understand>"}) → Find related execution flows
3. gitnexus_context({name: "<symbol>"})                     → Deep dive on specific symbol
4. Read source files for implementation details
```

> If index is stale → run `/gitnexus analyze` to rebuild.

## Checklist

```
- [ ] gitnexus_query for the concept you want to understand
- [ ] Review returned processes (execution flows)
- [ ] gitnexus_context on key symbols for callers/callees
- [ ] Read source files for implementation details
```

## Tools

**gitnexus_query** — find execution flows related to a concept:

```
gitnexus_query({query: "payment processing"})
→ Processes: CheckoutFlow, RefundFlow, WebhookHandler
→ Symbols grouped by flow with file locations
```

**gitnexus_context** — 360-degree view of a symbol:

```
gitnexus_context({name: "validateUser"})
→ Incoming calls: loginHandler, apiMiddleware
→ Outgoing calls: checkToken, getUserById
→ Processes: LoginFlow (step 2/5), TokenRefresh (step 1/3)
```

**gitnexus_cypher** — custom graph queries for deeper exploration:

```
gitnexus_cypher({query: "MATCH (f:Function)-[:CodeRelation {type: 'CALLS'}]->(g) WHERE f.name = 'main' RETURN g.name, g.filePath"})
```

## Example: "How does payment processing work?"

```
1. gitnexus_query({query: "payment processing"})
   → CheckoutFlow: processPayment → validateCard → chargeStripe
   → RefundFlow: initiateRefund → calculateRefund → processRefund

2. gitnexus_context({name: "processPayment"})
   → Incoming: checkoutHandler, webhookHandler
   → Outgoing: validateCard, chargeStripe, saveTransaction

3. Read src/payments/processor.ts for implementation details
```
