#!/usr/bin/env python3
"""
Zaya Co-Host: Break-even Analysis
Strix Halo (owned) vs MI300X (rented) 
"""

print("=" * 63)
print("  ZAYA Co-Host \u2014 Economics Calculator")
print("  $19.85/mo \u00b7 Agnostic Voice AI")
print("=" * 63)

# Model sizes
zaya_q4_gb = 5.2
codec_mb = 25

# KV cache per user
kv_4k_mb = 640
kv_32k_mb = 5120

# Strix Halo: 128.8 GB unified
strix_total = 128.0
strix_available = strix_total - zaya_q4_gb - 8  # OS buffer
strix_users_4k = int(strix_available * 1024 / kv_4k_mb)
strix_users_32k = int(strix_available * 1024 / kv_32k_mb)

# MI300X: 192 GB HBM3
mi_total = 192
mi_available = mi_total - zaya_q4_gb - 4  # overhead
mi_users_4k = int(mi_available * 1024 / kv_4k_mb)
mi_users_32k = int(mi_available * 1024 / kv_32k_mb)
mi_users_128k = int(mi_available * 1024 / (kv_32k_mb * 4))

# Inference speeds
strix_gpu_tps = 426
strix_npu_tps = 69
mi300x_tps = 250

voice_tps = 15
chat_tps = 30

print(f"\n{'Capacity':>25} {'Strix Halo':>15} {'MI300X':>15}")
print(f"  {'-'*53}")
print(f"  {'VRAM':20} {'128.8 GB':>15} {'192 GB':>15}")
print(f"  {'Model (ZAYA-8B Q4)':20} {'5.2 GB':>15} {'5.2 GB':>15}")
print(f"  {'Available for KV':20} {f'{strix_available:.0f} GB':>15} {f'{mi_available:.0f} GB':>15}")
print(f"  {'Users @ 4K ctx':20} {strix_users_4k:>15} {mi_users_4k:>15}")
print(f"  {'Users @ 32K ctx':20} {strix_users_32k:>15} {mi_users_32k:>15}")
print(f"  {'Users @ 128K ctx':20} {'--':>15} {mi_users_128k:>15}")
print(f"  {'Inference speed':20} {f'{strix_gpu_tps} tok/s':>15} {f'{mi300x_tps} tok/s':>15}")
print(f"  {'Voice users':20} {strix_gpu_tps//voice_tps:>15} {mi300x_tps//voice_tps:>15}")
print(f"  {'Chat users':20} {strix_gpu_tps//chat_tps:>15} {mi300x_tps//chat_tps:>15}")

# Revenue model
price = 19.85

print(f"\n{'Revenue ($19.85/mo/user)':^55}")
print(f"  {'Users':>8} {'Strix Only':>14} {'+MI300X':>14} {'Both':>14}")
print(f"  {'-'*48}")

for users in [1, 5, 10, 20, 50, 100, 200]:
    strix_cap = min(users, strix_gpu_tps // voice_tps)
    mi_cap = max(0, min(users, mi300x_tps // voice_tps))
    extra = max(0, users - strix_cap)
    
    strix_rev = strix_cap * price
    mi_rev = mi_cap * price
    both_rev = strix_rev + mi_rev
    
    # Only show MI300X column if we need it
    if strix_cap >= users:
        print(f"  {users:>8} {f'${strix_rev:.0f}/mo':>14} {'--':>14} {f'${both_rev:.0f}/mo':>14}")
    else:
        print(f"  {users:>8} {f'${strix_rev:.0f}/mo':>14} {f'${mi_rev:.0f}/mo':>14} {f'${both_rev:.0f}/mo':>14}")

# Costs
strix_power_w = 120
mi300x_rent_hr = 1.00

strix_power_cost = strix_power_w * 24 * 30 / 1000 * 0.12  # $0.12/kWh
mi300x_monthly = mi300x_rent_hr * 24 * 30
bandwidth_10 = 10

print(f"\n{'Monthly Costs':^45}")
print(f"  Strix Halo power (120W x 24/7 x $0.12/kWh)  ${strix_power_cost:.2f}/mo")
print(f"  MI300X rent (@ $1.00/hr 24/7)                ${mi300x_monthly:.0f}/mo")
print(f"  Bandwidth (10 users, 1TB)                    ${bandwidth_10:.0f}/mo")

print(f"\n{'Break-Even':^45}")
print(f"  {'Scenario':30} {'Users':>8} {'Net/mo':>10}")
print(f"  {'-'*48}")

# Strix only
for u in [5, 10, 20, 28]:
    rev = u * price
    costs = strix_power_cost + bandwidth_10
    net = rev - costs
    label = "Strix only @ " + str(u) + " users"
    print(f"  {label:30} {u:>8} ${net:>7.0f}")

# Strix + MI300X
print()
for u in [10, 20, 30, 50, 100]:
    rev = u * price
    costs = strix_power_cost + mi300x_monthly + bandwidth_10
    net = rev - costs
    label = "Strix + MI300X @ " + str(u) + " users"
    print(f"  {label:30} {u:>8} ${net:>7.0f}")

print(f"\n{'Key Numbers':^45}")
strix_breakeven = (strix_power_cost + bandwidth_10) / price
mi_breakeven = (strix_power_cost + mi300x_monthly + bandwidth_10) / price
strix_max = strix_gpu_tps // voice_tps

print(f"  Strix Halo breakeven:     {strix_breakeven:.1f} users (just power + bw)")
print(f"  Strix + MI300X breakeven: {mi_breakeven:.0f} users (covers rent)")
print(f"  Strix max voice users:    {strix_max} (GPU compute bottleneck)")
print(f"  Strix max memory users:   {strix_users_4k} (128GB memory bottleneck)")
print(f"  MI300X max voice users:   {mi300x_tps // voice_tps} (compute)")
print(f"  MI300X max memory users:  {mi_users_4k} (192GB memory)")

print(f"\n{'The Strategy':^45}")
print(f"  1. Strix alone:   {strix_max} users x $19.85 = ${strix_max * price:.0f}/mo pure margin")
print(f"  2. Hit {strix_max + 1}th user: rent MI300X, revenue covers rent at {mi_breakeven:.0f} users")
print(f"  3. At 50 users:  ${50 * price:.0f}/mo - ${mi300x_monthly:.0f} rent = ${50 * price - mi300x_monthly:.0f}/mo net")
print(f"  4. At 100 users: ${100 * price:.0f}/mo - ${mi300x_monthly:.0f} rent = ${100 * price - mi300x_monthly:.0f}/mo net")
print(f"")
print(f"  Bottom line:  Strix Halo is free inference forever.")
print(f"  MI300X only when you hit {strix_max}+ voice users.")
print(f"  HW you already own is the entire MVP.")
print(f"{'='*63}")
