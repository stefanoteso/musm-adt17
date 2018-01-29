\ Model inference
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  15491.74497380929 x13 + 19200.28943455296 x113 + 1681.626870531186 x114
   + 15491.74497380929 x23 + 19200.28943455296 x213
   + 1681.626870531186 x214 + 0.5 absdiff0 + 0.5 absdiff1 + 0.5 absdiff2
   + 0.5 absdiff3 + 0.5 absdiff4 + 0.5 absdiff5 + 0.5 absdiff6
   + 0.5 absdiff7 + 0.5 absdiff8 + 0.5 absdiff9 + 0.5 absdiff10
   + 0.5 absdiff11 + 0.5 absdiff12 + 0.5 absdiff13 + 0.5 absdiff14
   + 0.5 absdiff15
Subject To
 R0: - x10 + x20 + diff0 = 0
 R1: - x11 + x21 + diff1 = 0
 R2: - x12 + x22 + diff2 = 0
 R3: - x13 + x23 + diff3 = 0
 R4: - x14 + x24 + diff4 = 0
 R5: - x15 + x25 + diff5 = 0
 R6: - x16 + x26 + diff6 = 0
 R7: - x17 + x27 + diff7 = 0
 R8: - x18 + x28 + diff8 = 0
 R9: - x19 + x29 + diff9 = 0
 R10: - x110 + x210 + diff10 = 0
 R11: - x111 + x211 + diff11 = 0
 R12: - x112 + x212 + diff12 = 0
 R13: - x113 + x213 + diff13 = 0
 R14: - x114 + x214 + diff14 = 0
 R15: - x115 + x215 + diff15 = 0
 R16: x10 + x11 + x12 + x13 = 1
 R17: x14 + x15 + x16 + x17 = 1
 R18: x18 + x19 + x110 + x111 = 1
 R19: x112 + x113 + x114 + x115 = 1
 R20: x20 + x21 + x22 + x23 = 1
 R21: x24 + x25 + x26 + x27 = 1
 R22: x28 + x29 + x210 + x211 = 1
 R23: x212 + x213 + x214 + x215 = 1
Bounds
 -1 <= diff0 <= 1
 -1 <= diff1 <= 1
 -1 <= diff2 <= 1
 -1 <= diff3 <= 1
 -1 <= diff4 <= 1
 -1 <= diff5 <= 1
 -1 <= diff6 <= 1
 -1 <= diff7 <= 1
 -1 <= diff8 <= 1
 -1 <= diff9 <= 1
 -1 <= diff10 <= 1
 -1 <= diff11 <= 1
 -1 <= diff12 <= 1
 -1 <= diff13 <= 1
 -1 <= diff14 <= 1
 -1 <= diff15 <= 1
 absdiff0 <= 1
 absdiff1 <= 1
 absdiff2 <= 1
 absdiff3 <= 1
 absdiff4 <= 1
 absdiff5 <= 1
 absdiff6 <= 1
 absdiff7 <= 1
 absdiff8 <= 1
 absdiff9 <= 1
 absdiff10 <= 1
 absdiff11 <= 1
 absdiff12 <= 1
 absdiff13 <= 1
 absdiff14 <= 1
 absdiff15 <= 1
Binaries
 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x110 x111 x112 x113 x114 x115 x20
 x21 x22 x23 x24 x25 x26 x27 x28 x29 x210 x211 x212 x213 x214 x215
Generals
 diff0 diff1 diff2 diff3 diff4 diff5 diff6 diff7 diff8 diff9 diff10 diff11
 diff12 diff13 diff14 diff15 absdiff0 absdiff1 absdiff2 absdiff3 absdiff4
 absdiff5 absdiff6 absdiff7 absdiff8 absdiff9 absdiff10 absdiff11 absdiff12
 absdiff13 absdiff14 absdiff15
General Constraints
 GC0: absdiff0 = ABS ( diff0 )
 GC1: absdiff1 = ABS ( diff1 )
 GC2: absdiff2 = ABS ( diff2 )
 GC3: absdiff3 = ABS ( diff3 )
 GC4: absdiff4 = ABS ( diff4 )
 GC5: absdiff5 = ABS ( diff5 )
 GC6: absdiff6 = ABS ( diff6 )
 GC7: absdiff7 = ABS ( diff7 )
 GC8: absdiff8 = ABS ( diff8 )
 GC9: absdiff9 = ABS ( diff9 )
 GC10: absdiff10 = ABS ( diff10 )
 GC11: absdiff11 = ABS ( diff11 )
 GC12: absdiff12 = ABS ( diff12 )
 GC13: absdiff13 = ABS ( diff13 )
 GC14: absdiff14 = ABS ( diff14 )
 GC15: absdiff15 = ABS ( diff15 )
End
