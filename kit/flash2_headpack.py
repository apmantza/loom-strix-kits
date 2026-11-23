"""FLASH2_HEADPACK=1 (g_attention_vsplit only; HRX-like DESIGN-vsplit-headpack.md, E338 / E339 / E342): two query heads per WMMA
column tile. Today a wave's 16 WMMA columns are 16 tokens of one query head; a verify span has 8 tokens, so columns 8-15 compute
padding in every QK and PV WMMA. Packed, lane column c is (head 2p + c / 8, token c % 8) for head pair p = wave / 2, and a
workgroup is 3 head pairs x 2 dim halves = 6 waves instead of 12 (gen_flash2s.py sets WAVES and the staging split; this pass
rewrites the column -> (head, token) sites of the generated text).

Exact by construction, given WMMA column independence on gfx1151 (E342: a result column depends only on its own column's
operands; 4,096 random problems bit-identical under column replacement and permutation): every column sees the same K rows,
the same V^T rows (zeroed at the same keys on the masked path), the same Q column, the same partner S partial and the same
softmax state updates, in the same order, as it does in the unpacked kernel.

Span limit: a packed tile holds at most 8 tokens. The leader grid of FLASH2_SPLITK_ANY becomes 8 tokens (t % 8 == 0 leads,
the tile's columns end at the next multiple of 8), so no tile can carry a 9th token whatever the launch (the decode branch runs
vsplit for any launch of fewer than 64 tokens, so runs longer than 8 do occur). Served verify spans (8 tokens packed at 0, 8,
16, ..) and decode tokens have the same leaders on either grid; a longer or misaligned run is split into more leaders, whose
results are the same by the kernel's cut invariance (I2: absolute key splits, the span_alone kernel test).

Every anchor must match its stated count, or the build stops (a rewrite that silently skips a changed shape is a contract
break)."""


def _sub(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"FLASH2_HEADPACK: anchor found {n}x, want {count}: {old[:100]!r}")
    return text.replace(old, new)


def headpack(text, nfh, kb_n):
    # constants and the lane's (head select, token) from its column
    text = _sub(text, "  %head_in_group = index.div %wave, %c2 : index\n",
                "  %head_in_group = index.div %wave, %c2 : index\n"
                "  // FLASH2_HEADPACK: wave = head pair p (wave / 2) x dim half; lane column c = (head 2p + c / 8, token c % 8)\n"
                "  %c8hp = index.constant 8 : index\n"
                "  %lane_tok = index.rem %lane_col, %c8hp : index\n"
                "  %lane_hsel = index.div %lane_col, %c8hp : index\n")
    # head per lane: kv * G + 2p + c / 8
    text = _sub(text, "  %head_value = index.madd %kv, %cg, %head_in_group : index\n",
                "  %hp_pair2 = index.mul %head_in_group, %c2 : index\n"
                "  %hp_hig = index.add %hp_pair2, %lane_hsel : index\n"
                "  %head_value = index.madd %kv, %cg, %hp_hig : index\n")
    # QTOKEN: the column's token is first_token + c % 8 (off = 0 under FLASH2_SPLITK_ANY); slot check at pos0 + c % 8
    text = _sub(text, "  %col_ge = index.cmp uge, %lane_col, %off : index\n", "  %col_ge = index.cmp uge, %lane_tok, %off : index\n")
    text = _sub(text, "  %col_rel = index.sub %lane_col, %off : index\n", "  %col_rel = index.sub %lane_tok, %off : index\n")
    text = _sub(text, "  %lane_col32 = index.cast %lane_col : index to i32\n", "  %lane_col32 = index.cast %lane_tok : index to i32\n")
    # causal threshold per column: the token's position
    text = _sub(text, "  %query_pos = index.add %pos0, %lane_col : index\n", "  %query_pos = index.add %pos0, %lane_tok : index\n")
    # masked path: one pass per token (the V^T mask depends only on the token; both heads' columns of that token keep it)
    text = _sub(text, "    %lane_col32m = index.cast %lane_col : index to i32\n", "    %lane_tok32m = index.cast %lane_tok : index to i32\n")
    for kb in range(kb_n):
        for d in range(nfh):
            text = _sub(text, f"scf.for %mc_{d}_{kb} = [%c0 to %c16 step %c1]", f"scf.for %mc_{d}_{kb} = [%c0 to %c8hp step %c1]")
            text = _sub(text, f"%isc{d}_{kb} = scalar.cmpi eq, %lane_col32m, %mci{d}_{kb} : i32",
                        f"%isc{d}_{kb} = scalar.cmpi eq, %lane_tok32m, %mci{d}_{kb} : i32")
    # the 8-token leader grid (FLASH2_SPLITK_ANY's 16-token grid halved): a tile's columns end at the next multiple of 8
    text = _sub(text, "  %tmod = scalar.andi %t32, %c15_i32 : i32\n",
                "  %c7_i32hp = scalar.constant 7 : i32\n  %tmod = scalar.andi %t32, %c7_i32hp : i32\n")
    text = _sub(text, "  %c16_i32g = scalar.constant 16 : i32\n  %grid_left = scalar.subi %c16_i32g, %tmod : i32\n",
                "  %c8_i32g = scalar.constant 8 : i32\n  %grid_left = scalar.subi %c8_i32g, %tmod : i32\n")
    # staging: gen_flash2s gives 4 segments of 64 dims (128 lanes); each lane loads and stores its segment as two 32-dim halves,
    # so every load keeps today's 32-wide type (KV8's i8 reader rewrite of a 64-wide load crashed loom-compile 0.2.0). Each (key,
    # dim) of K and V^T is still written once, by lane (key, dim / 64), with the value today's lane (key, dim / 32) writes.
    text = _sub(text, "  %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, 192)] : index\n  %kz = vector.splatf 0.0 : vector<64xf16>\n",
                "  %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, 192)] : index\n"
                "  %chw = index.constant 32 : index\n"
                "  %st_dim_bv = index.add %st_dim, %chw : index\n"
                "  %st_dim_b = index.assume %st_dim_bv [range(%st_dim_bv, 32, 224)] : index\n"
                "  %kz = vector.splatf 0.0 : vector<32xf16>\n")
    v32 = "vector<32xf16>"
    text = _sub(text, "      %kvals, %vvals = scf.if %key_valid -> (vector<64xf16>, vector<64xf16>) {\n",
                f"      %kvals, %vvals, %kvals_b, %vvals_b = scf.if %key_valid -> ({v32}, {v32}, {v32}, {v32}) {{\n")
    text = _sub(text, """        %kh = vector.load %k_cache[%kv, %st_dim] : view<4x256xf16> -> vector<64xf16>
        %vh = vector.load %v_cache[%kv, %st_dim] : view<4x256xf16> -> vector<64xf16>
        scf.yield %kh, %vh : vector<64xf16>, vector<64xf16>
      } else {
        scf.yield %kz, %kz : vector<64xf16>, vector<64xf16>
      }
      vector.store %kvals, %k_view[%st_key, %st_dim] : vector<64xf16>, view<32x264xf16>
      scf.for %j = [%c0 to %cseg_w step %c1] unroll {
        %vj = vector.extract %vvals[%j] : vector<64xf16> -> f16
        %vt_row_value = index.add %st_dim, %j : index
        %vt_row = index.assume %vt_row_value [range(%vt_row_value, 0, 255)] : index
        view.store %vj, %vt_view[%vt_row, %st_key] : f16, view<256x40xf16>
      }
""", f"""        %kh = vector.load %k_cache[%kv, %st_dim] : view<4x256xf16> -> {v32}
        %vh = vector.load %v_cache[%kv, %st_dim] : view<4x256xf16> -> {v32}
        %kh_b = vector.load %k_cache[%kv, %st_dim_b] : view<4x256xf16> -> {v32}
        %vh_b = vector.load %v_cache[%kv, %st_dim_b] : view<4x256xf16> -> {v32}
        scf.yield %kh, %vh, %kh_b, %vh_b : {v32}, {v32}, {v32}, {v32}
      }} else {{
        scf.yield %kz, %kz, %kz, %kz : {v32}, {v32}, {v32}, {v32}
      }}
      vector.store %kvals, %k_view[%st_key, %st_dim] : {v32}, view<32x264xf16>
      vector.store %kvals_b, %k_view[%st_key, %st_dim_b] : {v32}, view<32x264xf16>
      scf.for %j = [%c0 to %chw step %c1] unroll {{
        %vj = vector.extract %vvals[%j] : {v32} -> f16
        %vt_row_value = index.add %st_dim, %j : index
        %vt_row = index.assume %vt_row_value [range(%vt_row_value, 0, 255)] : index
        view.store %vj, %vt_view[%vt_row, %st_key] : f16, view<256x40xf16>
        %vj_b = vector.extract %vvals_b[%j] : {v32} -> f16
        %vt_row_b_value = index.add %st_dim_b, %j : index
        %vt_row_b = index.assume %vt_row_b_value [range(%vt_row_b_value, 0, 255)] : index
        view.store %vj_b, %vt_view[%vt_row_b, %st_key] : f16, view<256x40xf16>
      }}
""")
    # no other reader of a lane's column as a token or head may remain
    for left in ("%lane_col32m", "index.add %pos0, %lane_col ", "%c16_i32g"):
        if left in text:
            raise SystemExit(f"FLASH2_HEADPACK: unrewritten column site left: {left!r}")
    return text
