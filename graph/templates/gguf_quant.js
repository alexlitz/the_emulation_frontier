/* The in-browser port of graph/gguf_quant.py: llama.cpp's GGUF quantisation sizes.

   Which ggml type llama-quantize gives every tensor for an ftype (a port of src/llama-quant.cpp
   llama_tensor_get_type_impl and its helpers at the commit in the tables), which type convert_hf_to_gguf.py
   writes it in, and the bytes of the file (tensor data padded to the GGUF alignment, plus the header).
   Every table (block sizes / bytes, ftype -> default type, the "may it be quantised" rules, the fallback
   switch, the converter's F32 tensors, the alignment, the LLM_TYPE_70B rules) comes from
   graph/data/llama_cpp_quant.json, written by `python -m graph.gguf_quant`. The source line ranges of every
   branch are in the Python, function for function; this file keeps its structure so the two diff cleanly.

   Inventory: [{name: "blk.3.attn_v.weight", shape: [ne0, ne1, ...] (GGML order), src: "BF16" | "F16" | "F32"}]
   opts (tensorTypes / ggufBytes / breakdown): {n_expert, n_gqa, n_layer, n_head, n_head_kv, n_layer_nextn,
   model_70b, has_imatrix}; sourceTypes(tensors, arch, outtype = "bf16", flags = {lm_head_fp32}).
   Byte totals are Numbers (exact below 2^53).

   tests/test_gguf_quant.py runs it under node against the Python on randomised inventories.

   API:  GgufQuant.setTables(json)
         GgufQuant.tensorTypes(tensors, ftype, arch, opts)      GgufQuant.sourceTypes(tensors, arch, outtype, flags)
         GgufQuant.tensorBytes(type, nElements, ne0)            GgufQuant.ggufBytes(tensors, ftype, arch, headerBytes, opts)
         GgufQuant.breakdown(tensors, ftype, arch, headerBytes, opts)
         GgufQuant.headerBytes(vocabSize, opts)                 GgufQuant.tensorCategory(name)  */
(function (root) {
  "use strict";
  let T = null;
  const RX = new Map();

  function setTables(t) { T = t; RX.clear(); }
  function tables(t) {
    if (t) setTables(t);
    if (!T) throw new Error("GgufQuant: call setTables(json) first");
    return T;
  }
  function rx(p) {
    if (!RX.has(p)) RX.set(p, new RegExp(p));
    return RX.get(p);
  }
  function utf8len(s) {
    let n = 0;
    for (const ch of s) {
      const c = ch.codePointAt(0);
      n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
    }
    return n;
  }
  const fdiv = (a, b) => Math.floor(a / b);
  const opt = (o, k, d) => (o && o[k] !== undefined && o[k] !== null) ? o[k] : d;

  function nDims(shape) {                     // ggml_n_dims
    for (let i = shape.length - 1; i > 0; i--) if (shape[i] > 1) return i + 1;
    return 1;
  }
  function nElements(shape) { let n = 1; for (const d of shape) n *= d; return n; }
  function blkLayer(name) {                   // sscanf(name, "blk.%d.", &il)
    const m = /^blk\.[ \t\n\r\f\v]*([+-]?[0-9]+)/.exec(name);
    return m ? parseInt(m[1], 10) : null;
  }
  function cmpWeights(a, b) {                 // weight_name_comparer: layer (-1 when absent), then name
    const la = blkLayer(a), lb = blkLayer(b);
    const xa = la === null ? -1 : la, xb = lb === null ? -1 : lb;
    if (xa !== xb) return xa < xb ? -1 : 1;
    return a < b ? -1 : a > b ? 1 : 0;
  }

  function tensorCategory(name) {
    if (name === "output.weight") return "OUTPUT";
    if (name === "token_embd.weight" || name === "per_layer_token_embd.weight") return "TOKEN_EMBD";
    if (name.includes("attn_qkv.weight") || name.includes("attn_qkv_gate.weight")) return "ATTENTION_QKV";
    if (name.includes("attn_kv_b.weight")) return "ATTENTION_KV_B";
    if (name.includes("attn_v.weight")) return "ATTENTION_V";
    if (name.includes("attn_k.weight")) return "ATTENTION_K";
    if (name.includes("attn_q.weight")) return "ATTENTION_Q";
    if (name.includes("attn_output.weight")) return "ATTENTION_OUTPUT";
    if (name.includes("ffn_up")) return "FFN_UP";
    if (name.includes("ffn_gate")) return "FFN_GATE";
    if (name.includes("ffn_latent")) return "FFN_LATENT";
    if (name.includes("attn_gate.weight")) return "ATTENTION_GATE";
    if (name.includes("ffn_down")) return "FFN_DOWN";
    return "OTHER";
  }
  const isExpert = name => name.includes("_exps.") || name.includes("_chexps.");
  const isAttnV = c => c === "ATTENTION_V" || c === "ATTENTION_QKV" || c === "ATTENTION_KV_B";
  const among = (x, ...xs) => xs.includes(x);

  function allowsQuantization(name, shape, arch) {
    let quantize = true;
    for (const r of T.allow_rules) {
      if (r.arch !== null && r.arch !== arch) continue;
      const op = r.op;
      if (op === "min_dims") { if (nDims(shape) < r.n) return false; }
      else if (op === "ends_weight") quantize = name.endsWith("weight") || utf8len(name) === 5;
      else if (op === "not_contains") quantize = quantize && !name.includes(r.s);
      else if (op === "not_prefix") quantize = quantize && !name.startsWith(r.s);
      else if (op === "not_equal") quantize = quantize && name !== r.s;
      else throw new Error("unknown rule op " + op);
    }
    return quantize;
  }

  function bpwLt(a, b) {
    const ta = T.types[a], tb = T.types[b];
    return ta.bytes * tb.block < tb.bytes * ta.block;
  }

  function useMoreBits(i, n) {
    return i < fdiv(n, 8) || i >= fdiv(7 * n, 8) || (i - fdiv(n, 8)) % 3 === 2;
  }

  function getTypeImpl(qs, newType, name, ne0, ftype, cat, arch) {
    const nExpert = Math.max(1, qs.n_expert);
    const nExpertH = qs.n_expert;
    const gqa = qs.n_gqa;
    const layerInfo = (iLayer, nLayer) => {
      if (nExpert > 1) {
        iLayer = blkLayer(name);
        if (iLayer === null) throw new Error("Failed to determine layer for tensor " + name);
        if (iLayer < 0 || iLayer >= nLayer)
          throw new Error("Bad layer " + iLayer + " for tensor " + name + ". Must be in [0, " + nLayer + ")");
      }
      return [iLayer, nLayer];
    };

    if (arch === "glm5-next" && (name.includes("attn_q_a") || name.includes("attn_q_b") || name.includes("nextn.eh_proj"))) {
      if (!among(newType, "F32", "BF16", "F16")) return "Q8_0";
    }
    if (arch === "muse-spark" && among(cat, "FFN_LATENT", "ATTENTION_GATE")) {
      if (T.types[newType].quantized && bpwLt(newType, "Q8_0")) newType = "Q8_0";
    }

    if (cat === "OUTPUT" || (qs.has_tied_embeddings && cat === "TOKEN_EMBD")) {
      const qk = T.types[newType].block;
      if (ftype === "MXFP4_MOE") newType = "Q8_0";
      else if (arch === "falcon" || ne0 % qk !== 0) newType = "Q8_0";
      else if (among(ftype, "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ2_S", "IQ2_M", "IQ1_M")) newType = "Q5_K";
      else if (newType !== "Q8_0") newType = "Q6_K";
    } else if (ftype === "MXFP4_MOE") {
      newType = isExpert(name) ? "MXFP4" : "Q8_0";
    } else if (cat === "TOKEN_EMBD") {
      if (among(ftype, "IQ2_XXS", "IQ2_XS", "IQ1_S", "IQ1_M")) newType = "Q2_K";
      else if (among(ftype, "IQ2_S", "IQ2_M")) newType = "IQ3_S";
      else if (ftype === "IQ3_XXS") newType = "IQ3_S";
      else if (among(ftype, "TQ1_0", "TQ2_0", "Q2_0")) newType = "Q4_K";
    } else if (among(ftype, "IQ2_XXS", "IQ2_XS", "IQ1_S", "IQ2_S", "IQ2_M", "IQ1_M")) {
      if (isAttnV(cat)) {
        if (gqa >= 4 || nExpertH >= 4) newType = "Q4_K";
        else newType = among(ftype, "IQ2_S", "IQ2_M") ? "IQ3_S" : "Q2_K";
        qs.i_attention_wv += 1;
      } else if (nExpertH === 8 && cat === "ATTENTION_K") {
        newType = "Q4_K";
      } else if (cat === "FFN_DOWN") {
        if (qs.i_ffn_down < fdiv(qs.n_ffn_down, 8)) newType = among(ftype, "IQ2_S", "IQ2_M") ? "IQ3_S" : "Q2_K";
        qs.i_ffn_down += 1;
      } else if (cat === "ATTENTION_OUTPUT") {
        if (nExpertH === 8) newType = "Q5_K";
        else if (among(ftype, "IQ1_S", "IQ1_M")) newType = "IQ2_XXS";
        else if (among(ftype, "IQ2_S", "IQ2_M")) newType = "IQ3_S";
      }
    } else if (isAttnV(cat)) {
      if (ftype === "Q2_K") newType = gqa >= 4 ? "Q4_K" : "Q3_K";
      else if (ftype === "Q2_K_S" && gqa >= 4) newType = "Q4_K";
      else if (ftype === "IQ3_XXS") newType = gqa >= 4 ? "Q4_K" : (!qs.has_imatrix ? "IQ3_S" : "IQ3_XXS");
      else if (among(ftype, "IQ3_XS", "IQ3_S") && gqa >= 4) newType = "Q4_K";
      else if (ftype === "IQ3_M") newType = "Q4_K";
      else if (ftype === "Q3_K_M") newType = qs.i_attention_wv < 2 ? "Q5_K" : "Q4_K";
      else if (ftype === "Q3_K_L") newType = "Q5_K";
      else if (among(ftype, "IQ4_NL", "IQ4_XS") && gqa >= 4) newType = "Q5_K";
      else if (among(ftype, "Q4_K_M", "Q5_K_M") && useMoreBits(qs.i_attention_wv, qs.n_attention_wv)) newType = "Q6_K";
      else if (ftype === "Q4_K_S" && qs.i_attention_wv < 4) newType = "Q5_K";
      if (qs.is_70b) { if (among(newType, "Q3_K", "Q4_K")) newType = "Q5_K"; }
      if (nExpertH === 8) newType = "Q8_0";
      qs.i_attention_wv += 1;
    } else if (cat === "ATTENTION_K") {
      if (nExpertH === 8) newType = "Q8_0";
      else if (ftype === "IQ3_XS") newType = "IQ3_XXS";
      else if (ftype === "IQ3_XXS") newType = "IQ2_S";
    } else if (cat === "ATTENTION_Q") {
      if (ftype === "IQ3_XS") newType = "IQ3_XXS";
      else if (ftype === "IQ3_XXS") newType = "IQ2_S";
    } else if (cat === "FFN_DOWN") {
      const [iLayer, nLayer] = layerInfo(qs.i_ffn_down, qs.n_ffn_down);
      if (ftype === "Q2_K") newType = "Q3_K";
      else if (ftype === "Q2_K_S") { if (iLayer < fdiv(nLayer, 8)) newType = "Q4_K"; }
      else if (ftype === "IQ3_XXS" && !qs.has_imatrix) newType = iLayer < fdiv(nLayer, 8) ? "Q4_K" : "Q3_K";
      else if (ftype === "Q3_K_M") {
        newType = iLayer < fdiv(nLayer, 16) ? "Q5_K"
          : (arch !== "falcon" || useMoreBits(iLayer, nLayer)) ? "Q4_K" : "Q3_K";
      }
      else if (ftype === "IQ3_M" && (iLayer < fdiv(nLayer, 8) || (nExpertH === 8 && useMoreBits(iLayer, nLayer)))) newType = "Q4_K";
      else if (ftype === "Q3_K_L") newType = arch === "falcon" ? "Q4_K" : "Q5_K";
      else if (ftype === "Q4_K_M") {
        if (arch === "falcon") {
          newType = iLayer < fdiv(nLayer, 16) ? "Q6_K" : useMoreBits(iLayer, nLayer) ? "Q5_K" : "Q4_K";
        } else if (useMoreBits(iLayer, nLayer)) newType = "Q6_K";
      }
      else if (iLayer < fdiv(nLayer, 8) && among(ftype, "IQ4_NL", "IQ4_XS") && !qs.has_imatrix) newType = "Q5_K";
      else if (ftype === "Q5_K_M" && useMoreBits(iLayer, nLayer)) newType = "Q6_K";
      else if (ftype === "Q4_K_S" && arch !== "falcon" && iLayer < fdiv(nLayer, 8)) newType = "Q5_K";
      else if (among(ftype, "Q4_0", "Q5_0") && qs.has_imatrix && iLayer < fdiv(nLayer, 8)) newType = ftype === "Q4_0" ? "Q4_1" : "Q5_1";
      qs.i_ffn_down += 1;
    } else if (cat === "ATTENTION_OUTPUT") {
      if (arch !== "falcon") {
        if (nExpertH === 8) {
          if (among(ftype, "Q2_K", "IQ3_XS", "IQ3_XXS", "Q3_K_S", "Q3_K_M", "IQ4_NL", "Q4_K_S", "Q4_K_M",
                    "IQ3_S", "IQ3_M", "IQ4_XS")) newType = "Q5_K";
        } else {
          if (ftype === "Q2_K") newType = "Q3_K";
          else if (ftype === "IQ3_XXS") newType = "IQ3_S";
          else if (ftype === "Q3_K_M") newType = "Q4_K";
          else if (ftype === "Q3_K_L") newType = "Q5_K";
          else if (ftype === "IQ3_M") newType = "Q4_K";
        }
      } else {
        if (ftype === "Q3_K_L") newType = "Q4_K";
      }
    } else if (cat === "ATTENTION_QKV") {
      // unreachable, as in llama.cpp: isAttnV(ATTENTION_QKV) takes it above
      if (among(ftype, "Q3_K_M", "Q3_K_L", "IQ3_M")) newType = "Q4_K";
      else if (ftype === "Q4_K_M") newType = "Q5_K";
      else if (ftype === "Q5_K_M") newType = "Q6_K";
    } else if (cat === "FFN_GATE") {
      const [iLayer, nLayer] = layerInfo(qs.i_ffn_gate, qs.n_ffn_gate);
      if (ftype === "IQ3_XS" && (iLayer >= fdiv(nLayer, 8) && iLayer < fdiv(7 * nLayer, 8))) newType = "IQ3_XXS";
      qs.i_ffn_gate += 1;
    } else if (cat === "FFN_UP") {
      const [iLayer, nLayer] = layerInfo(qs.i_ffn_up, qs.n_ffn_up);
      if (ftype === "IQ3_XS" && (iLayer >= fdiv(nLayer, 8) && iLayer < fdiv(7 * nLayer, 8))) newType = "IQ3_XXS";
      qs.i_ffn_up += 1;
    }
    return newType;
  }

  function fallback(qs, ne0, target) {
    const qk = T.types[target].block;
    let ret = target;
    if (ne0 % qk !== 0) {
      qs.n_fallback += 1;
      if (Object.prototype.hasOwnProperty.call(T.fallback, target)) ret = T.fallback[target];
      else if (qk <= 32) ret = target;
      else throw new Error("no tensor type fallback is defined for type " + target);
      if (ne0 % T.types[ret].block !== 0) ret = "F16";
    }
    return ret;
  }

  function is70b(arch, nLayer, nHead, nHeadKv, nExpert, nLayerNextn) {
    const n = nLayer - (nLayerNextn || 0);
    for (const r of T.type_70b) {
      if (r.arch !== arch || r.n_layer !== n) continue;
      if (r.not_n_expert !== null && (nExpert || 0) === r.not_n_expert) continue;
      if (r.cond === "gqa" && nHead !== null && nHead !== undefined && nHeadKv !== null && nHeadKv !== undefined
          && nHead === nHeadKv) continue;
      return true;
    }
    return false;
  }

  function tensorTypesImpl(tensors, ftype, arch, opts, stats) {
    let srcs = tensors.map(t => (t.src === undefined ? null : t.src));
    if (srcs.some(s => s === null)) {
      const st = sourceTypesImpl(tensors, arch, "bf16", {});
      srcs = srcs.map((s, i) => (s !== null ? s : st[i]));
    }
    if (ftype === null || ftype === undefined) return srcs;
    if (!Object.prototype.hasOwnProperty.call(T.ftypes, ftype)) throw new Error("invalid output file type " + ftype);
    const defaultType = T.ftypes[ftype];
    const nExpert = opt(opts, "n_expert", 0);
    let nGqa = opt(opts, "n_gqa", 1);
    let nLayer = opt(opts, "n_layer", null);
    const nHead = opt(opts, "n_head", null), nHeadKv = opt(opts, "n_head_kv", null);
    const nLayerNextn = opt(opts, "n_layer_nextn", 0);
    let model70b = opt(opts, "model_70b", null);
    const hasImatrix = opt(opts, "has_imatrix", false);
    if (nLayer === null) {
      let mx = -1;
      for (const t of tensors) { const l = blkLayer(t.name); if (l !== null && l > mx) mx = l; }
      nLayer = 1 + mx;
    }
    if (nHead !== null && nHeadKv !== null) nGqa = nHeadKv === 0 ? 0 : fdiv(nHead, nHeadKv);
    if (model70b === null) model70b = is70b(arch, nLayer, nHead, nHeadKv, nExpert, nLayerNextn);
    const qs = {
      n_expert: nExpert, n_gqa: nGqa, is_70b: model70b, has_imatrix: hasImatrix,
      n_attention_wv: 0, n_ffn_down: nLayer, n_ffn_gate: nLayer, n_ffn_up: nLayer,
      i_attention_wv: 0, i_ffn_down: 0, i_ffn_gate: 0, i_ffn_up: 0, n_fallback: 0, has_tied_embeddings: true,
    };
    const names = tensors.map(t => t.name);
    const cats = names.map(tensorCategory);
    for (const c of cats) {
      if (isAttnV(c)) qs.n_attention_wv += 1;
      if (c === "OUTPUT") qs.has_tied_embeddings = false;
    }
    const order = names.map((_, i) => i).sort((a, b) => cmpWeights(names[a], names[b]) || (a - b));
    const out = new Array(tensors.length).fill(null);
    const quantizedDefault = T.types[defaultType].quantized;
    for (const i of order) {
      const shape = tensors[i].shape;
      if (!allowsQuantization(names[i], shape, arch)) { out[i] = srcs[i]; continue; }
      let newType = defaultType;
      if (quantizedDefault) {
        newType = getTypeImpl(qs, newType, names[i], shape[0], ftype, cats[i], arch);
        newType = fallback(qs, shape[0], newType);
      }
      out[i] = newType;
    }
    if (stats) stats.n_fallback = qs.n_fallback;
    return out;
  }

  function archForceF32(name, arch, flags) {
    const spec = T.converter.arch_f32[arch];
    if (!spec) return false;
    for (const r of spec.rules) {
      let ok = true;
      if (r.prefix) ok = ok && r.prefix.some(s => name.startsWith(s));
      if (r.contains) ok = ok && r.contains.some(s => name.includes(s));
      if (r.suffix) ok = ok && r.suffix.some(s => name.endsWith(s));
      if (r.equal) ok = ok && r.equal.includes(name);
      if (r.regex) ok = ok && r.regex.some(x => rx(x).test(name));
      if (r.flag) ok = ok && !!(flags && flags[r.flag]);
      if (ok) return true;
    }
    return false;
  }

  function sourceTypesImpl(tensors, arch, outtype, flags) {
    const C = T.converter;
    const base = { bf16: "BF16", f16: "F16", f32: "F32", q8_0: "Q8_0" }[outtype || "bf16"];
    if (!base) throw new Error("unknown outtype " + outtype);
    return tensors.map(t => {
      const name = t.name, shape = t.shape, n = shape.length;
      if (archForceF32(name, arch, flags || {})) return "F32";
      if (n <= 1 || name.endsWith(C.norm_suffix)) return "F32";
      if (C.requires_f32.some(r => rx(r.regex).test(name)) || !C.weight_suffixes.includes(name.slice(-7))) return "F32";
      if (base === "Q8_0" && shape[0] % T.types.Q8_0.block !== 0) return "F16";
      return base;
    });
  }

  function tensorBytes(type, nElements, ne0) {
    const t = T.types[type];
    if (ne0 <= 0 || nElements % ne0 !== 0) throw new Error("n_elements " + nElements + " is not a multiple of ne0 " + ne0);
    if (ne0 % t.block !== 0) throw new Error("ne0 " + ne0 + " is not a multiple of the " + type + " block " + t.block);
    return (nElements / ne0) * (t.bytes * (ne0 / t.block));
  }
  const pad = (n, a) => Math.ceil(n / a) * a;

  function breakdown(tensors, ftype, arch, headerBytes, opts) {
    tables();
    headerBytes = headerBytes || 0;
    const stats = {};
    const types = tensorTypesImpl(tensors, ftype, arch, opts, stats);
    const align = T.alignment;
    const byType = {}, byKind = {}, rows = [];
    let data = 0;
    tensors.forEach((t, i) => {
      const ty = types[i];
      const n = nElements(t.shape);
      const b = pad(tensorBytes(ty, n, t.shape[0]), align);
      data += b;
      const kind = tensorCategory(t.name).toLowerCase() + (isExpert(t.name) ? "_exps" : "");
      for (const [d, k] of [[byType, ty], [byKind, kind]]) {
        if (!Object.prototype.hasOwnProperty.call(d, k)) d[k] = { tensors: 0, elements: 0, bytes: 0 };
        d[k].tensors += 1; d[k].elements += n; d[k].bytes += b;
      }
      rows.push({ name: t.name, type: ty, bytes: b });
    });
    return { ftype: ftype === undefined ? null : ftype, header: headerBytes, data, total: headerBytes + data,
      n_fallback: stats.n_fallback || 0, by_type: byType, by_kind: byKind, tensors: rows };
  }

  // the header (ggml/src/gguf.cpp gguf_write_out); see gguf_header_bytes in the Python for what is estimated
  const SCALAR = { u8: 1, i8: 1, bool: 1, u16: 2, i16: 2, u32: 4, i32: 4, f32: 4, u64: 8, i64: 8, f64: 8 };
  function kvBytes(key, kind, value) {
    const n = 8 + utf8len(key) + 4;
    if (kind === "str") return n + 8 + value;
    if (kind.startsWith("arr:")) {
      const el = kind.slice(4), [count, per] = value;
      return n + 4 + 8 + (el === "str" ? count * (8 + per) : count * SCALAR[el]);
    }
    return n + SCALAR[kind];
  }
  function headerKvs(vocabSize, o) {
    const arch = opt(o, "arch", "llama"), tokenizer = opt(o, "tokenizer", "gpt2");
    const avgTok = opt(o, "avg_token_bytes", 7.0), nameBytes = opt(o, "name_bytes", 16);
    const a = arch;
    const kvs = [["general.architecture", "str", utf8len(arch)], ["general.type", "str", 5],
      ["general.name", "str", nameBytes], ["general.basename", "str", nameBytes],
      ["general.size_label", "str", 3], ["general.file_type", "u32", 0],
      ["general.quantization_version", "u32", 0],
      [a + ".block_count", "u32", 0], [a + ".context_length", "u32", 0], [a + ".embedding_length", "u32", 0],
      [a + ".feed_forward_length", "u32", 0], [a + ".attention.head_count", "u32", 0],
      [a + ".attention.head_count_kv", "u32", 0], [a + ".rope.freq_base", "f32", 0],
      [a + ".attention.layer_norm_rms_epsilon", "f32", 0], [a + ".attention.key_length", "u32", 0],
      [a + ".attention.value_length", "u32", 0], [a + ".vocab_size", "u32", 0],
      [a + ".rope.dimension_count", "u32", 0]];
    if (opt(o, "moe", false)) kvs.push([a + ".expert_count", "u32", 0], [a + ".expert_used_count", "u32", 0]);
    kvs.push(["tokenizer.ggml.model", "str", utf8len(tokenizer)], ["tokenizer.ggml.pre", "str", 8],
      ["tokenizer.ggml.tokens", "arr:str", [vocabSize, avgTok]],
      ["tokenizer.ggml.token_type", "arr:i32", [vocabSize, 0]]);
    if (tokenizer === "gpt2") {
      const nm = opt(o, "n_merges", vocabSize - 256);
      const per = opt(o, "avg_merge_bytes", avgTok + 1);
      kvs.push(["tokenizer.ggml.merges", "arr:str", [nm, per]]);
    } else {
      kvs.push(["tokenizer.ggml.scores", "arr:f32", [vocabSize, 0]]);
    }
    kvs.push(["tokenizer.ggml.bos_token_id", "u32", 0], ["tokenizer.ggml.eos_token_id", "u32", 0],
      ["tokenizer.ggml.padding_token_id", "u32", 0], ["tokenizer.ggml.add_bos_token", "bool", 0]);
    const ct = opt(o, "chat_template_bytes", 0);
    if (ct) kvs.push(["tokenizer.chat_template", "str", ct]);
    for (const kv of opt(o, "extra_kv", [])) kvs.push(kv);
    return kvs;
  }
  function headerBytes(vocabSize, o) {
    tables();
    o = o || {};
    const align = opt(o, "alignment", T.alignment);
    let total = 4 + 4 + 8 + 8;
    for (const [k, kind, v] of headerKvs(vocabSize, o)) total += kvBytes(k, kind, v);
    const tensors = opt(o, "tensors", null);
    if (tensors !== null) {
      for (const t of tensors) total += 8 + utf8len(t.name) + 4 + 8 * nDims(t.shape) + 4 + 8;
    } else {
      total += (opt(o, "n_tensors", 0) || 0) * (8 + opt(o, "avg_name_bytes", 24) + 4 + 8 * 2 + 4 + 8);
    }
    total = Math.ceil(total);
    return pad(total, align);
  }

  const API = {
    setTables,
    tensorTypes: (tensors, ftype, arch, opts) => { tables(); return tensorTypesImpl(tensors, ftype, arch, opts || {}, null); },
    sourceTypes: (tensors, arch, outtype, flags) => { tables(); return sourceTypesImpl(tensors, arch, outtype, flags); },
    tensorBytes: (type, nElements, ne0) => { tables(); return tensorBytes(type, nElements, ne0); },
    ggufBytes: (tensors, ftype, arch, hb, opts) => breakdown(tensors, ftype, arch, hb, opts).total,
    breakdown,
    headerBytes,
    headerKvs: (vocabSize, o) => headerKvs(vocabSize, o || {}),
    tensorCategory,
    allowsQuantization: (name, shape, arch) => { tables(); return allowsQuantization(name, shape, arch); },
    is70b: (arch, nLayer, nHead, nHeadKv, nExpert, nLayerNextn) => { tables(); return is70b(arch, nLayer, nHead, nHeadKv, nExpert, nLayerNextn); },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.GgufQuant = API;
})(typeof window !== "undefined" ? window : globalThis);
