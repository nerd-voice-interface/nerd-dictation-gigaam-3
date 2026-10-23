#!/usr/bin/env python3
"""Turn the upstream GigaAM v3 CTC ONNX model into the one this backend ships.

Every rewrite is exact: it computes the same values, only through kernels
that run faster on the target GPU (a Pascal Quadro P500; see
~/forge/gigaam-profile/REPORT.md for the profiling behind each pass).

  conv1x1   A pointwise Conv1d (kernel 1, stride 1, no padding, one group)
            on [B, C, T] is MatMul(W[O, C], X) + bias[O, 1]: the same
            products and sums, through cuBLAS GEMM instead of cuDNN's
            implicit-GEMM convolution.
  halfscale The macaron feed-forward modules end in "x + 0.5 * (h W2 + b2)".
            0.5 is a power of two, so scaling W2 and b2 by it is exact and
            the Mul kernel goes away.

  gemm      MatMul(X[B, T, K], W) followed by Add(bias[N]) runs as two
            kernels: onnxruntime fuses MatMul+Add into Gemm only for 2-D
            inputs. Reshape X to [B*T, K] (free on the GPU), Gemm with the
            bias, reshape back to [B, T, N]: the bias goes into cuBLAS's
            epilogue (beta = 1), the Add kernel goes away.

Usage: optimize_model.py UPSTREAM.onnx OUT.onnx [--passes conv1x1,halfscale]
"""
import argparse
import collections
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

PASSES = ("conv1x1", "halfscale", "gemm")


class Graph:
    def __init__(self, model):
        self.model = model
        self.g = model.graph
        self.init = {t.name: t for t in self.g.initializer}

    def const(self, name):
        """Value of an initializer, else None."""
        t = self.init.get(name)
        return None if t is None else numpy_helper.to_array(t)

    def add_init(self, array, name):
        t = numpy_helper.from_array(np.ascontiguousarray(array), name)
        self.g.initializer.append(t)
        self.init[name] = t
        return name

    def consumers(self):
        c = collections.defaultdict(list)
        for n in self.g.node:
            for i in n.input:
                c[i].append(n)
        return c

    def replace_nodes(self, nodes):
        del self.g.node[:]
        self.g.node.extend(nodes)

    def drop_unused_initializers(self):
        used = {i for n in self.g.node for i in n.input}
        keep = [t for t in self.g.initializer if t.name in used]
        del self.g.initializer[:]
        self.g.initializer.extend(keep)
        self.init = {t.name: t for t in keep}


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def conv1x1(gr):
    out, done = [], 0
    for node in gr.g.node:
        w = gr.const(node.input[1]) if node.op_type == "Conv" else None
        a = attrs(node) if w is not None else {}
        if (w is not None and w.ndim == 3 and w.shape[2] == 1
                and a.get("group", 1) == 1
                and list(a.get("strides", [1])) == [1]
                and list(a.get("pads", [0, 0])) == [0, 0]
                and list(a.get("dilations", [1])) == [1]
                and a.get("auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET")):
            w2 = gr.add_init(w[:, :, 0], node.name + "_w2d")
            has_bias = len(node.input) > 2 and node.input[2]
            mm_out = node.output[0] + "_mm" if has_bias else node.output[0]
            out.append(helper.make_node("MatMul", [w2, node.input[0]], [mm_out],
                                        name=node.name + "_matmul"))
            if has_bias:
                b = gr.add_init(gr.const(node.input[2]).reshape(-1, 1), node.name + "_b2d")
                out.append(helper.make_node("Add", [mm_out, b], [node.output[0]],
                                            name=node.name + "_bias"))
            done += 1
        else:
            out.append(node)
    gr.replace_nodes(out)
    return done


def halfscale(gr):
    """Mul(Add(MatMul(h, W2), b2), 0.5) -> Add(MatMul(h, W2/2), b2/2)."""
    prod = {o: n for n in gr.g.node for o in n.output}
    cons = gr.consumers()
    renamed, drop, done = {}, set(), 0
    for mul in gr.g.node:
        if mul.op_type != "Mul":
            continue
        scalars = [i for i in mul.input if (v := gr.const(i)) is not None
                   and v.size == 1 and float(v.reshape(-1)[0]) == 0.5]
        if len(scalars) != 1:
            continue
        x = [i for i in mul.input if i != scalars[0]][0]
        add = prod.get(x)
        if add is None or add.op_type != "Add" or len(cons[add.output[0]]) != 1:
            continue
        bias = [i for i in add.input if (v := gr.const(i)) is not None and v.ndim == 1]
        mm = [prod.get(i) for i in add.input if i not in bias]
        if len(bias) != 1 or len(mm) != 1 or mm[0] is None or mm[0].op_type != "MatMul":
            continue
        mm = mm[0]
        w = gr.const(mm.input[1])
        if w is None or len(cons[mm.output[0]]) != 1:
            continue
        # The weights and the bias may be shared with another node: write new ones.
        mm.input[1] = gr.add_init(w * np.float32(0.5), mm.name + "_w_half")
        add.input[list(add.input).index(bias[0])] = gr.add_init(
            gr.const(bias[0]) * np.float32(0.5), add.name + "_b_half")
        renamed[mul.output[0]] = add.output[0]
        drop.add(id(mul))
        done += 1
    gr.replace_nodes([n for n in gr.g.node if id(n) not in drop])
    for n in gr.g.node:
        for k, i in enumerate(n.input):
            if i in renamed:
                n.input[k] = renamed[i]
    for o in gr.g.output:
        if o.name in renamed:
            raise SystemExit("halfscale: a removed Mul feeds a graph output")
    return done


def gemm(gr):
    prod = {o: n for n in gr.g.node for o in n.output}
    cons = gr.consumers()
    out, done = [], 0
    # One back-shape per distinct (X, N): [B, T] of X followed by N.
    shapes = {}
    k_shape = {}
    for node in gr.g.node:
        out.append(node)
    replace = {}
    for mm in gr.g.node:
        if mm.op_type != "MatMul":
            continue
        w = gr.const(mm.input[1])
        cs = cons[mm.output[0]]
        if w is None or w.ndim != 2 or len(cs) != 1 or cs[0].op_type != "Add":
            continue
        add = cs[0]
        bias = [i for i in add.input if (v := gr.const(i)) is not None
                and v.ndim == 1 and v.shape[0] == w.shape[1]]
        if len(bias) != 1:
            continue
        x, (k, n) = mm.input[0], w.shape
        base = mm.name or mm.output[0]
        if k not in k_shape:
            k_shape[k] = gr.add_init(np.array([-1, k], np.int64), f"gemm_shape_flat_{k}")
        nodes = [helper.make_node("Reshape", [x, k_shape[k]], [base + "_x2d"],
                                  name=base + "_flatten")]
        if (x, n) not in shapes:
            lead = shapes.get(x)
            if lead is None:
                nodes += [
                    helper.make_node("Shape", [x], [base + "_xshape"], name=base + "_shape"),
                    helper.make_node("Slice", [base + "_xshape",
                                               gr.add_init(np.array([0], np.int64), base + "_s0"),
                                               gr.add_init(np.array([-1], np.int64), base + "_s1")],
                                     [base + "_lead"], name=base + "_lead"),
                ]
                lead = shapes[x] = base + "_lead"
            nodes.append(helper.make_node(
                "Concat", [lead, gr.add_init(np.array([n], np.int64), base + "_n")],
                [base + "_yshape"], axis=0, name=base + "_yshape"))
            shapes[(x, n)] = base + "_yshape"
        nodes += [
            helper.make_node("Gemm", [base + "_x2d", mm.input[1], bias[0]], [base + "_y2d"],
                             name=base + "_gemm"),
            helper.make_node("Reshape", [base + "_y2d", shapes[(x, n)]], [add.output[0]],
                             name=base + "_unflatten"),
        ]
        replace[id(mm)] = nodes
        replace[id(add)] = []
        done += 1
    new = []
    for node in out:
        new.extend(replace.get(id(node), [node]))
    gr.replace_nodes(new)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--passes", default=",".join(PASSES))
    args = ap.parse_args()
    passes = [p for p in args.passes.split(",") if p]
    unknown = set(passes) - set(PASSES)
    if unknown:
        sys.exit(f"unknown passes: {', '.join(sorted(unknown))}")
    gr = Graph(onnx.load(args.src))
    for p in passes:
        print(f"{p}: {globals()[p](gr)} rewritten")
    gr.drop_unused_initializers()
    onnx.checker.check_model(gr.model, full_check=False)
    meta = gr.model.metadata_props.add()
    meta.key, meta.value = "nerd_dictation_gigaam_passes", ",".join(passes)
    onnx.save(gr.model, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
