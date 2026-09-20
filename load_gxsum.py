"""Load GXSum with source documents attached.

GXSum ships only (id, summary). The source articles are BBC content and are not
redistributed; this helper joins them back in from EdinburghNLP/xsum by BBC id,
returning the same {"id", "document", "summary"} rows the paper's code expects.

    from load_gxsum import load_gxsum
    ds = load_gxsum("small")            # DatasetDict with train / validation / test
    ds = load_gxsum("large", split="test")

Both datasets are public; no token is needed.
"""
from datasets import load_dataset


def load_gxsum(config="small", split=None):
    gx = load_dataset("GlycerinLOL/GXSum", config, split=split)
    xs = load_dataset("EdinburghNLP/xsum")
    id2doc = {r["id"]: r["document"] for s in xs for r in xs[s]}

    def attach(batch):
        return {"document": [id2doc[i] for i in batch["id"]]}

    return gx.map(attach, batched=True)


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "small"
    ds = load_gxsum(cfg, split="test")
    print(ds)
    print(ds[0])
