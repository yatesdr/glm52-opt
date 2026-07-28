import json, random, string, urllib.request
NEEDLE="738216"
NEEDLE_SENT=("For reference, the maintenance ticket number for the Facility 27 "
             f"compressor overhaul is {NEEDLE}. ")
FILLER=("The consolidation entries for the period were reviewed by the plant "
        "controllers before submission. Intercompany balances cleared without "
        "manual adjustment, and the hedging position rolled forward unchanged. "
        "Tooling amortization followed the agreed schedule, while indirect "
        "spend stayed within the corridor set at the last quarterly review. ")
QUESTION=("\n\nFrom the document above: what is the maintenance ticket number for the "
          "Facility 27 compressor overhaul? Reply with the number only.")
MODEL="GLM-5.2"; BASE="http://localhost:5001"

def post(p,pl,t=300):
    r=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
                             headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
def ntok(s): return post("/tokenize",{"model":MODEL,"prompt":s})["count"]

per=ntok(FILLER)
for label,depth,seed in [("ladder 350k rep1",350000,760000+350000+1),
                         ("ladder 350k rep2",350000,760000+350000+2),
                         ("earlier PASSING",350000,20260725)]:
    rnd=random.Random(seed)
    pkt="".join(rnd.choices(string.ascii_uppercase+string.digits,k=4))+"-"+"".join(rnd.choices(string.digits,k=6))
    head=(f"Review packet {pkt} for the period ending 2026-07-19, prepared by controller "
          f"{rnd.randint(10,99)} of the Facility {rnd.randint(2,26)} consolidation group. ")
    blocks=max(4,(depth-ntok(head+QUESTION+NEEDLE_SENT))//per)
    pre=int(blocks*0.40)
    prompt=head+FILLER*pre+NEEDLE_SENT+FILLER*(blocks-pre)+QUESTION
    ctx=ntok(prompt)
    idx=prompt.find(NEEDLE)
    pre_tok=ntok(prompt[:idx]) if idx>=0 else -1
    print(f"{label:20} seed={seed} ctx={ctx} needle_count={prompt.count(NEEDLE)} "
          f"char_idx={idx} tok_before_needle={pre_tok} depth_pct={100*pre_tok/ctx:.1f}%")
    print(f"{'':20} head={head[:90]!r}")
