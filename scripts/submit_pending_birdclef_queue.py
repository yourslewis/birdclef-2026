"""Submit pending completed BirdCLEF kernels in order, with quota retry."""
import json, os, re, time
from kaggle.api.kaggle_api_extended import KaggleApi

version=int(os.environ.get("KAGGLE_KERNEL_VERSION","1"))
PENDING=[
    {"name":"v247","kernel":"yourslewis/birdclef-2026-v247-minimal-temporal-smoothing","version":version,"message":"v247: v245/v246 axis + minimal temporal smoothing center 0.90"},
    {"name":"v248","kernel":"yourslewis/birdclef-2026-v248-v245-gamma080","version":version,"message":"v248: v245 smoothing + power gamma 0.80"},
    {"name":"v249","kernel":"yourslewis/birdclef-2026-v249-v245-gamma090","version":version,"message":"v249: v245 smoothing + power gamma 0.90"},
    {"name":"v250","kernel":"yourslewis/birdclef-2026-v250-immediate-temporal-smoothing","version":version,"message":"v250: v245 center mass + immediate-only temporal smoothing"},
    {"name":"v251","kernel":"yourslewis/birdclef-2026-v251-v245-context015","version":version,"message":"v251: v245 smoothing + gentler file context alpha 0.15"},
    {"name":"v252","kernel":"yourslewis/birdclef-2026-v252-v245-context010","version":version,"message":"v252: v245 smoothing + lighter file context alpha 0.10"},
    {"name":"v253","kernel":"yourslewis/birdclef-2026-v253-v245-context000","version":version,"message":"v253: v245 smoothing + no file context boost"},
    {"name":"v254","kernel":"yourslewis/birdclef-2026-v254-v245-context005","version":version,"message":"v254: v245 smoothing + very light file context alpha 0.05"},
    {"name":"v255","kernel":"yourslewis/birdclef-2026-v255-v245-topk-contrast","version":version,"message":"v255: v245 smoothing + light top-k confidence contrast"},
    {"name":"v256","kernel":"yourslewis/birdclef-2026-v256-v245-strong-topk-contrast","version":version,"message":"v256: v245 smoothing + stronger top-k confidence contrast"},
    {"name":"v257","kernel":"yourslewis/birdclef-2026-v257-v245-tail-damp","version":version,"message":"v257: v245 smoothing + low-rank tail dampening"},
    {"name":"v258","kernel":"yourslewis/birdclef-2026-v258-v245-strong-tail-damp","version":version,"message":"v258: v245 smoothing + stronger low-rank tail dampening"},
    {"name":"v259","kernel":"yourslewis/birdclef-2026-v259-v245-gentle-tail-damp","version":version,"message":"v259: v245 smoothing + gentle low-rank tail dampening"},
    {"name":"v260","kernel":"yourslewis/birdclef-2026-v260-v245-quantile-alpha055","version":version,"message":"v260: v245 smoothing + quantile mix alpha 0.55"},
    {"name":"v261","kernel":"yourslewis/birdclef-2026-v261-v245-quantile-alpha045","version":version,"message":"v261: v245 smoothing + quantile mix alpha 0.45"},
    {"name":"v262","kernel":"yourslewis/birdclef-2026-v262-v245-quantile-alpha0525","version":version,"message":"v262: v245 smoothing + quantile mix alpha 0.525"},
    {"name":"v263","kernel":"yourslewis/birdclef-2026-v263-v245-protossm-ew055","version":version,"message":"v263: v245 smoothing + ProtoSSM ensemble weight 0.55"},
    {"name":"v264","kernel":"yourslewis/birdclef-2026-v264-v245-protossm-ew050","version":version,"message":"v264: v245 smoothing + ProtoSSM ensemble weight 0.50"},
    {"name":"v265","kernel":"yourslewis/birdclef-2026-v265-v245-gamma0875","version":version,"message":"v265: v245 smoothing + power gamma 0.875"},
    {"name":"v266","kernel":"yourslewis/birdclef-2026-v266-v245-gamma0825","version":version,"message":"v266: v245 smoothing + power gamma 0.825"},
    {"name":"v267","kernel":"yourslewis/birdclef-2026-v267-v245-temporal075","version":version,"message":"v267: v245 family + intermediate temporal smoothing center 0.75"},
    {"name":"v268","kernel":"yourslewis/birdclef-2026-v268-v245-context025","version":version,"message":"v268: v245 smoothing + stronger file context alpha 0.25"},
    {"name":"v269","kernel":"yourslewis/birdclef-2026-v269-immediate-temporal075","version":version,"message":"v269: immediate-only temporal smoothing center 0.75"},
    {"name":"v270","kernel":"yourslewis/birdclef-2026-v270-immediate-gamma080","version":version,"message":"v270: immediate-only temporal smoothing + power gamma 0.80"},
    {"name":"v271","kernel":"yourslewis/birdclef-2026-v271-immediate-gamma090","version":version,"message":"v271: immediate-only temporal smoothing + power gamma 0.90"},
    {"name":"v272","kernel":"yourslewis/birdclef-2026-v272-immediate-quantile055","version":version,"message":"v272: immediate-only temporal smoothing + quantile mix alpha 0.55"},
    {"name":"v273","kernel":"yourslewis/birdclef-2026-v273-immediate-quantile045","version":version,"message":"v273: immediate-only temporal smoothing + quantile mix alpha 0.45"},
    {"name":"v274","kernel":"yourslewis/birdclef-2026-v274-immediate-quantile0525","version":version,"message":"v274: immediate-only temporal smoothing + quantile mix alpha 0.525"},
    {"name":"v275","kernel":"yourslewis/birdclef-2026-v275-immediate-quantile0475","version":version,"message":"v275: immediate-only temporal smoothing + quantile mix alpha 0.475"},
    {"name":"v276","kernel":"yourslewis/birdclef-2026-v276-immediate-protossm-ew055","version":version,"message":"v276: immediate-only temporal smoothing + ProtoSSM ensemble weight 0.55"},
    {"name":"v277","kernel":"yourslewis/birdclef-2026-v277-immediate-protossm-ew050","version":version,"message":"v277: immediate-only temporal smoothing + ProtoSSM ensemble weight 0.50"},
    {"name":"v278","kernel":"yourslewis/bc26-v278-immediate-selective-context","version":version,"message":"v278: immediate-only temporal smoothing + selective top-k file context boost"},
    {"name":"v279","kernel":"yourslewis/bc26-v279-selective-context-top16","version":version,"message":"v279: immediate-only temporal smoothing + selective top-16 file context boost"},
    {"name":"v280","kernel":"yourslewis/bc26-v280-selective-context-top8","version":version,"message":"v280: immediate-only temporal smoothing + selective top-8 file context boost"},
    {"name":"v281","kernel":"yourslewis/bc26-v281-file-mean-blend","version":version,"message":"v281: immediate-only temporal smoothing + file mean blend context"},
    {"name":"v282","kernel":"yourslewis/bc26-v282-file-mean-alpha010","version":version,"message":"v282: immediate-only temporal smoothing + file mean blend alpha 0.10"},
    {"name":"v283","kernel":"yourslewis/bc26-v283-file-mean-alpha015","version":version,"message":"v283: immediate-only temporal smoothing + file mean blend alpha 0.15"},
    {"name":"v284","kernel":"yourslewis/bc26-v284-file-mean-alpha005","version":version,"message":"v284: immediate-only temporal smoothing + file mean blend alpha 0.05"},
    {"name":"v285","kernel":"yourslewis/bc26-v285-prob-temporal-smooth","version":version,"message":"v285: immediate temporal smoothing after sigmoid in probability space"},
    {"name":"v286","kernel":"yourslewis/bc26-v286-prob-smooth-mean010","version":version,"message":"v286: probability temporal smoothing + file mean blend alpha 0.10"},
    {"name":"v287","kernel":"yourslewis/bc26-v287-prob-smooth-q0475","version":version,"message":"v287: probability temporal smoothing + quantile mix alpha 0.475"},
    {"name":"v288","kernel":"yourslewis/bc26-v288-prob-smooth-q0525","version":version,"message":"v288: probability temporal smoothing + quantile mix alpha 0.525"},
]
with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f: os.environ["KAGGLE_API_TOKEN"]=json.load(f)["key"]
api=KaggleApi(); api.authenticate()
def recent_messages(): return {str(getattr(s,"description","")) for s in api.competition_submissions("birdclef-2026")[:50]}
def is_complete(kernel):
    status=api.kernels_status(kernel); print(f"Kernel status {kernel}: {status}", flush=True); return "COMPLETE" in str(getattr(status,"status",status)).upper()
def quota_sleep_seconds(text):
    m=re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m: return max(300,int(float(m.group(1))*3600)+120)
    m=re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m: return max(300,int(m.group(1))*60+120)
    return 3600
while True:
    messages=recent_messages(); progressed=False; all_done=True
    for item in PENDING:
        if item["message"] in messages:
            print(f"{item['name']} already submitted; skipping.", flush=True); continue
        all_done=False
        if not is_complete(item["kernel"]):
            print(f"{item['name']} not complete yet; sleeping 10 minutes.", flush=True); time.sleep(600); progressed=True; break
        print(f"Submitting {item['name']} kernel version {item['version']}...", flush=True)
        try:
            res=api.competition_submit_code(file_name="submission.csv", message=item["message"], competition="birdclef-2026", kernel=item["kernel"], kernel_version=item["version"]); print("Submission result:", res, flush=True); progressed=True; time.sleep(30); break
        except Exception as exc:
            response=getattr(exc,"response",None); text=getattr(response,"text","") if response is not None else ""
            print(f"Submission attempt failed for {item['name']}: {type(exc).__name__}: {exc}", flush=True)
            if text: print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                sleep_s=quota_sleep_seconds(text); print(f"Daily submission allowance exhausted; sleeping {sleep_s} seconds before retry.", flush=True); time.sleep(sleep_s); progressed=True; break
            raise
    if all_done:
        print("All pending kernels are already submitted.", flush=True); break
    if not progressed: time.sleep(600)
