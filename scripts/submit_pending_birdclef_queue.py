"""Submit pending completed BirdCLEF kernels in order, with quota retry."""
import json, os, re, time
import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

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
    {"name":"v362","kernel":"yourslewis/bc26-v362-ew0675-g0825-ctx025","version":version,"message":"v362: immediate top-k + ProtoSSM EW0.675 + gamma 0.825 + context alpha 0.25"},
    {"name":"v366","kernel":"yourslewis/bc26-v366-ew055-g0825-ctx025","version":version,"message":"v366: immediate top-k + ProtoSSM EW0.55 + gamma 0.825 + context alpha 0.25"},
    {"name":"v370","kernel":"yourslewis/bc26-v370-ew0575-g0825-ctx025","version":version,"message":"v370: immediate top-k + ProtoSSM EW0.575 + gamma 0.825 + context alpha 0.25"},
    {"name":"v367","kernel":"yourslewis/bc26-v367-ew060-g0825-ctx025","version":version,"message":"v367: immediate top-k + ProtoSSM EW0.60 + gamma 0.825 + context alpha 0.25"},
    {"name":"v368","kernel":"yourslewis/bc26-v368-ew0625-g0825-ctx025","version":version,"message":"v368: immediate top-k + ProtoSSM EW0.625 + gamma 0.825 + context alpha 0.25"},
    {"name":"v371","kernel":"yourslewis/bc26-v371-ew0575-g080-ctx025","version":version,"message":"v371: immediate top-k + ProtoSSM EW0.575 + gamma 0.80 + context alpha 0.25"},
    {"name":"v372","kernel":"yourslewis/bc26-v372-ew0575-g0875-ctx025","version":version,"message":"v372: immediate top-k + ProtoSSM EW0.575 + gamma 0.875 + context alpha 0.25"},
    {"name":"v369","kernel":"yourslewis/bc26-v369-ew065-g0825-ctx025","version":version,"message":"v369: immediate top-k + ProtoSSM EW0.65 + gamma 0.825 + context alpha 0.25"},
    {"name":"v364","kernel":"yourslewis/bc26-v364-ew0675-g0825-ctx0275","version":version,"message":"v364: immediate top-k + ProtoSSM EW0.675 + gamma 0.825 + context alpha 0.275"},
    {"name":"v363","kernel":"yourslewis/bc26-v363-ew0675-g0825-ctx030","version":version,"message":"v363: immediate top-k + ProtoSSM EW0.675 + gamma 0.825 + context alpha 0.30"},
    {"name":"v365","kernel":"yourslewis/bc26-v365-ew0675-g0825-ctx0225","version":version,"message":"v365: immediate top-k + ProtoSSM EW0.675 + gamma 0.825 + context alpha 0.225"},
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
    {"name":"v289","kernel":"yourslewis/bc26-v289-prob-smooth-gamma080","version":version,"message":"v289: probability temporal smoothing + power gamma 0.80"},
    {"name":"v290","kernel":"yourslewis/bc26-v290-prob-smooth-gamma090","version":version,"message":"v290: probability temporal smoothing + power gamma 0.90"},
    {"name":"v291","kernel":"yourslewis/bc26-v291-prob-smooth-gamma0825","version":version,"message":"v291: probability temporal smoothing + power gamma 0.825"},
    {"name":"v292","kernel":"yourslewis/bc26-v292-prob-smooth-gamma0875","version":version,"message":"v292: probability temporal smoothing + power gamma 0.875"},
    {"name":"v293","kernel":"yourslewis/bc26-v293-topk-gamma090","version":version,"message":"v293: v245 smoothing + strong top-k contrast + gamma 0.90"},
    {"name":"v294","kernel":"yourslewis/bc26-v294-topk-gamma080","version":version,"message":"v294: v245 smoothing + strong top-k contrast + gamma 0.80"},
    {"name":"v295","kernel":"yourslewis/bc26-v295-topk-gamma0825","version":version,"message":"v295: v245 smoothing + strong top-k contrast + gamma 0.825"},
    {"name":"v296","kernel":"yourslewis/bc26-v296-topk-gamma0875","version":version,"message":"v296: v245 smoothing + strong top-k contrast + gamma 0.875"},
    {"name":"v297","kernel":"yourslewis/bc26-v297-topk-k3","version":version,"message":"v297: v245 smoothing + focused top-k contrast K=3"},
    {"name":"v298","kernel":"yourslewis/bc26-v298-topk-k8","version":version,"message":"v298: v245 smoothing + broad top-k contrast K=8"},
    {"name":"v299","kernel":"yourslewis/bc26-v299-topk-k4","version":version,"message":"v299: v245 smoothing + top-k contrast K=4"},
    {"name":"v300","kernel":"yourslewis/bc26-v300-topk-k6","version":version,"message":"v300: v245 smoothing + top-k contrast K=6"},
    {"name":"v301","kernel":"yourslewis/bc26-v301-topk-k7","version":version,"message":"v301: v245 smoothing + top-k contrast K=7"},
    {"name":"v302","kernel":"yourslewis/bc26-v302-topk-stronger","version":version,"message":"v302: v245 smoothing + stronger top-k contrast K=5 p0.85/1.15"},
    {"name":"v303","kernel":"yourslewis/bc26-v303-topk-boost-only","version":version,"message":"v303: v245 smoothing + top-k boost-only K=5 p0.85/1.00"},
    {"name":"v304","kernel":"yourslewis/bc26-v304-topk-damp-only","version":version,"message":"v304: v245 smoothing + non-top damp-only K=5 p1.00/1.15"},
    {"name":"v305","kernel":"yourslewis/bc26-v305-immediate-topk","version":version,"message":"v305: immediate-only temporal smoothing + top-k contrast K=5"},
    {"name":"v306","kernel":"yourslewis/bc26-v306-immediate-boostonly","version":version,"message":"v306: immediate-only temporal smoothing + top-k boost-only K=5 p0.85/1.00"},
    {"name":"v307","kernel":"yourslewis/bc26-v307-immediate-damponly","version":version,"message":"v307: immediate-only temporal smoothing + non-top damp-only K=5 p1.00/1.15"},
    {"name":"v308","kernel":"yourslewis/bc26-v308-immediate-stronger-topk","version":version,"message":"v308: immediate-only temporal smoothing + stronger top-k contrast K=5 p0.85/1.15"},
    {"name":"v309","kernel":"yourslewis/bc26-v309-immediate-topk-alpha0475","version":version,"message":"v309: immediate top-k contrast + rank-heavy quantile alpha 0.475"},
    {"name":"v310","kernel":"yourslewis/bc26-v310-immediate-topk-alpha0525","version":version,"message":"v310: immediate top-k contrast + mean-heavy quantile alpha 0.525"},
    {"name":"v311","kernel":"yourslewis/bc26-v311-immediate-topk-alpha045","version":version,"message":"v311: immediate top-k contrast + rank-heavy quantile alpha 0.45"},
    {"name":"v312","kernel":"yourslewis/bc26-v312-immediate-topk-alpha055","version":version,"message":"v312: immediate top-k contrast + mean-heavy quantile alpha 0.55"},
    {"name":"v313","kernel":"yourslewis/bc26-v313-immediate-topk-gamma090","version":version,"message":"v313: immediate top-k contrast + power gamma 0.90"},
    {"name":"v314","kernel":"yourslewis/bc26-v314-immediate-topk-gamma080","version":version,"message":"v314: immediate top-k contrast + power gamma 0.80"},
    {"name":"v315","kernel":"yourslewis/bc26-v315-immediate-topk-gamma0825","version":version,"message":"v315: immediate top-k contrast + power gamma 0.825"},
    {"name":"v316","kernel":"yourslewis/bc26-v316-immediate-topk-gamma0875","version":version,"message":"v316: immediate top-k contrast + power gamma 0.875"},
    {"name":"v317","kernel":"yourslewis/bc26-v317-immediate-tail-damp","version":version,"message":"v317: immediate-only temporal smoothing + low-rank tail dampening K10 p1.05"},
    {"name":"v318","kernel":"yourslewis/bc26-v318-immediate-strong-tail-damp","version":version,"message":"v318: immediate-only temporal smoothing + stronger low-rank tail dampening K5 p1.10"},
    {"name":"v319","kernel":"yourslewis/bc26-v319-immediate-gentle-tail-damp","version":version,"message":"v319: immediate-only temporal smoothing + gentle low-rank tail dampening K20 p1.03"},
    {"name":"v320","kernel":"yourslewis/bc26-v320-immediate-topk-tail-damp","version":version,"message":"v320: immediate top-k contrast + gentle tail dampening K10 p1.05"},
    {"name":"v321","kernel":"yourslewis/bc26-v321-immediate-topk-strong-tail","version":version,"message":"v321: immediate top-k contrast + strong tail dampening K5 p1.10"},
    {"name":"v322","kernel":"yourslewis/bc26-v322-immediate-topk-gentle-tail","version":version,"message":"v322: immediate top-k contrast + gentle tail dampening K20 p1.03"},
    {"name":"v323","kernel":"yourslewis/bc26-v323-immediate-topk-k7","version":version,"message":"v323: immediate-only temporal smoothing + broader top-k contrast K=7"},
    {"name":"v324","kernel":"yourslewis/bc26-v324-immediate-topk-k3","version":version,"message":"v324: immediate-only temporal smoothing + focused top-k contrast K=3"},
    {"name":"v325","kernel":"yourslewis/bc26-v325-immediate-topk-k4","version":version,"message":"v325: immediate-only temporal smoothing + focused top-k contrast K=4"},
    {"name":"v326","kernel":"yourslewis/bc26-v326-immediate-topk-k6","version":version,"message":"v326: immediate-only temporal smoothing + broader top-k contrast K=6"},
    {"name":"v327","kernel":"yourslewis/bc26-v327-immediate-topk-context025","version":version,"message":"v327: immediate top-k contrast + stronger file context alpha 0.25"},
    {"name":"v328","kernel":"yourslewis/bc26-v328-immediate-topk-context030","version":version,"message":"v328: immediate top-k contrast + strong file context alpha 0.30"},
    {"name":"v329","kernel":"yourslewis/bc26-v329-immediate-topk-context0225","version":version,"message":"v329: immediate top-k contrast + file context alpha 0.225"},
    {"name":"v330","kernel":"yourslewis/bc26-v330-immediate-topk-context0175","version":version,"message":"v330: immediate top-k contrast + file context alpha 0.175"},
    {"name":"v331","kernel":"yourslewis/bc26-v331-immediate-topk-context01875","version":version,"message":"v331: immediate top-k contrast + file context alpha 0.1875"},
    {"name":"v332","kernel":"yourslewis/bc26-v332-immediate-topk-context02125","version":version,"message":"v332: immediate top-k contrast + file context alpha 0.2125"},
    {"name":"v333","kernel":"yourslewis/bc26-v333-immediate-topk-context020625","version":version,"message":"v333: immediate top-k contrast + file context alpha 0.20625"},
    {"name":"v334","kernel":"yourslewis/bc26-v334-immediate-topk-context019375","version":version,"message":"v334: immediate top-k contrast + file context alpha 0.19375"},
    {"name":"v335","kernel":"yourslewis/bc26-v335-immediate-topk-context0203125","version":version,"message":"v335: immediate top-k contrast + file context alpha 0.203125"},
    {"name":"v336","kernel":"yourslewis/bc26-v336-immediate-topk-context0196875","version":version,"message":"v336: immediate top-k contrast + file context alpha 0.196875"},
    {"name":"v337","kernel":"yourslewis/bc26-v337-immediate-topk-context02015625","version":version,"message":"v337: immediate top-k contrast + file context alpha 0.2015625"},
    {"name":"v338","kernel":"yourslewis/bc26-v338-immediate-topk-protossm-ew0625","version":version,"message":"v338: immediate top-k contrast + ProtoSSM ensemble weight 0.625"},
    {"name":"v339","kernel":"yourslewis/bc26-v339-immediate-topk-protossm-ew0575","version":version,"message":"v339: immediate top-k contrast + ProtoSSM ensemble weight 0.575"},
    {"name":"v340","kernel":"yourslewis/bc26-v340-immediate-topk-protossm-ew055","version":version,"message":"v340: immediate top-k contrast + ProtoSSM ensemble weight 0.55"},
    {"name":"v341","kernel":"yourslewis/bc26-v341-immediate-topk-protossm-ew065","version":version,"message":"v341: immediate top-k contrast + ProtoSSM ensemble weight 0.65"},
    {"name":"v342","kernel":"yourslewis/bc26-v342-immediate-topk-ew055-tail","version":version,"message":"v342: immediate top-k + ProtoSSM EW0.55 + tail dampening"},
    {"name":"v343","kernel":"yourslewis/bc26-v343-immediate-topk-ew055-alpha0525","version":version,"message":"v343: immediate top-k + ProtoSSM EW0.55 + quantile alpha 0.525"},
    {"name":"v344","kernel":"yourslewis/bc26-v344-immediate-topk-ew055-alpha0475","version":version,"message":"v344: immediate top-k + ProtoSSM EW0.55 + quantile alpha 0.475"},
    {"name":"v345","kernel":"yourslewis/bc26-v345-immediate-topk-ew055-gamma0825","version":version,"message":"v345: immediate top-k + ProtoSSM EW0.55 + power gamma 0.825"},
    {"name":"v346","kernel":"yourslewis/bc26-v346-immediate-topk-ew055-gamma0875","version":version,"message":"v346: immediate top-k + ProtoSSM EW0.55 + power gamma 0.875"},
    {"name":"v347","kernel":"yourslewis/bc26-v347-immediate-topk-ew055-gamma080","version":version,"message":"v347: immediate top-k + ProtoSSM EW0.55 + power gamma 0.80"},
    {"name":"v348","kernel":"yourslewis/bc26-v348-immediate-topk-ew055-gamma090","version":version,"message":"v348: immediate top-k + ProtoSSM EW0.55 + power gamma 0.90"},
    {"name":"v349","kernel":"yourslewis/bc26-v349-immediate-topk-ew060-gamma0825","version":version,"message":"v349: immediate top-k + ProtoSSM EW0.60 + power gamma 0.825"},
    {"name":"v350","kernel":"yourslewis/bc26-v350-immediate-topk-ew060-gamma0875","version":version,"message":"v350: immediate top-k + ProtoSSM EW0.60 + power gamma 0.875"},
    {"name":"v351","kernel":"yourslewis/bc26-v351-immediate-topk-ew060-gamma090","version":version,"message":"v351: immediate top-k + ProtoSSM EW0.60 + power gamma 0.90"},
    {"name":"v352","kernel":"yourslewis/bc26-v352-immediate-topk-ew060-gamma080","version":version,"message":"v352: immediate top-k + ProtoSSM EW0.60 + power gamma 0.80"},
    {"name":"v353","kernel":"yourslewis/bc26-v353-immediate-topk-ew0625-gamma0825","version":version,"message":"v353: immediate top-k + ProtoSSM EW0.625 + power gamma 0.825"},
    {"name":"v354","kernel":"yourslewis/bc26-v354-immediate-topk-ew0625-gamma0875","version":version,"message":"v354: immediate top-k + ProtoSSM EW0.625 + power gamma 0.875"},
    {"name":"v355","kernel":"yourslewis/bc26-v355-immediate-topk-ew0625-gamma090","version":version,"message":"v355: immediate top-k + ProtoSSM EW0.625 + power gamma 0.90"},
    {"name":"v356","kernel":"yourslewis/bc26-v356-immediate-topk-ew0625-gamma080","version":version,"message":"v356: immediate top-k + ProtoSSM EW0.625 + power gamma 0.80"},
    {"name":"v357","kernel":"yourslewis/bc26-v357-immediate-topk-ew065-gamma0825","version":version,"message":"v357: immediate top-k + ProtoSSM EW0.65 + power gamma 0.825"},
    {"name":"v358","kernel":"yourslewis/bc26-v358-immediate-topk-ew065-gamma0875","version":version,"message":"v358: immediate top-k + ProtoSSM EW0.65 + power gamma 0.875"},
    {"name":"v359","kernel":"yourslewis/bc26-v359-immediate-topk-ew065-gamma090","version":version,"message":"v359: immediate top-k + ProtoSSM EW0.65 + power gamma 0.90"},
    {"name":"v360","kernel":"yourslewis/bc26-v360-immediate-topk-ew065-gamma080","version":version,"message":"v360: immediate top-k + ProtoSSM EW0.65 + power gamma 0.80"},
    {"name":"v361","kernel":"yourslewis/bc26-v361-immediate-topk-ew0675-gamma0825","version":version,"message":"v361: immediate top-k + ProtoSSM EW0.675 + power gamma 0.825"},
]
with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
    token=json.load(f)["key"]
http=KaggleHttpClient(api_token=token)
competitions=CompetitionApiClient(http)
kernels=KernelsApiClient(http)

def recent_messages():
    req=ApiListSubmissionsRequest(); req.competition_name="birdclef-2026"; req.page_size=50
    return {str(s.description) for s in competitions.list_submissions(req).submissions}

def split_kernel(kernel):
    owner, slug = kernel.split("/", 1)
    return owner, slug

def is_complete(kernel):
    owner, slug = split_kernel(kernel)
    req=ApiGetKernelSessionStatusRequest(); req.user_name=owner; req.kernel_slug=slug
    status=kernels.get_kernel_session_status(req)
    print(f"Kernel status {kernel}: {status}", flush=True)
    return "COMPLETE" in str(status.status).upper()

def quota_sleep_seconds(text):
    m=re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m: return max(300,int(float(m.group(1))*3600)+120)
    m=re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m: return max(300,int(m.group(1))*60+120)
    return 3600

def submit_code(item):
    owner, slug = split_kernel(item["kernel"])
    req=ApiCreateCodeSubmissionRequest()
    req.competition_name="birdclef-2026"
    req.kernel_owner=owner
    req.kernel_slug=slug
    req.kernel_version=item["version"]
    req.file_name="submission.csv"
    req.submission_description=item["message"]
    return competitions.create_code_submission(req)

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
            res=submit_code(item); print("Submission result:", res, flush=True); progressed=True; time.sleep(30); break
        except requests.exceptions.HTTPError as exc:
            response=getattr(exc,"response",None); text=getattr(response,"text","") if response is not None else str(exc)
            print(f"Submission attempt failed for {item['name']}: {type(exc).__name__}: {exc}", flush=True)
            if text: print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                sleep_s=quota_sleep_seconds(text); print(f"Daily submission allowance exhausted; sleeping {sleep_s} seconds before retry.", flush=True); time.sleep(sleep_s); progressed=True; break
            raise
    if all_done:
        print("All pending kernels are already submitted.", flush=True); break
    if not progressed: time.sleep(600)
