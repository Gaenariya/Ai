# -*- coding: utf-8 -*-
"""이매지니어 AI 서버 — RunPod 등 일반 서버용.
24번 노트북(2~8단계)과 **같은 코드**예요. 노트북을 고치면 make_runpod.py를 다시 돌려서 이 파일을 새로 만들면 돼요.

실행:  bash run.sh        (또는  uvicorn app:app --host 0.0.0.0 --port 8000)
키:    환경변수 OPENAI_API_KEY / AI_SERVER_KEY / HF_TOKEN  (RunPod의 Environment Variables에 넣으면 돼요)
재료:  ./materials/floors  (바닥재),  ./materials/walls  (벽지)   ← 백엔드가 /floors/sync로 넣어주면 비어 있어도 됨
명세:  서버 주소 + /docs
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "hf-cache"))   # 모델을 이 폴더에 받아둠 (재시작해도 다시 안 받게)
# 설정 (노트북 2단계와 같음) (여기만 고치면 됨)

MODEL = "gpt-5.4-mini"    # 분석에 쓸 OpenAI 모델 (사진을 볼 수 있고 가벼운 모델. 계정에서 쓸 수 있는 이름으로 바꾸기)
N_PROPOSALS = 3           # 시안 개수 (벽 색 + 바닥재 조합). 2~5 권장
MAX_REFERENCES = 5        # 레퍼런스 사진 최대 장수 (1장도 가능)
SEND_MAX_SIDE = 512       # AI에 보내는 레퍼런스 사진의 긴 변 크기 (작을수록 빠름. 색·분위기 분석에는 512로 충분)

SEG_MODEL = ("oneformer", "shi-labs/oneformer_ade20k_swin_large")
# SEG_MODEL = ("mask2former", "facebook/mask2former-swin-large-ade-semantic")

# --- 사진 크기 (분할은 작게, 합성은 크게) ---
SEG_MAX_SIDE = 1024       # 영역 분할에 쓸 크기. 모델이 내부에서 줄여 보기 때문에 더 키워도 정확도는 비슷하고 느려지기만 함
OUTPUT_MAX_SIDE = 2000    # 결과(합성) 사진의 긴 변 상한. None이면 원본 그대로 (느리고 메모리를 많이 씀)

# --- 천장 ---
CEILING_MODE = "auto"     # "auto"=벽 색보다 살짝 밝게 / "none"=천장은 그대로 둠 / "#RRGGBB"=지정한 색
CEILING_LIGHTEN = 1.12    # auto일 때 벽보다 얼마나 밝게 (1.0=벽과 같은 색, 1.2=꽤 밝게)

# --- 몰딩 (천장 몰딩 + 걸레받이) ---
MOLDING_MODE = "wall"     # "wall"=벽과 같은 색으로 덮기(몰딩 지운 느낌, 입체는 남음) / "keep"=원래 몰딩 그대로 / "#RRGGBB"=지정한 색
MOLDING_FIND = True       # 몰딩 띠를 자동으로 찾기 (분할 모델에는 몰딩 클래스가 없어서 직접 찾음)
MOLDING_MAX_THICK = 0.05  # 몰딩 두께 상한 (사진 높이 대비). 몰딩이 아주 두꺼운 집이면 0.08까지
MOLDING_DIFF = 22         # 벽 바탕색과 이만큼 이상 다르면 몰딩 후보 (작을수록 많이 잡음)
MOLDING_MIN_COLS = 0.25   # 세로줄 중 이 비율 이상에서 띠가 보여야 '몰딩 있음'으로 인정

# --- 벽/바닥/천장에서 빼둘 것 (여기에 벽지·바닥재가 발리면 안 되는 것들) ---
KEEP_AS_IS = ["windowpane", "door", "curtain", "painting", "mirror", "screen door"]
#   분할 모델이 찾아낸 이 영역들은 벽에 포함하지 않고 **원본 그대로** 남겨요
#   (안 빼면 창문에도 벽지가 발려요). 모델이 못 찾으면 어쩔 수 없고, 붓으로 지우면 돼요

# --- SAM 3 (말로 영역을 찾아주는 모델) ---
MOLDING_SOURCE = os.environ.get("MOLDING_SOURCE", "auto")   # 환경변수로 바꿀 수 있어요. 몰딩을 무엇으로 찾을지: "auto"=SAM 3 되면 그걸로, 안 되면 색으로 / "sam3" / "color" / "none"
SAM3_MODEL = "facebook/sam3" # 승인받은 계정의 HF_TOKEN이 필요해요 (Colab 🔑 또는 환경변수)
SAM_PROMPTS = {"molding_top": "crown molding on the ceiling edge", "molding_bottom": "baseboard at the bottom of the wall"}
SAM_SCORE = 0.4              # 이 확신도 아래는 버림 (낮추면 더 많이 잡고, 올리면 확실한 것만)
SAM_MAX_SIDE = 1024          # SAM에 넣기 전에 사진을 이 크기로 줄임
SAM_KEEP_PROMPTS = []        # 예: ["window", "door", "curtain"] — 벽에서 빼둘 것을 SAM으로도 찾고 싶을 때

# --- 몰딩 색 제안 ---
MOLDING_SUGGEST_LIGHTEN = 1.12   # AI가 몰딩 색을 안 주면 벽 색을 이만큼 밝게 해서 몰딩 색으로 씀

# --- 벽 면 나누기 (백엔드가 wall1, wall2, wall3로 받고 싶어 함) ---
WALL_FACE_MIN = 0.01      # 벽 면 하나가 화면에서 이 비율보다 작으면 옆 면에 합침

# --- 벽을 '덮어씌울지' '필터로 입힐지' 자동으로 고르기 ---
WALL_METHOD = os.environ.get("WALL_METHOD", "auto")   # 환경변수로 바꿀 수 있어요. "auto"=무늬가 있으면 덮어씌우고 없으면 필터 / "overlay"=항상 덮어씌움 / "filter"=항상 필터
WALL_PATTERN_T = 0.18     # 벽에서 '무늬'로 본 면적이 이 비율을 넘으면 무늬 있는 벽으로 봄 (꽃무늬 벽지, 낙서, 포인트 벽)
#   9단계에 면마다 점수가 찍히니, 민무늬인데 덮어씌운다면 올리고 / 무늬가 비치면 내리기 (참고: 꽃무늬 0.4, 벽지 결 0.1 정도)
#  · 덮어씌우기(overlay): 바닥재처럼 새 재료를 깔고 **원래 방의 빛만** 가져옴 → 옛 무늬가 완전히 사라짐
#  · 필터(filter): 원래 벽의 결·굴곡을 그대로 곱함 → 민무늬 벽에서 가장 자연스러움 (지우개를 아예 안 씀)
#  · 벽지 '사진'을 쓸 때는 언제나 덮어씌우기 (사진이 원래 벽을 가리니까)

# --- 벽지 사진으로 덮어씌우기 (색상 코드 대신 진짜 벽지 사진을 깔기) ---
WALL_FOLDER = os.environ.get("WALL_FOLDER", os.path.join(BASE_DIR, "materials", "walls"))   # 벽지 사진을 모아둔 Drive 폴더 (비어 있으면 색상 코드만 씀)
WALL_USE_LIBRARY = "auto"   # "auto"=벽지 폴더에 사진이 있으면 벽지로, 없으면 색으로 / False=항상 색 / True=반드시 벽지
WALL_MODE = "perspective"   # "flat"=정면 벽처럼 무늬를 곧게 깔기 / "perspective"=옆벽은 안쪽으로 갈수록 좁아지게
WALL_SPLIT = None           # 벽 모서리의 가로 위치 (x1, x2). None=바닥 경계선이 꺾이는 자리에서 자동 추정
WALL_TILE = 0.3             # 벽지 무늬 한 칸의 가로 크기 (정면 벽 기준, 이미지 가로 대비). 0.3이면 화면에 세 칸쯤
WALL_LIGHT = 1.0            # 원래 방의 빛(밝은 곳/그림자) 유지 정도. 0=평평, 1=원본 그대로
WALL_SPEC = 0.3             # 원래 사진의 반짝임을 다시 얹는 정도 (벽은 바닥보다 약하게)
WALL_EXPOSURE = 1.0         # 벽지 밝기 배율. 아주 하얀 벽지에서 빛이 뭉치면 0.9쯤으로

# --- 벽 패턴 지우기 (굴곡·그림자는 남기고 무늬·낙서만) ---
WALL_PATTERN_REMOVE = 1.0   # **덮어씌우기 방식에서만** 씀 (필터 방식은 원래 결을 그대로 둠). 1=옛 무늬를 최대한 지움
PATTERN_DETAIL_T = 10.0     # 작고 촘촘한 무늬를 잡는 기준 (작을수록 많이 지움)
PATTERN_CHROMA_T = 12.0     # 벽 바탕과 '색'이 다른 정도의 기준 (색 있는 낙서·포인트 벽지에 잘 들음)
PATTERN_TONE_T = 60.0       # 넓고 진한 얼룩을 잡는 기준. 낮추면 공격적이지만 턱 그림자까지 평평해질 수 있음

# --- 영역 분할 다듬기 (원룸처럼 복잡한 방에서 가장자리가 흐릿하면 여기) ---
MASK_SMOOTH = 0.3         # 마스크 가장자리를 둥글리는 정도. 0=안 둥글림(제일 또렷), 1=예전 19번처럼 많이
MASK_MIN_AREA = 0.002     # 이 비율(화면 넓이 대비)보다 작은 조각·구멍은 정리
EDGE_SNAP = True          # 경계 주변에서 사진의 진짜 경계선(가구·문틀)에 맞춰 마스크를 다시 그림
EDGE_SNAP_BAND = 0.01     # 경계에서 얼마나 떨어진 곳까지 다듬을지 (이미지 긴 변 대비). 경계가 많이 어긋나면 0.02
EDGE_FEATHER = 0.6        # 칠할 때 가장자리 부드러움(픽셀). 0=완전히 또렷(계단이 보일 수 있음)
EDGE_STYLE = "straight"   # 영역 경계 모양: "straight" = 딱딱한 직선, "smooth" = 부드러운 곡선 (EDGE_SMOOTH만 적용)
EDGE_STRAIGHT = 0.004     # straight일 때 직선 단순화 정도 (이미지 긴 변 대비). 클수록 각진 큰 직선, 작을수록 원래 모양에 가까움
EDGE_SMOOTH = 0.0025     # 영역 경계의 잔톱니·계단을 다듬는 정도 (이미지 긴 변 대비). 0=끔. 몰딩이 너무 가늘어지면 0.0015, 아직 울퉁불퉁하면 0.004

# --- 벽 색 ---
WALL_MAX_LUMA = 0.93      # 벽 색의 밝기 상한 (0~1). 밝은 시안이 하얗게 날아가면 0.88쯤으로 낮추기
WALL_SOFT_HIGHLIGHT = True  # 밝은 부분이 255를 넘어 뭉개지지 않게 부드럽게 눌러줌 (끄면 19번과 같음)
DARK_LUMA = 110.0         # 영역 평균 밝기(0~255)가 이보다 어두우면 명암 차이를 줄여서 입혀요 (어두운 사진의 하얀 얼룩 방지). 0=끔
DARK_MIN_KEEP = 0.35      # 아주 어두울 때 명암을 남기는 최소 비율 (작을수록 더 평평하게)

def dark_keep(avg_gray):
    """영역 평균 밝기(0~255)로 '명암을 얼마나 남길지' 비율을 정해요. 밝은 사진은 1(그대로), 어두울수록 작아짐.
    어두운 사진은 평균이 작아서, 작은 빛 번짐도 '평균의 몇 배'로 계산돼 새 색 위에 하얀 얼룩이 생기기 때문."""
    if DARK_LUMA <= 0:
        return 1.0
    return float(np.clip(float(avg_gray) / DARK_LUMA, DARK_MIN_KEEP, 1.0))

# --- 바닥재 폴더 ---
FLOOR_FOLDER = os.environ.get("FLOOR_FOLDER", os.path.join(BASE_DIR, "materials", "floors"))   # 바닥재 사진을 모아둔 Drive 폴더
FLOOR_MIRROR_TILE = False  # True: 이음새가 있는 사진을 좌우·상하로 뒤집어 이어 붙여 이음새를 감춤 (무늬가 대칭으로 보일 수 있음)

# --- 바닥재 고르기 (CLIP) ---
CLIP_MODEL = "openai/clip-vit-base-patch32"   # 정확도를 더 올리고 싶으면 "openai/clip-vit-large-patch14" (느리고 무거움)
FLOOR_COLOR_WEIGHT = 4.0   # 대표 색이 비슷한 바닥재를 얼마나 우대할지. 0=색 무시, 클수록 색을 중요하게
FLOOR_FROM_REFERENCE = False   # True: 레퍼런스 사진의 바닥 부분을 잘라 바닥재 사진과 직접 비교하는 점수도 더함
FLOOR_REF_WEIGHT = 1.0     # 위 옵션을 켰을 때의 비중

# --- 바닥 깔기 (원근) ---
FLOOR_TILE = 0.5          # 바닥재 한 칸의 가로 크기 (화면 맨 아래 기준, 이미지 가로 대비). 작을수록 잘게 반복됨
HORIZON = None            # 수평선 높이 (이미지 높이 대비 0~1, 위가 0). None=자동. 줄이 이상하게 모이면 11단계 미리보기 보며 조절
FOV = 0.9                 # 카메라 화각 느낌 (클수록 원근이 약함, 0.7~1.3)
ROTATE = 0                # 바닥 무늬 회전 각도(도). 사진을 비스듬히 찍었으면 -30~30 사이로 조절

# --- 바닥 빛과 반사 ---
FLOOR_LIGHT = 1.0         # 원래 방의 빛(밝은 곳/그림자) 유지 정도. 0=평평, 1=원본 그대로
FLOOR_SPEC = 1.0          # 원래 사진의 반짝이는 하이라이트를 다시 얹는 정도
FLOOR_REFLECT = 0.0       # 거울 같은 반사. 0=없음, 0.3=은은한 광택, 0.7=반들반들
FLOOR_ROUGH = 0.3         # 반사의 흐릿함. 0=또렷, 1=매우 흐림
FLOOR_EXPOSURE = 1.0      # 바닥재 밝기 배율. 아주 하얀 바닥재에서 빛이 뭉치면 0.85쯤으로

# 3단계: 영역 분할 (OneFormer를 한 번만 불러와서 계속 재사용 → 두 번째 사진부터는 몇 초)
import gc, io, time
import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image_rgb(path_or_image, max_side=1024):
    """사진을 RGB로 읽고, 너무 크면 긴 변을 max_side로 줄여요. max_side=None이면 원본 크기 그대로.
    휴대폰 사진의 회전 정보(EXIF)를 반영하고, 이미 열어 둔 사진을 넣어도 돼요."""
    from PIL import ImageOps
    image = path_or_image if isinstance(path_or_image, Image.Image) else ImageOps.exif_transpose(Image.open(path_or_image))
    image = image.convert("RGB")
    if max_side is None:
        return image
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image

_seg_cache = {}
def get_segmenter():
    """분할 모델을 처음 한 번만 불러와요 (이후엔 그대로 재사용)."""
    kind, name = SEG_MODEL
    if kind not in _seg_cache:
        if kind == "oneformer":
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
            processor = OneFormerProcessor.from_pretrained(name)
            model = OneFormerForUniversalSegmentation.from_pretrained(name).eval().to(DEVICE)
        elif kind == "mask2former":
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            processor = AutoImageProcessor.from_pretrained(name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(name).eval().to(DEVICE)
        else:
            raise ValueError(f"알 수 없는 모델 종류: {kind}")
        _seg_cache[kind] = (processor, model)
    return _seg_cache[kind]

def find_class_id(id2label, name):
    for class_id, label in id2label.items():
        if label.split(",")[0].strip() == name:
            return int(class_id)
    raise KeyError(f"{name} 클래스를 찾을 수 없음")

def segment_room(image):
    """방 사진 → 벽/바닥/천장 마스크 (True/False 지도)."""
    if SEG_MODEL[0] == "sam3":
        return segment_room_sam3(image)
    processor, model = get_segmenter()
    kwargs = {"task_inputs": ["semantic"]} if SEG_MODEL[0] == "oneformer" else {}
    inputs = {k: v.to(DEVICE) for k, v in processor(images=image, return_tensors="pt", **kwargs).items()}
    with torch.no_grad():
        outputs = model(**inputs)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
    id2label = model.config.id2label
    out = {name: seg == find_class_id(id2label, name) for name in ["wall", "floor", "ceiling"]}
    out["keep"] = keep_as_is_mask(seg, id2label)      # 창문·문처럼 그대로 둬야 하는 곳
    return out

def keep_as_is_mask(seg, id2label):
    """KEEP_AS_IS(창문·문·커튼 등)에 해당하는 픽셀을 모아요 → 벽/바닥/천장에서 빼둘 영역.
    분할 모델에 없는 이름은 그냥 건너뛰어요."""
    keep = np.zeros(seg.shape, bool)
    for name in KEEP_AS_IS:
        try:
            keep |= (seg == find_class_id(id2label, name))
        except KeyError:
            pass
    return keep

# ---- SAM 3: 말(영어)로 영역 찾기 ----
_sam_cache = {}

def get_sam3():
    """SAM 3 모델을 올려요 (처음 한 번만). 못 쓰면 None을 돌려주고 이유를 한 번만 알려줘요.
    필요한 것: transformers 5.12 이상 + 승인받은 계정의 HF_TOKEN."""
    if "model" in _sam_cache or "failed" in _sam_cache:
        return _sam_cache.get("model")
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError:
        _sam_cache["failed"] = "transformers에 Sam3Model이 없어요 (pip install -U transformers, 5.12 이상)"
        print(f"⚠️ SAM 3를 못 써요: {_sam_cache['failed']} → 몰딩은 색으로 찾아요"); return None
    token = get_key("HF_TOKEN")
    try:
        t = time.time()
        processor = Sam3Processor.from_pretrained(SAM3_MODEL, token=token)
        model = Sam3Model.from_pretrained(SAM3_MODEL, token=token).to(DEVICE).eval()
        _sam_cache.update(model=model, processor=processor)
        print(f"SAM 3 준비 완료 ({time.time() - t:.0f}초)")
        return model
    except Exception as e:
        _sam_cache["failed"] = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"⚠️ SAM 3를 못 불러왔어요: {_sam_cache['failed']} → 몰딩은 색으로 찾아요")
        print("   (승인: huggingface.co/facebook/sam3 / 토큰: 🔑 또는 환경변수 HF_TOKEN)")
        return None

def sam3_find(image, text, score=None, max_side=None):
    """사진에서 text(짧은 영어)로 찾은 것들을 **하나의 마스크**로 합쳐서 돌려줘요 (못 찾으면 전부 False).
    image: PIL 또는 배열. 돌려주는 마스크는 image와 같은 크기."""
    model = get_sam3()
    arr = np.array(image) if not isinstance(image, np.ndarray) else image
    H, W = arr.shape[:2]
    if model is None:
        return np.zeros((H, W), bool)
    processor, score = _sam_cache["processor"], (SAM_SCORE if score is None else score)
    s = (SAM_MAX_SIDE if max_side is None else max_side) / max(H, W)
    small = cv2.resize(arr, (max(16, int(W * s)), max(16, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else arr
    inputs = processor(images=Image.fromarray(small), text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = inputs.get("original_sizes")
    results = processor.post_process_instance_segmentation(
        outputs, threshold=score, mask_threshold=0.5,
        target_sizes=sizes.tolist() if hasattr(sizes, "tolist") else [[small.shape[0], small.shape[1]]])[0]
    found = np.zeros(small.shape[:2], bool)
    for m, sc in zip(results.get("masks", []), results.get("scores", [])):
        if float(sc) < score:
            continue
        m = np.array(m.cpu() if hasattr(m, "cpu") else m)
        found |= (m.squeeze() > 0.5) if m.dtype != bool else m.squeeze()
    if found.shape != (H, W):
        found = cv2.resize(found.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return found

def find_moldings_sam3(masks, image):
    """SAM 3로 천장 몰딩·걸레받이를 찾아요 → (마스크들, 정보). 벽 근처만 남겨서 엉뚱한 곳이 칠해지지 않게 해요."""
    wall = masks["wall"]
    H, W = wall.shape
    near_wall = cv2.dilate(wall.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2) > 0
    out, info = {}, {}
    for key, prompt in SAM_PROMPTS.items():
        found = sam3_find(image, prompt) & near_wall
        found = clean_mask(found, 0.0002, 0.2) if found.any() else found
        rows = np.where(found.any(axis=1))[0]
        cols_ratio = float(found.any(axis=0).mean())
        if found.any() and cols_ratio >= MOLDING_MIN_COLS * 0.5:        # 너무 조금 보이면 버림
            out[key] = found
            info[key] = {"found": True, "thickness": int(found.sum() / max(1, found.any(axis=0).sum())),
                         "cols_ratio": cols_ratio, "rows": [int(rows.min()), int(rows.max())], "by": "sam3"}
        else:
            info[key] = {"found": False, "cols_ratio": cols_ratio, "by": "sam3"}
    return out, info

def find_moldings_auto(masks, image, source=None):
    """몰딩 찾기 — SAM 3를 먼저 써보고, 안 되면 색으로 찾는 방법으로 돌아가요 (2단계 MOLDING_SOURCE).
    찾은 뒤에는 '천장 몰딩은 천장 바로 아래, 걸레받이는 바닥 바로 위'에 있는 부분만 남겨요 (가구 테두리를 몰딩으로 착각하는 것 방지)."""
    out, info = _find_moldings_any(masks, image, source)
    out, info = keep_moldings_near_edges(masks, out, info)
    if MOLDING_REGULARIZE:
        try:
            out, info = regularize_moldings(masks, out, info)
        except Exception as e:                            # 실패해도 찾은 모양 그대로 써요
            print(f"⚠️ 몰딩 모양 다듬기를 건너뛰었어요 ({type(e).__name__}: {str(e)[:80]})")
    return out, info

MOLDING_REGULARIZE = True   # 몰딩을 '벽-바닥(천장) 경계를 따라 쭉 이어진 일정한 두께의 띠'로 다시 그려요. False면 찾은 모양 그대로
MOLDING_MIN_COVER = 0.15    # 경계 중 이 비율 이상에서 몰딩이 실제로 보여야 다시 그려요 (아니면 찾은 모양 그대로)
MOLDING_BAND_MAX = 0.03     # 다시 그릴 때 몰딩 두께 상한 (사진 높이 대비). 보통 걸레받이는 0.015~0.025. 너무 얇으면 0.04
MOLDING_THICK_Q = 35        # 잰 두께 중 몇 % 지점 값을 쓸지 (작을수록 얇게, 50=중앙값). 덩어리에 끌려 두꺼워지는 걸 막음

def _edge_rows(lower, upper, from_top):
    """세로줄마다 'upper 바로 아래에 lower가 시작되는' 경계 y를 찾아요 (없으면 -1).
    from_top=True: 바닥처럼 아래 영역(lower)의 가장 위 경계 / False: 천장처럼 위 영역의 가장 아래 경계."""
    H, W = lower.shape
    if from_top:                                             # lower(바닥) 픽셀 중 1~3픽셀 위에 upper(벽)가 있는 곳
        near = np.zeros((H, W), bool)
        for d in (1, 2, 3):
            near[d:] |= upper[:-d]
        hit = lower & near
        ys = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)              # 가장 위의 경계
    else:                                                    # upper(천장) 픽셀 중 1~3픽셀 아래에 lower(벽)가 있는 곳
        near = np.zeros((H, W), bool)
        for d in (1, 2, 3):
            near[:-d] |= lower[d:]
        hit = upper & near
        ys = np.where(hit.any(axis=0), H - 1 - hit[::-1].argmax(axis=0), -1)  # 가장 아래의 경계
    return ys.astype(np.int32)

def _smooth_edge(ys, W):
    """경계 높이의 작은 들쭉날쭉을 펴고, 아주 짧은 끊김(잡음)만 이어요. 긴 끊김(가구에 가린 곳)은 그대로 끊김."""
    ys = ys.astype(np.float32)
    ok = ys >= 0
    max_gap = max(3, int(0.012 * W))
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):                      # 짧은 끊김 메우기
        if 1 < b - a <= max_gap + 1:
            ys[a + 1:b] = np.linspace(ys[a], ys[b], b - a + 1)[1:-1]
            ok[a + 1:b] = True
    win = max(3, int(0.02 * W)) | 1
    out = ys.copy()
    half = win // 2
    for x in np.where(ok)[0]:                                # 이어진 구간 안에서만 중앙값으로 펴기
        lo, hi = max(0, x - half), min(W, x + half + 1)
        seg = ys[lo:hi][ok[lo:hi]]
        out[x] = np.median(seg)
    return np.where(ok, out, -1)

def regularize_moldings(masks, out, info):
    """찾은 몰딩 덩어리를 '경계를 따라 쭉 이어진 띠'로 다시 그려요.
    - 두께: 찾은 몰딩에서 잰 두께를 원근(수평선에서 멀수록 두꺼움)에 맞춰 세로줄마다 정함
    - 모양: 벽과 바닥(천장)이 실제로 맞닿은 세로줄 전체 → 가구에 가린 곳에서만 끊김"""
    wall, floor, ceil = masks.get("wall"), masks.get("floor"), masks.get("ceiling")
    if wall is None or not out:
        return out, info
    H, W = wall.shape
    wallish = wall | np.logical_or.reduce([m for m in out.values()])
    max_t = max(2, int(MOLDING_MAX_THICK * H))               # 두께를 잴 때 보는 범위
    band_max = max(2, int(MOLDING_BAND_MAX * H))             # 새로 그릴 때 두께 상한
    y_h = estimate_horizon(floor, None) if floor is not None and floor.any() else 0.5 * H
    xs = np.arange(W)
    for key, region, below in (("molding_bottom", floor, True), ("molding_top", ceil, False)):
        if key not in out or region is None or not region.any():
            continue
        found = out[key]
        edge = _edge_rows(region, wallish, True) if below else _edge_rows(wallish, region, False)
        cols = np.where(edge >= 0)[0]
        if len(cols) < 10:
            continue
        # 세로줄마다 찾은 몰딩 두께 재기: 경계에서부터 끊기지 않고 이어진 부분만 (위에 떨어져 붙은 덩어리는 안 셈)
        thick = np.zeros(W, np.float32)
        for x in cols:
            y = edge[x]
            seg = found[max(0, y - max_t):y, x][::-1] if below else found[y + 1:min(H, y + 1 + max_t), x]
            run = 0
            for i, v in enumerate(seg):                       # 경계 바로 옆 1~2픽셀 빈틈은 봐줌
                if v:
                    run = i + 1
                elif i - run >= 2:
                    break
            thick[x] = run
        seen = cols[thick[cols] >= 2]
        if len(seen) < MOLDING_MIN_COVER * len(cols):         # 몰딩이 보인 곳이 너무 적으면 덩어리 그대로 둠
            info[key] = {**info.get(key, {}), "regularized": False}
            continue
        edge_s = _smooth_edge(edge, W)
        valid = edge_s >= 0
        # 두께: 세로줄마다 '근처(가로 ±10%)에서 실제로 잰 두께'를 따라감 → 먼 곳은 얇게, 가까운 곳은 두껍게가 저절로 반영
        t_glob = float(np.percentile(thick[seen], MOLDING_THICK_Q))
        win = max(5, int(0.10 * W))
        t = np.full(W, t_glob, np.float32)
        for x in np.where(valid)[0]:
            near = seen[np.abs(seen - x) <= win]
            if len(near) >= 3:
                t[x] = np.percentile(thick[near], MOLDING_THICK_Q)
        k = max(3, int(0.03 * W)) | 1                         # 두께가 갑자기 바뀌지 않게 가로로 부드럽게
        t = np.convolve(np.pad(t, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        t = np.clip(np.round(t), 2, band_max).astype(np.int32)
        rows = np.arange(H)[:, None]
        e = edge_s.astype(np.int32)[None, :]
        if below:
            band = (rows < e) & (rows >= e - t[None, :])
        else:
            band = (rows > e) & (rows <= e + t[None, :])
        band &= valid[None, :] & wallish
        for o in out:                                         # 다른 몰딩과 겹치지 않게
            if o != key:
                band &= ~out[o]
        if band.sum() >= 20:
            out[key] = band
            info[key] = {**info.get(key, {}), "regularized": True,
                         "cover": round(float(valid.mean()), 3), "thickness_px": int(np.median(t[valid]))}
    return out, info

def keep_moldings_near_edges(masks, out, info, gap=None):
    """몰딩 후보 중 제자리에 있는 부분만 남겨요.
    - molding_bottom(걸레받이): 바닥에서 gap 이내
    - molding_top(천장 몰딩): 천장에서 gap 이내 (천장을 못 찾았으면 세로줄마다 벽 맨 윗부분에서 gap 이내)
    gap: 사진 높이 대비 (기본 = 몰딩 두께 상한 + 2%). 남은 게 너무 작으면 그 몰딩은 '못 찾음'으로 처리."""
    if not out:
        return out, info
    wall = masks.get("wall")
    if wall is None:
        return out, info
    H, W = wall.shape
    gap_px = max(3, int(((MOLDING_MAX_THICK + 0.02) if gap is None else gap) * H))
    min_px = max(20, int(0.0005 * H * W))

    def near(region):
        if region is None or not region.any():
            return None
        dist = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
        return dist <= gap_px

    near_floor = near(masks.get("floor"))
    near_ceil = near(masks.get("ceiling"))
    if near_ceil is None and wall.any():                     # 천장을 못 찾았으면 벽의 맨 윗줄 근처
        top = np.where(wall.any(axis=0), wall.argmax(axis=0), H)
        near_ceil = (np.arange(H)[:, None] - top[None, :]) <= gap_px
    for key, zone in (("molding_bottom", near_floor), ("molding_top", near_ceil)):
        if key not in out or zone is None:
            continue
        before = int(out[key].sum())
        band = out[key] & zone
        n, lab, st, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
        small = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < max(4, min_px // 10)]
        if small:
            band &= ~np.isin(lab, small)
        if int(band.sum()) < min_px:
            out.pop(key)
            info[key] = {**info.get(key, {}), "found": False, "dropped": "제자리(바닥·천장 근처)에 없어서 뺐어요"}
        else:
            out[key] = band
            if int(band.sum()) < before:
                info[key] = {**info.get(key, {}), "trimmed_ratio": round(1 - band.sum() / max(1, before), 3)}
    return out, info

def _find_moldings_any(masks, image, source=None):
    source = MOLDING_SOURCE if source is None else source
    if source == "none":
        return {}, {}
    if source in ("auto", "sam3"):
        try:
            out, info = find_moldings_sam3(masks, image)
        except Exception as e:
            print(f"⚠️ SAM 3로 몰딩 찾다가 실패했어요 ({type(e).__name__}: {str(e)[:120]})")
            out, info = {}, {}
        if out:
            missing = [k for k in SAM_PROMPTS if k not in out]
            if missing and source == "auto":              # SAM 3가 한쪽만 찾았으면, 놓친 쪽만 색으로 한 번 더 찾아봐요
                try:
                    c_out, c_info = find_moldings(masks, image)
                except Exception as e:
                    print(f"⚠️ 색으로 몰딩 보완하다가 실패했어요 ({type(e).__name__})")
                    c_out, c_info = {}, {}
                for k in missing:
                    if k in c_out:
                        others = np.logical_or.reduce([out[o] for o in out])
                        band = c_out[k] & ~others
                        if band.any():
                            out[k] = band
                            info[k] = {**c_info.get(k, {}), "by": "color", "found": True}
            return out, info
        if source == "sam3":
            return out, info
    return find_moldings(masks, image)      # 색으로 찾기 (원래 방법)

# 4단계: 마스크 정리·경계 스냅 + 벽 무늬 지우기 + 몰딩 찾기 + 벽지·바닥재 깔기(원근) + 빛·반사
import cv2

def clean_mask(mask, min_area_ratio=None, smooth=None):
    """마스크의 지저분한 부분을 정리해요 (후처리).
    1) 작은 점/구멍 없애기(open/close)  2) 너무 작은 조각 제거  3) 작은 구멍 메우기
    smooth: 둥글리기 정도. 0이면 둥글리지 않아 가장자리가 가장 또렷하고, 크면 모서리가 뭉개져요 (19번은 1에 해당)."""
    min_area_ratio = MASK_MIN_AREA if min_area_ratio is None else min_area_ratio
    smooth = MASK_SMOOTH if smooth is None else smooth
    H, W = mask.shape
    mask_u8 = mask.astype(np.uint8)
    k = int(min(H, W) * 0.01 * smooth) | 1 if smooth > 0 else 0      # 이미지 크기에 맞춘 홀수 커널
    if k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)     # 튀어나온 점 제거
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)    # 작은 구멍 메우기

    min_area = H * W * min_area_ratio
    # 작은 조각 제거
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_u8[labels == i] = 0
    # 작은 구멍(배경 조각) 메우기
    holes = (1 - mask_u8).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_u8[labels == i] = 1
    return mask_u8.astype(bool)


def _region_stats(image, confident):
    """영역 안쪽(확실한 부분)의 평균 색과 색이 퍼진 정도를 구해요."""
    pixels = image[confident].astype(np.float32)
    mean = pixels.mean(axis=0)
    sigma = max(float(np.sqrt(((pixels - mean) ** 2).sum(axis=1).mean())), 14.0)   # 너무 작으면 과민해져서 하한을 둠
    return mean, sigma

def snap_edges(masks, image, band=None, keep_pull=0.6):
    """경계 주변의 띠 안에서만, 픽셀 색을 각 영역 '안쪽 색'과 비교해 다시 나눠요.
    → 마스크 경계가 사진의 진짜 경계선(벽과 바닥이 만나는 선, 가구 끝)에 붙음.
    띠 밖(영역 한가운데)은 건드리지 않아서 분할 결과가 통째로 바뀌지는 않고,
    원래 영역에서 멀수록 벌점(keep_pull)을 줘서 멀리 있는 영역이 끌어가지 못하게 함.
    가구처럼 어디에도 안 잡힌 자리도 하나의 영역으로 함께 비교함."""
    band = EDGE_SNAP_BAND if band is None else band
    H, W = image.shape[:2]
    r = max(2, int(band * max(H, W)))
    names = list(masks)
    groups = [masks[n] for n in names] + [np.logical_not(np.logical_or.reduce([masks[n] for n in names]))]   # 마지막은 '그 외'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    costs, usable = [], []
    for m in groups:
        confident = cv2.erode(m.astype(np.uint8), kernel) > 0        # 경계에서 충분히 안쪽
        if confident.sum() < 50:
            costs.append(None); usable.append(False); continue
        mean, sigma = _region_stats(image, confident)
        color_cost = np.sqrt(((image.astype(np.float32) - mean) ** 2).sum(axis=2)) / sigma
        away = cv2.distanceTransform((~confident).astype(np.uint8), cv2.DIST_L2, 3) / r
        costs.append(color_cost + keep_pull * away)
        usable.append(True)
    if sum(usable) < 2:
        return dict(masks)                                            # 비교할 영역이 모자라면 그대로
    big = np.full((H, W), np.inf, np.float32)
    stack = np.stack([c if c is not None else big for c in costs])
    winner = stack.argmin(axis=0).astype(np.uint8)
    winner = cv2.medianBlur(winner, 3)                                # 점점이 튀는 것 정리
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = np.zeros((H, W), np.uint8)
    for m in groups[:-1]:
        u = m.astype(np.uint8)
        boundary |= cv2.dilate(u, small) - cv2.erode(u, small)        # 지금 마스크의 테두리
    nearby = cv2.dilate(boundary, kernel) > 0
    out = {}
    for i, n in enumerate(names):
        new = masks[n].copy()
        if usable[i]:
            new[nearby] = (winner == i)[nearby]
        out[n] = new
    return out


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))

# ---- 벽의 '굴곡'과 '무늬'를 갈라내기 ----
def _box(x, r):
    """사각형 평균 흐리기 (크기와 상관없이 빠름)."""
    return cv2.boxFilter(x.astype(np.float32), -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REPLICATE)

def _masked_mean(x, m, r):
    """마스크 안쪽만 보고 평균 내기 (밖은 아예 안 섞이게)."""
    return _box(x * m, r) / (_box(m, r) + 1e-6)

def smooth_keep_edges(lum, m, r, eps=12.0):
    """밝기를 부드럽게 하되 또렷한 경계(턱, 모서리)는 남겨요 (가이디드 필터, 자기 자신을 안내로).
    eps보다 변화가 작은 잔무늬는 뭉개지고, 큰 경계는 그대로 남아요."""
    mean = _masked_mean(lum, m, r)
    mean_sq = _masked_mean(lum * lum, m, r)
    var = np.maximum(mean_sq - mean * mean, 0)
    a = var / (var + eps * eps)
    b = mean * (1 - a)
    return _masked_mean(a, m, r) * lum + _masked_mean(b, m, r)

def pattern_weight(image, mask, lum, base, remove):
    """'여기는 무늬·낙서다'라고 볼 정도를 0~1로 매겨요.
    세 가지를 봄: (1) 작고 촘촘한 무늬  (2) 벽 바탕과 색이 다름  (3) 넓고 진한 얼룩"""
    H, W = mask.shape
    m = mask.astype(np.float32)
    r_small = max(2, int(0.015 * max(H, W)))
    r_big = max(6, int(0.08 * max(H, W)))
    detail = np.abs(lum - base) / PATTERN_DETAIL_T                     # (1) 잔무늬
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_a, ref_b = (np.median(lab[..., 1][mask]), np.median(lab[..., 2][mask])) if mask.any() else (0.0, 0.0)
    chroma = np.sqrt((lab[..., 1] - ref_a) ** 2 + (lab[..., 2] - ref_b) ** 2) / PATTERN_CHROMA_T   # (2) 색이 다름
    coarse = _masked_mean(lum, m, r_big)                               # 아주 넓게 본 밝기 = 조명
    tone = np.abs(base - coarse) / PATTERN_TONE_T                      # (3) 조명보다 유난히 어둡/밝은 넓은 자국
    w = np.clip(np.maximum(np.maximum(detail, chroma), tone), 0, 1) * float(remove) * m
    w = np.clip(cv2.GaussianBlur(w, (0, 0), r_small * 0.6) * 1.4, 0, 1)   # 무늬 가장자리까지 조금 넉넉히
    return w * m

def wall_shading(image, mask, remove=None):
    """벽의 '밝기 지도'를 만들어요 → (밝기, 무늬로 본 정도)
    remove=0이면 원본 밝기 그대로(=21번까지의 동작), 1에 가까울수록 무늬·낙서를 지우고 굴곡만 남겨요."""
    remove = WALL_PATTERN_REMOVE if remove is None else remove
    H, W = mask.shape
    lum = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    if remove <= 0 or not mask.any():
        return lum, np.zeros((H, W), np.float32)
    m = mask.astype(np.float32)
    r_small = max(2, int(0.015 * max(H, W)))
    base = smooth_keep_edges(lum, m, r_small)              # 굴곡·그림자는 남고 잔무늬는 사라진 밝기
    w = pattern_weight(image, mask, lum, base, remove)
    r_fill = max(4, int(0.05 * max(H, W)))
    keep = m * (1 - w)
    fill = _masked_mean(lum, keep, r_fill)                  # 무늬 자리를 '주변의 멀쩡한 밝기'로 메움
    shading = base * (1 - w) + fill * w
    return shading, w

def detail_shade(image, mask, pattern_remove=None, lo=0.45, hi=1.6):
    """원본의 밝기 무늬를 가져와요 (마스크 안쪽 평균을 1.0으로).
    벽에 턱이 있으면 턱 위/아래의 밝기 차이, 모서리 그림자, 창가 빛이 새 색 위에도 그대로 남음.
    pattern_remove > 0이면 벽지 무늬·낙서는 지우고 굴곡만 남겨요 (2단계 WALL_PATTERN_REMOVE)."""
    shading, _ = wall_shading(image, mask, pattern_remove)
    average = shading[mask].mean() if mask.any() else 128.0
    ratio = np.clip(shading / (average + 1e-6), 1e-3, None) ** dark_keep(average)   # 어두운 사진이면 명암 차이를 줄임
    return np.clip(ratio, lo, hi)

def blend_region(image, mask, painted, edge=None):
    """칠한 결과(painted)를 마스크 영역에만 합쳐요. edge: 가장자리 부드러움(픽셀). 안 주면 2단계의 EDGE_FEATHER."""
    edge = EDGE_FEATHER if edge is None else edge
    alpha = mask.astype(np.float32)
    if edge > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), edge)
    alpha = alpha[..., None]
    out = image.astype(np.float32) * (1 - alpha) + painted.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def _lin(x):
    """화면 밝기값(0~255)을 실제 빛의 양(0~1)으로 바꿔요. 빛은 이 단위에서 곱해야 자연스러움."""
    return (np.asarray(x, np.float32) / 255.0) ** 2.2

def _gamma(x):
    return np.clip(x, 0, 1) ** (1 / 2.2) * 255.0

def cap_luma(hex_code, max_luma=None):
    """벽 색이 너무 밝으면(하얗게 날아가면) 상한까지만 밝기를 낮춰요. 색감(어느 쪽 색인지)은 그대로.
    반환: (바뀐 색, 낮췄는지 여부)"""
    max_luma = WALL_MAX_LUMA if max_luma is None else max_luma
    rgb = np.array(hex_to_rgb(hex_code), np.float32) / 255.0
    luma = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    if luma <= max_luma or luma <= 0:
        return hex_code.upper(), False
    rgb = np.clip(rgb * (max_luma / luma), 0, 1)
    return "#{:02X}{:02X}{:02X}".format(*(rgb * 255).round().astype(int)), True

# ---- 몰딩 찾기 (분할 모델에는 몰딩 클래스가 없어서 직접 찾음) ----
def lighten_hex(hex_code, factor, max_luma=0.97):
    """색의 밝기만 올려요 (색감은 유지). 천장 색을 벽보다 살짝 밝게 만들 때 사용."""
    rgb = np.array(hex_to_rgb(hex_code), np.float32) / 255.0
    rgb = np.clip(rgb * float(factor), 0, 1)
    luma = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    if luma > max_luma and luma > 0:
        rgb = rgb * (max_luma / luma)
    return "#{:02X}{:02X}{:02X}".format(*(np.clip(rgb, 0, 1) * 255).round().astype(int))

def ceiling_color(wall_hex, mode=None):
    """시안의 벽 색에서 천장 색을 정해요. "auto"면 벽보다 살짝 밝게, "none"이면 안 칠함."""
    mode = CEILING_MODE if mode is None else mode
    if not mode or mode == "none":
        return None
    if mode == "auto":
        return lighten_hex(wall_hex, CEILING_LIGHTEN) if wall_hex else None
    return mode

def _find_band(wall, image, cols, inward, max_thick):
    """벽의 위(또는 아래) 가장자리에서 안쪽으로, 벽 바탕색과 다른 픽셀이 몇 줄이나 이어지는지 세요.
    바탕색은 세로줄마다 '띠가 끝난 다음 구간'에서 따로 구해요 → 벽에 밝기 변화(그라데이션)가 있어도 괜찮음."""
    H = wall.shape[0]
    first, last = wall.argmax(axis=0), H - 1 - wall[::-1].argmax(axis=0)
    y0 = (first if inward > 0 else last)[cols]
    pick = lambda ys: (np.clip(ys, 0, H - 1), (ys >= 0) & (ys < H))
    ref_y, ref_in = pick(y0[None, :] + inward * np.arange(max_thick, 3 * max_thick)[:, None])
    ref_ok = (wall[ref_y, cols[None, :]] & ref_in).astype(np.float32)
    ref = (image[ref_y, cols[None, :]].astype(np.float32) * ref_ok[..., None]).sum(axis=0) / (ref_ok.sum(axis=0)[:, None] + 1e-6)
    ys, inside = pick(y0[None, :] + inward * np.arange(max_thick)[:, None])
    d = np.sqrt(((image[ys, cols[None, :]].astype(np.float32) - ref[None, :, :]) ** 2).sum(axis=2))
    hit = (d > MOLDING_DIFF) & wall[ys, cols[None, :]] & inside
    return y0, np.cumprod(hit, axis=0).sum(axis=0)        # 가장자리부터 '끊기지 않고' 이어진 줄 수

def find_moldings(masks, image, max_thick=None, min_cols=None):
    """천장 몰딩(molding_top)과 걸레받이(molding_bottom)를 찾아요 → ({이름: 마스크}, {이름: 정보})
    방법: 벽의 위/아래 가장자리에서 안쪽으로, 벽 바탕색과 다른 띠가 일정 두께로 쭉 이어지는지 봄.
    못 찾으면 그 항목은 빠져요 (몰딩이 없는 집이거나, 벽과 색이 비슷하면 못 찾음)."""
    wall = masks["wall"]
    H, W = wall.shape
    out, info = {}, {}
    if not MOLDING_FIND or not wall.any():
        return out, info
    max_thick = int((MOLDING_MAX_THICK if max_thick is None else max_thick) * H)
    min_cols = MOLDING_MIN_COLS if min_cols is None else min_cols
    cols = np.where(wall.any(axis=0))[0]
    if len(cols) < 10 or max_thick < 3:
        return out, info
    for name, inward in (("molding_top", 1), ("molding_bottom", -1)):
        y0, thick = _find_band(wall, image, cols, inward, max_thick)
        seen = thick >= 2
        t = int(np.median(thick[seen])) if seen.any() else 0
        if seen.mean() < min_cols or t < 2 or t >= max_thick:    # 너무 얇거나, 상한까지 꽉 찼으면(벽 전체가 다른 색) 믿지 않음
            info[name] = {"found": False, "thickness": t, "cols_ratio": float(seen.mean())}
            continue
        band = np.zeros((H, W), bool)
        ys = np.clip(y0[None, :] + inward * np.arange(t)[:, None], 0, H - 1)
        band[ys, cols[None, :]] = True
        band &= wall
        out[name] = band
        info[name] = {"found": True, "thickness": t, "cols_ratio": float(seen.mean()), "ratio": float(band.mean())}
    return out, info

def molding_union(room):
    """찾은 몰딩을 하나로 합친 마스크 (없으면 None)."""
    mold = room.get("moldings") or {}
    if not mold:
        return None
    return np.logical_or.reduce(list(mold.values()))

def paint_fast(image, mask, source, detail=1.0, edge=None, soft_highlight=None, pattern_remove=None, norm_mask=None,
               shade_from=None):
    """[벽용] 마스크 영역을 칠해요. source: "#RRGGBB" 색상 코드, 또는 이미지 크기와 같은 (높이, 너비, 3) 배열.
    detail: 원본 밝기 무늬 반영 정도 (0=완전 평평, 1=원본 그대로, 1.3=대비 더 강하게)
    edge: 가장자리 부드러움(픽셀). 안 주면 2단계의 EDGE_FEATHER
    soft_highlight: 밝은 부분이 255를 넘어 하얗게 뭉개지지 않게 '남은 여유' 안에서만 밝히기 (끄면 19번과 같은 단순 곱셈)
    pattern_remove: 벽지 무늬·낙서를 지우는 정도 (0~1). 안 주면 2단계의 WALL_PATTERN_REMOVE
    shade_from: 빛(밝은 곳/그림자)을 읽을 사진. 안 주면 image에서 읽음. 벽을 면마다 나눠 칠할 때 원본을 넣으면
                칠한 결과가 다음 면의 밝기에 영향을 주지 않아요
    norm_mask: 밝기의 기준(평균)을 낼 영역. 몰딩만 빼고 칠할 때도 벽 전체를 기준으로 삼아 밝기가 달라지지 않게 함"""
    if not mask.any():
        return image
    edge = EDGE_FEATHER if edge is None else edge
    soft_highlight = WALL_SOFT_HIGHLIGHT if soft_highlight is None else soft_highlight
    light = image if shade_from is None else shade_from
    shade = 1.0 + detail * (detail_shade(light, mask if norm_mask is None else norm_mask, pattern_remove) - 1.0)
    if isinstance(source, str):
        color = np.array(hex_to_rgb(source), np.float32)[None, None, :]
    else:
        color = source.astype(np.float32)
    if soft_highlight:
        b = _lin(color)                                                   # 빛의 양으로 바꿔서 계산
        r = shade[..., None]
        room_left = np.maximum(0.995 - b, 0.005)                          # 더 밝아질 수 있는 여유
        brighter = b + room_left * (1 - np.exp(np.clip(-(r - 1) * b / room_left, -60, 0)))   # 밝아질 땐 여유 안에서 부드럽게
        painted = _gamma(np.where(r >= 1, brighter, b * r))                # 어두워질 땐 그냥 곱함 (그림자)
    else:
        painted = np.clip(color * shade[..., None], 0, 255)
    return blend_region(image, mask, painted, edge)

# ---- 벽지 사진을 벽에 깔기 (바닥재와 같은 생각: 원근 + 원래 방의 빛) ----
def _mip_sample(texture, u, v, foot):
    """무늬 좌표(u, v: 무늬 픽셀 단위)에서 색을 뽑아요.
    foot = 화면 1픽셀이 무늬 몇 픽셀에 해당하는지 → 크기별 축소본(미프맵)을 섞어 써서 깨알 번쩍임을 막음."""
    th, tw = texture.shape[:2]
    level = np.clip(np.log2(np.maximum(foot, 1.0)), 0, 6)
    out = np.zeros(u.shape + (3,), np.float32)
    for k in range(7):
        w = np.clip(1 - np.abs(level - k), 0, 1)
        if not (w > 0).any():
            continue
        s = 2 ** k
        tex_k = texture if k == 0 else cv2.resize(texture, (max(4, tw // s), max(4, th // s)), interpolation=cv2.INTER_AREA)
        hk, wk = tex_k.shape[:2]
        mx = ((u * wk / tw) % wk).astype(np.float32)
        my = ((v * hk / th) % hk).astype(np.float32)
        out += cv2.remap(tex_k, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32) * w[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def _lay_flat(shape, texture, tile, y_top):
    """정면 벽: 무늬를 곧게 반복해서 깔아요 (벽 맨 윗줄에서 시작)."""
    H, W = shape
    th, tw = texture.shape[:2]
    scale = tw / max(8.0, tile * W)                       # 화면 1픽셀 = 무늬 scale 픽셀
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return _mip_sample(texture, xs * scale, (ys - y_top) * scale, np.full((H, W), scale, np.float32))

WALL_CORNER_WRAP = True     # True: 벽이 꺾이는 모서리에서 벽지 무늬가 끊기지 않고 이어지게 (실제 벽지처럼 모서리를 돌아감)

def _lay_side(shape, texture, tile, x_corner, left, y_h, fov, y_top=None):
    """옆벽: 카메라에서 옆으로 떨어진 '세로 평면'으로 보고 깔아요 → 안쪽으로 갈수록 무늬가 좁아짐.
    픽셀 (x, y)가 보는 벽면 위의 점은 깊이 Z = fx / s, 높이 Y = (y - y_h) / s  (s = 화면 중앙에서 옆으로 떨어진 거리).
    WALL_CORNER_WRAP이면 모서리(x_corner)에서 정면 벽과 무늬의 **크기와 위치**가 딱 맞게 이어 붙임:
      - 크기: 두 벽이 공유하는 건 '세로 모서리선'이라 세로 방향 무늬 크기를 맞춤
      - 위치: 모서리선에서 정면 벽의 무늬 좌표(u, v)와 같은 값에서 시작해, 옆벽을 따라 이어서 셈"""
    H, W = shape
    th, tw = texture.shape[:2]
    cx, fx = W / 2.0, fov * W
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    s = np.maximum((cx - xs) if left else (xs - cx), 1.0)
    edge = max(abs(cx - x_corner), 1.0)
    if not WALL_CORNER_WRAP:                                 # 예전 방식 (모서리에서 무늬가 끊길 수 있음)
        T = max(tile * W * fx / (edge * edge), 1e-6)
        u = (fx / s) / T * tw
        v = ((ys - y_h) / s) / T * tw
        foot = np.maximum(fx / (s * s), 1.0 / s) / T * tw
        return _mip_sample(texture, u, v, foot)
    scale = tw / max(8.0, tile * W)                         # 정면 벽: 화면 1픽셀 = 무늬 scale 픽셀
    T = max(tile * W / edge, 1e-6)                          # 무늬 한 칸의 '세계 크기' (모서리에서 세로 크기가 정면 벽과 같아지게)
    Z, Zc = fx / s, fx / edge                               # 옆벽 위 깊이, 모서리의 깊이
    u_corner = x_corner * scale                              # 모서리에서 정면 벽의 가로 무늬 좌표
    along = (Z - Zc) if left else (Zc - Z)                   # 벽을 따라 왼쪽→오른쪽으로 걸을 때의 거리 (모서리=0)
    u = u_corner + along / T * tw
    y0 = y_h if y_top is None else y_top
    v = ((ys - y_h) / s) / T * tw + (y_h - y0) * scale        # 모서리에서 정면 벽의 세로 무늬 좌표와 같게
    foot = np.maximum(fx / (s * s), 1.0 / s) / T * tw
    return _mip_sample(texture, u, v, foot)

def estimate_wall_split(masks, slope_t=0.3, min_flat=0.12):
    """벽을 왼쪽/정면/오른쪽으로 나눌 세로 모서리 두 곳을 추정해요 → (x1, x2) 또는 None.
    바닥과 벽이 만나는 선이 '평평한 구간'이 정면 벽, 기울어진 구간이 옆벽이라고 봄."""
    floor = masks["floor"]
    H, W = floor.shape
    if floor.mean() < 0.03:
        return None
    seen = floor.any(axis=0)
    y = floor.argmax(axis=0).astype(np.float32)
    y = cv2.GaussianBlur(y.reshape(1, -1), (0, 0), max(2.0, W * 0.01)).ravel()
    flat = seen & (np.abs(np.gradient(y)) < slope_t)
    best, run, start = (0, 0, 0), 0, 0
    for x in range(W):
        if flat[x]:
            if run == 0:
                start = x
            run += 1
            if run > best[0]:
                best = (run, start, x)
        else:
            run = 0
    length, x1, x2 = best
    if length < min_flat * W or length > 0.92 * W:        # 평평한 구간이 너무 짧거나 거의 전부면 '정면 벽 하나'로 봄
        return None
    return int(x1), int(x2)

WALL_VANISHING = True       # True: 벽마다 '벽-바닥 / 벽-천장 경계선'으로 진짜 소실점을 찾아 원근을 줌 (비스듬한 벽은 약하게)
                            # False: 예전 방식 (옆벽을 카메라와 직각인 벽으로 가정 → 비스듬한 방에서 무늬가 길쭉해짐)

def _fit_line(xs, ys):
    """점들에 직선 y = a*x + b를 맞춰요 (튀는 점은 두 번 걸러냄) → (a, b) 또는 None."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    if len(xs) < 8 or np.ptp(xs) < 5:
        return None
    keep = np.ones(len(xs), bool)
    for _ in range(3):
        a, b = np.polyfit(xs[keep], ys[keep], 1)
        r = np.abs(ys - (a * xs + b))
        lim = max(2.0, 2.5 * np.median(r[keep]))
        new = r <= lim
        if new.sum() < 8 or (new == keep).all():
            break
        keep = new
    return float(a), float(b)

def _wall_planes(wall, floor, ceiling, H, W):
    """벽-바닥 경계선을 꺾이는 곳마다 잘라 '벽 평면'들로 나눠요 → [{xa, xb, floor:(a,b), ceil:(a,b)|None}] (왼쪽부터)
    경계가 거의 안 보이면 None."""
    edge = _edge_rows(floor, wall, True)
    xs = np.where(edge >= 0)[0]
    if len(xs) < 0.10 * W:
        return None
    pts = np.stack([xs, edge[xs]], 1).astype(np.float32).reshape(-1, 1, 2)
    poly = cv2.approxPolyDP(pts, 0.012 * max(H, W), False).reshape(-1, 2)
    cuts = [float(p[0]) for p in poly[1:-1]]                 # 꺾이는 x 위치
    min_len = 0.08 * W
    while cuts:                                              # 너무 짧은 조각은 이웃과 합침 (그 조각의 한쪽 자름선을 없앰)
        lens = np.diff([0.0] + cuts + [float(W)])
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        cuts.pop(i if i < len(cuts) else i - 1)
    bounds = [0] + [int(round(c)) for c in cuts] + [W]
    cedge = _edge_rows(wall, ceiling, False) if ceiling is not None and ceiling.any() else np.full(W, -1)
    planes = []
    for xa, xb in zip(bounds[:-1], bounds[1:]):
        m = xs[(xs >= xa) & (xs < xb)]
        fl = _fit_line(m, edge[m]) if len(m) >= 8 else None
        cm = np.where(cedge[xa:xb] >= 0)[0] + xa
        cl = _fit_line(cm, cedge[cm]) if len(cm) >= max(8, 0.3 * (xb - xa)) else None
        planes.append({"xa": xa, "xb": xb, "floor": fl, "ceil": cl})
    if not any(p["floor"] for p in planes):
        return None
    for i, p in enumerate(planes):                           # 경계가 안 보인 조각은 이웃 직선을 빌려 씀
        if p["floor"] is None:
            nb = [q for q in planes if q["floor"]]
            p["floor"] = min(nb, key=lambda q: abs((q["xa"] + q["xb"]) - (p["xa"] + p["xb"])))["floor"]
    return planes

def _lay_vanishing(shape, texture, tile, planes, y_top, y_h, fov):
    """벽 평면마다 자기 소실점으로 무늬를 깔고, 모서리에서 무늬 크기·위치를 이어 붙여요.
    벽 위 점의 무늬 좌표:  u = A ± K / s,  v = C·(y − y_h) / s + D   (s = 화면에서 소실점까지 가로 거리)
    정면 벽(경계선이 수평)은 원근 없이 u = m·x + A, v = m·(y − y_h) + D."""
    H, W = shape
    th, tw = texture.shape[:2]
    cx, fx = W / 2.0, fov * W
    scale = tw / max(8.0, tile * W)                          # 기준 크기: 화면 1픽셀 = 무늬 scale 픽셀
    ys = np.arange(H, dtype=np.float32)[:, None]
    U = np.zeros((H, W), np.float32); V = np.zeros((H, W), np.float32); F = np.ones((H, W), np.float32)
    D = scale * (y_h - y_top)                                # 벽 맨 윗줄에서 무늬가 시작되게 (모든 면 공통)
    x_ref = planes[0]["xb"] if len(planes) > 1 else (planes[0]["xa"] + planes[0]["xb"]) / 2.0
    prev = None                                              # (모서리에서의 u, 세로 기울기)
    for i, p in enumerate(planes):
        xa, xb = p["xa"], p["xb"]
        xs = np.arange(xa, xb, dtype=np.float32)[None, :]
        a, b = p["floor"]
        vx = None
        if abs(a) > 0.015:                                   # 기울어진 경계선 → 소실점 있음
            vx = (y_h - b) / a
            if xa - 0.02 * W < vx < xb + 0.02 * W or abs(vx - cx) > 60 * W:
                vx = None                                    # 소실점이 벽 안쪽이면 믿지 않음 / 너무 멀면 정면 벽
        x0 = float(xa) if i > 0 else float(x_ref)            # 크기를 맞출 기준 열 (첫 면은 기준 열, 나머지는 왼쪽 모서리)
        if vx is None:                                       # 정면 벽
            m = scale if prev is None else prev[1]
            A = (0.0 - m * x0) if prev is None else (prev[0] - m * x0)
            U[:, xa:xb] = m * xs + A
            V[:, xa:xb] = m * (ys - y_h) + D
            F[:, xa:xb] = m
            prev = (m * xb + A, m)
        else:
            sgn = 1.0 if vx > xb else -1.0
            s_of = lambda x: np.maximum(sgn * (vx - x), 0.02 * W)
            Dz = 1.0 / np.sqrt(1.0 + ((vx - cx) / fx) ** 2)
            s0 = float(s_of(x0))
            C = (scale if prev is None else prev[1]) * s0
            K = C * fx / Dz
            A = (0.0 - sgn * K / s0) if prev is None else (prev[0] - sgn * K / s0)
            s = s_of(xs)
            U[:, xa:xb] = A + sgn * K / s
            V[:, xa:xb] = C * (ys - y_h) / s + D
            F[:, xa:xb] = np.maximum(K / (s * s), C / s)
            se = float(s_of(float(xb)))
            prev = (A + sgn * K / se, C / se)
    return _mip_sample(texture, U, V, F)

def lay_wall_texture(shape, mask, texture, tile=None, mode=None, split=None, horizon=None, fov=None,
                     floor=None, ceiling=None):
    """벽 전체에 벽지를 깔아요 → (높이, 너비, 3) 배열. 옆벽에는 원근을 줌.
    floor·ceiling 마스크를 주고 WALL_VANISHING이면, 벽마다 진짜 소실점을 찾아 원근을 줘요 (못 찾으면 예전 방식)."""
    H, W = shape
    mode = WALL_MODE if mode is None else mode
    tile = WALL_TILE if tile is None else tile
    fov = FOV if fov is None else fov
    rows = np.where(mask.any(axis=1))[0]
    y_top = int(rows.min()) if len(rows) else 0
    if mode == "perspective" and WALL_VANISHING and floor is not None and floor.any():
        try:
            planes = _wall_planes(mask, floor, ceiling, H, W)
            if planes:
                vys = []                                     # 수평선: 벽-바닥 선과 벽-천장 선이 만나는 높이들의 중앙값
                for p in planes:
                    if p["ceil"]:
                        (a1, b1), (a2, b2) = p["floor"], p["ceil"]
                        if abs(a1 - a2) > 0.01:
                            xv = (b2 - b1) / (a1 - a2)
                            vys.append(a1 * xv + b1)
                vys = [v for v in vys if -1.0 * H < v < 2.0 * H]
                if horizon is not None:
                    y_h = horizon * H
                elif vys:
                    y_h = float(np.median(vys))
                else:
                    y_h = estimate_horizon(floor, HORIZON)
                return _lay_vanishing((H, W), texture, tile, planes, y_top, y_h, fov)
        except Exception as e:
            print(f"⚠️ 벽 소실점 계산을 건너뛰었어요 ({type(e).__name__}: {str(e)[:80]})")
    out = _lay_flat((H, W), texture, tile, y_top)
    if mode == "perspective" and split:
        x1, x2 = split
        y_h = (HORIZON if horizon is None else horizon)
        y_h = (0.5 if y_h is None else y_h) * H
        margin = 0.02 * W                                     # 옆벽 원근은 모서리가 화면 가운데의 바깥쪽에 있을 때만
        if 2 < x1 < W / 2.0 - margin:                         # (아니면 가운데 쪽에서 계산이 망가져 무늬가 흰 단색처럼 뭉개짐)
            out[:, :x1] = _lay_side((H, W), texture, tile, x1, True, y_h, fov, y_top)[:, :x1]
        if W / 2.0 + margin < x2 < W - 2:
            out[:, x2:] = _lay_side((H, W), texture, tile, x2, False, y_h, fov, y_top)[:, x2:]
    return out

def wall_lighting(image, mask, base, keep=None, spec=None, exposure=None, pattern=None, sigma_ratio=0.012):
    """벽지(base)에 원래 방의 빛을 입혀요. **원래 벽의 무늬(꽃무늬 등)는 지우고 빛만** 가져와요.
    바닥에 쓰던 방식과 같되, 빛의 출처로 '무늬를 지운 밝기'를 써서 옛 무늬가 비치지 않게 함."""
    keep = WALL_LIGHT if keep is None else keep
    spec = WALL_SPEC if spec is None else spec
    exposure = WALL_EXPOSURE if exposure is None else exposure
    pattern = WALL_PATTERN_REMOVE if pattern is None else pattern
    H, W = mask.shape
    shading, w = wall_shading(image, mask, pattern)        # 무늬를 지운 밝기 (턱·그림자·창가 빛은 남음)
    lin = _lin(shading)
    m = mask.astype(np.float32)
    average = (lin * m).sum() / (m.sum() + 1e-6)
    dk = dark_keep(average ** (1 / 2.2) * 255.0)           # 어두운 사진이면 명암·반짝임을 줄임
    diffuse = np.clip(lin / (average + 1e-6), 0.15, 4.0) ** (keep * dk)
    sigma = max(3.0, sigma_ratio * max(H, W))
    low = cv2.GaussianBlur(lin * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)
    raw = _lin(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    glare = np.clip(raw - low * 1.08, 0, None) / (average + 1e-6) * (1 - w) * dk    # 무늬로 본 자리는 반짝임에서 제외
    b = _lin(base) * exposure
    r = diffuse[..., None]
    room_left = np.maximum(0.97 - b, 0.03)
    brighter = b + room_left * (1 - np.exp(-(r - 1) * b / room_left))
    out = np.where(r >= 1, brighter, b * r)
    return _gamma(out + spec * 0.5 * np.clip(glare, 0, 2)[..., None])

def load_texture(path, max_side=1024):
    """바닥재 사진을 읽어요 (너무 크면 줄임). 위에서 내려다본, 이음새 없이 반복되는 사진이 좋아요."""
    img = Image.open(path).convert("RGB")
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return np.array(img)

def sample_marble(size=512, seed=0):
    """[테스트용] 가짜 대리석 무늬 (이음새 없이 반복됨). 진짜 사진이 생기면 load_texture로 바꿔 끼우면 돼요."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    turb = np.zeros((size, size))
    for _ in range(6):
        kx, ky = rng.integers(1, 6, 2)
        turb += rng.uniform(0.15, 0.5) / (kx + ky) * np.sin(2 * np.pi * (kx * x + ky * y) + rng.uniform(0, 6.28))
    vein = np.exp(-np.abs(np.sin(2 * np.pi * (2 * x + 3 * y + 3 * turb))) * 4)
    vein2 = np.exp(-np.abs(np.sin(2 * np.pi * (3 * x - 2 * y + 4 * turb))) * 8) * 0.6
    cloud = 0.5 + 0.5 * np.sin(2 * np.pi * (x + 2 * turb)) * np.cos(2 * np.pi * (y - turb))
    base = np.array([238.0, 235.0, 228.0]) - cloud[..., None] * 14
    out = base - (vein + vein2)[..., None] * np.array([78.0, 80.0, 86.0])
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_horizon(mask, horizon=None):
    """수평선 높이(픽셀)를 정해요. horizon을 주면 그 비율(0~1)을, 없으면 사진 중간쯤을 쓰되 바닥 맨 윗줄보다는 위로 잡아요."""
    H = mask.shape[0]
    rows = np.where(mask.any(axis=1))[0]
    top = rows.min() if len(rows) else H // 2
    y_h = (horizon if horizon is not None else 0.5) * H
    return float(min(y_h, top - 0.02 * H))

def floor_perspective(shape, mask, texture, tile=0.5, horizon=None, fov=0.9, rotate=0.0):
    """바닥 영역에 원근을 맞춰 무늬를 깔아요 → (높이, 너비, 3) 배열.
    카메라가 바닥 위에서 수평으로 보고 있다고 가정: 수평선 위치(y_h)만 알면 화면 아래 픽셀이 바닥의 어느 지점인지 계산됨.
      깊이 Z = fx / (y - y_h),  좌우 X = (x - W/2) / (y - y_h)   (카메라 높이 = 1로 둠)
    멀리 갈수록 무늬가 작아지므로 7단계 크기별 축소본(미프맵)에서 골라 써서 깨알 같은 번쩍임을 막음."""
    H, W = shape
    y_h = estimate_horizon(mask, horizon)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.maximum(ys - y_h, 1.0)
    fx = fov * W
    Z = fx / d
    X = (xs - W / 2.0) / d
    if rotate:
        a = np.deg2rad(rotate)
        X, Z = X * np.cos(a) - Z * np.sin(a), X * np.sin(a) + Z * np.cos(a)
    rows = np.where(mask.any(axis=1))[0]
    y_near = rows.max() if len(rows) else H - 1
    tile_world = tile * W / max(y_near - y_h, 1.0)             # 화면 맨 아래에서 무늬 한 칸이 이미지 가로의 tile 비율이 되도록
    th, tw = texture.shape[:2]
    u = X / tile_world * tw
    v = Z / tile_world * tw                                    # 무늬의 가로세로 비율 유지 (세로 한 칸 = 가로 한 칸 × th/tw)
    # 화면 한 픽셀이 무늬 몇 픽셀에 해당하는지 → 어느 크기의 축소본을 쓸지
    foot = np.maximum(tw / (tile_world * d), fx * tw / (tile_world * d * d))
    level = np.clip(np.log2(np.maximum(foot, 1.0)), 0, 6)
    out = np.zeros((H, W, 3), np.float32)
    for k in range(7):
        w = np.clip(1 - np.abs(level - k), 0, 1)
        if not (w > 0).any():
            continue
        s = 2 ** k
        tex_k = texture if k == 0 else cv2.resize(texture, (max(4, tw // s), max(4, th // s)), interpolation=cv2.INTER_AREA)
        hk, wk = tex_k.shape[:2]
        mx = ((u * wk / tw) % wk).astype(np.float32)
        my = ((v * hk / th) % hk).astype(np.float32)
        sample = cv2.remap(tex_k, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out += sample.astype(np.float32) * w[..., None]
    mean_color = texture.reshape(-1, 3).mean(axis=0)
    fade = np.clip((np.log2(np.maximum(foot, 1.0)) - 6.0) / 1.5, 0, 1)[..., None]   # 아주 먼 곳은 무늬 평균색으로 자연스럽게
    out = out * (1 - fade) + mean_color[None, None, :] * fade
    return np.clip(out, 0, 255).astype(np.uint8)

def floor_lighting(image, mask, base, keep=1.0, spec=1.0, exposure=1.0, sigma_ratio=0.012):
    """바닥재(base)에 원래 방의 빛을 입혀요. 빛을 두 가지로 나눠서 따로 다룸.
      1) 퍼진 빛(밝은 곳/그림자): 원본 바닥 밝기를 살짝 흐려서(원래 바닥 무늬는 지우고 빛만 남김) 새 바닥재에 곱함
      2) 반짝임(하이라이트): 주변보다 확 밝은 부분만 따로 뽑아서 흰빛으로 다시 얹음
    밝은 바닥재일 때 빛이 하얗게 날아가던 문제를 막으려고, 밝아지는 쪽은 \'남은 여유(1에서 모자란 만큼)\'를 넘지 않게 부드럽게 눌러줌.
    exposure: 바닥재 자체의 밝기 배율. 아주 하얀 바닥재라면 0.8~0.9로 낮춰야 빛의 모양이 보일 여유가 생김."""
    H, W = mask.shape
    lin = _lin(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    m = mask.astype(np.float32)
    sigma = max(3.0, sigma_ratio * max(H, W))
    low = cv2.GaussianBlur(lin * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)   # 마스크 안쪽만 보는 흐린 밝기
    average = (lin * m).sum() / (m.sum() + 1e-6)
    ratio = np.clip(low / (average + 1e-6), 0.15, 4.0)
    dk = dark_keep(average ** (1 / 2.2) * 255.0)                     # 어두운 사진이면 명암·반짝임을 줄임
    diffuse = ratio ** (keep * dk)                                   # keep=0이면 1(빛 없음), 1이면 원본 그대로
    glare = np.clip(lin - low * 1.08, 0, None) / (average + 1e-6) * dk    # 주변 빛보다 8% 넘게 밝은 부분
    b = _lin(base) * exposure
    r = diffuse[..., None]
    room_left = np.maximum(0.97 - b, 0.03)                              # 더 밝아질 수 있는 여유
    brighter = b + room_left * (1 - np.exp(-(r - 1) * b / room_left))    # 밝아질 땐 여유 안에서 부드럽게 (기울기는 곱셈과 같게 시작)
    out = np.where(r >= 1, brighter, b * r)                              # 어두워질 땐 그냥 곱함 (그림자)
    out = out + spec * 0.5 * np.clip(glare, 0, 2)[..., None]
    return _gamma(out)

def floor_reflection(scene, mask, y_h, y_near, rough=0.3):
    """바닥에 비친 모습(거울상)을 만들어요 → (반사 이미지, 세기 지도).
    바닥 윗가장자리(벽과 만나는 선)를 거울면으로 보고, 선 아래 픽셀은 선 위의 같은 거리만큼 위쪽 장면을 비춤.
    멀리 있는 바닥일수록 더 잘 비치도록(프레넬 느낌) 세기를 줌. 가구는 벽 기준으로 대충 비추므로 근사임."""
    H, W = mask.shape
    cols = mask.any(axis=0)
    base_y = np.where(cols, mask.argmax(axis=0), 0).astype(np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    map_y = 2.0 * base_y[None, :] - ys
    valid = ((map_y >= 0) & (map_y < ys)).astype(np.float32)
    map_yc = np.clip(map_y, 0, H - 1).astype(np.float32)
    sharp = cv2.remap(scene, xs, map_yc, cv2.INTER_LINEAR).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 0.025 * max(H, W))
    refl = sharp * (1 - rough) + blurred * rough
    src_is_floor = cv2.remap(mask.astype(np.float32), xs, map_yc, cv2.INTER_LINEAR)
    weight = valid * (1 - np.clip(src_is_floor * 1.5, 0, 1)) * np.clip(map_y / (0.08 * H), 0, 1)   # 바닥이 바닥을 비추거나 사진 맨 위는 약하게
    t = np.clip((ys - y_h) / max(y_near - y_h, 1.0), 0, 1)           # 0=먼 곳, 1=가까운 곳
    fresnel = 0.25 + 0.75 * (1 - t) ** 2
    return refl, weight * fresnel

def light_floor(room, scene, base, keep=None, spec=None, reflect=None, rough=None, exposure=None):
    """바닥재(base: 색상 코드 또는 배열)에 빛(퍼진 빛 + 하이라이트)과 반사를 적용한 바닥 이미지를 만들어요."""
    keep = FLOOR_LIGHT if keep is None else keep
    spec = FLOOR_SPEC if spec is None else spec
    reflect = FLOOR_REFLECT if reflect is None else reflect
    rough = FLOOR_ROUGH if rough is None else rough
    exposure = FLOOR_EXPOSURE if exposure is None else exposure
    mask = room["masks"]["floor"]
    H, W = mask.shape
    if isinstance(base, str):
        base = np.full((H, W, 3), hex_to_rgb(base), np.uint8)
    lit = floor_lighting(room["image"], mask, base, keep, spec, exposure)
    if reflect > 0:
        rows = np.where(mask.any(axis=1))[0]
        y_h = estimate_horizon(mask, HORIZON)
        refl, strength = floor_reflection(scene, mask, y_h, rows.max(), rough)
        w = np.clip(reflect * strength, 0, 1)[..., None]
        lit = lit * (1 - w) + refl * w
    return lit

def split_wall_faces(masks, split=None, min_ratio=None):
    """벽 마스크를 모서리 기준으로 쪼개요 → ({"wall1": 마스크, ...}, 나눠졌는지 여부)
    **왼쪽부터 번호**를 매김 (백엔드 약속). 모서리를 못 찾았으면 벽 전체가 wall1 하나가 됨.
    너무 작은 조각(min_ratio 미만)은 가장 가까운 면에 합쳐서, 벽 픽셀이 하나도 빠지지 않게 함."""
    min_ratio = WALL_FACE_MIN if min_ratio is None else min_ratio
    wall = masks["wall"]
    H, W = wall.shape
    if not wall.any():
        return {}, False
    if not split:
        return {"wall1": wall.copy()}, False
    x1, x2 = sorted(int(x) for x in split)
    bounds = [0, max(0, min(W, x1)), max(0, min(W, x2)), W]
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= 0:
            continue
        piece = np.zeros((H, W), bool)
        piece[:, a:b] = wall[:, a:b]
        pieces.append(piece)
    big = [i for i, p in enumerate(pieces) if p.mean() >= min_ratio]
    if not big:
        return {"wall1": wall.copy()}, False
    merged = {i: pieces[i].copy() for i in big}
    for i, p in enumerate(pieces):                       # 작은 조각은 가장 가까운 면에 합침
        if i not in merged:
            merged[min(big, key=lambda k: abs(k - i))] |= p
    faces = {f"wall{n}": merged[i] for n, i in enumerate(sorted(merged), 1)}
    return faces, len(faces) > 1

PATTERN_SCORE_SIDE = 512          # 점수를 낼 때 사진을 이 크기로 줄여서 봄 (사진 잡티·벽지 결은 뭉개지고 무늬만 남게)

def wall_pattern_score(image, mask):
    """이 벽에 무늬·낙서가 얼마나 있는지 0~1로 (무늬로 본 면적의 비율).
    0에 가까우면 민무늬 벽(필터가 예쁨), WALL_PATTERN_T를 넘으면 꽃무늬 같은 벽(덮어씌우는 게 예쁨).
    작게 줄여서 보기 때문에 **사진 잡티나 벽지 결 같은 1~2픽셀짜리는 무늬로 치지 않아요.**"""
    if not mask.any():
        return 0.0
    s = PATTERN_SCORE_SIDE / max(image.shape[:2])
    if s < 1:                                              # 줄여 보기 (잡티 제거 + 빠름)
        size = (max(16, int(image.shape[1] * s)), max(16, int(image.shape[0] * s)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        if not mask.any():
            return 0.0
    image = cv2.GaussianBlur(image, (0, 0), 1.6)           # 한 번 더 뭉개기: 결·잡티는 사라지고 '무늬'만 남음
    lum = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    r_small = max(2, int(0.015 * max(mask.shape)))
    base = smooth_keep_edges(lum, mask.astype(np.float32), r_small)
    w = pattern_weight(image, mask, lum, base, 1.0)
    return float(w[mask].mean())

def wall_method_for(room, area, method=None):
    """이 벽(면)을 덮어씌울지 필터로 입힐지 정해요 → "overlay" 또는 "filter" (점수는 방에 저장해 재사용)."""
    method = WALL_METHOD if method is None else method
    if method in ("overlay", "filter"):
        return method
    cache = room.setdefault("_pattern_score", {})
    key = (int(area.sum()), int(np.argmax(area.any(axis=0))))      # 면마다 따로 기억
    if key not in cache:
        cache[key] = wall_pattern_score(room["image"], area)
    return "overlay" if cache[key] >= WALL_PATTERN_T else "filter"

def solid_layer(hex_code, shape):
    """색상 코드를 '그 색으로 꽉 찬 사진'으로 (덮어씌우기 방식에서 벽지 사진 대신 씀)."""
    return np.broadcast_to(np.array(hex_to_rgb(cap_luma(hex_code)[0]), np.uint8), (shape[0], shape[1], 3))

def wall_report(room):
    """면마다 무늬 점수와 고른 방식을 알려줘요 (9단계에서 확인용)."""
    for name, m in (room.get("wall_faces") or {"wall": room["masks"]["wall"]}).items():
        score = wall_pattern_score(room["image"], m)
        how = "덮어씌우기(무늬 있음)" if score >= WALL_PATTERN_T else "필터(민무늬)"
        print(f"  {name}: 무늬 점수 {score:.2f} → {how}" + ("" if WALL_METHOD == "auto" else f"  (WALL_METHOD={WALL_METHOD}로 고정)"))

def wall_laid_for(room, texture, tile=None):
    """벽지를 깔아 둔 결과를 방에 캐시해요 (같은 벽지·같은 크기면 다시 계산 안 함)."""
    cache = room.setdefault("_wall_laid", {})
    key = (id(texture), tile)
    if key not in cache:
        cache[key] = lay_wall_texture(room["image"].shape[:2], room["masks"]["wall"], texture, tile,
                                      split=room.get("wall_split"),
                                      floor=room["masks"].get("floor"), ceiling=room["masks"].get("ceiling"))
    return cache[key]

def recolor(room, wall=None, floor=None, detail=1.0, edge=None, floor_laid=None, wall_laid=None, walls=None,
            ceiling_hex=None, molding_mode=None, molding=None, pattern=None, wall_tile=None, method=None,
            keep=None, spec=None, reflect=None, rough=None, exposure=None,
            wall_keep=None, wall_spec=None, wall_exposure=None):
    """준비된 방에 벽(색 또는 벽지) · 천장 · 몰딩 · 바닥을 바로 입혀요. 분할은 다시 안 함.
    wall: 벽 전체에 쓸 재료 — "#RRGGBB" 색상 코드 또는 벽지 사진 배열
    walls: 벽 면마다 다르게 — {"wall1": "#RRGGBB" 또는 벽지 배열, "wall2": ...}  (wall보다 우선)
    molding: 몰딩에 쓸 재료 (색상 코드/사진 배열). {"molding_top": ..., "molding_bottom": ...}로 주면 **위아래를 따로** 칠해요.
             안 주면 molding_mode를 따름
    ceiling_hex: 천장 색 ("auto"면 벽 재료보다 살짝 밝게, "none"이면 안 칠함)
    method: "auto"(기본, 무늬가 있으면 덮어씌우고 없으면 필터) / "overlay"(항상 덮어씌움) / "filter"(항상 필터)
    pattern: 덮어씌울 때 옛 무늬를 지우는 정도 (0~1). 필터 방식에서는 쓰지 않음
    keep/spec/reflect/rough/exposure: 바닥의 빛·하이라이트·반사·반사 흐림·밝기
    wall_keep/wall_spec/wall_exposure: 벽의 빛·반짝임·벽지 밝기"""
    out = np.array(room["image"])
    masks = room["masks"]
    mode = MOLDING_MODE if molding_mode is None else molding_mode
    mold = molding_union(room)
    faces = room.get("wall_faces") or {}

    items = []                                            # [(칠할 영역, 재료)]
    if walls:
        for name, mat in walls.items():
            m = faces.get(name)
            if m is not None and m.any() and mat is not None:
                items.append((m, mat))
    elif wall_laid is not None:
        items.append((masks["wall"], wall_laid))
    elif wall is not None:
        items.append((masks["wall"], wall))
    first_hex = None
    for area, mat in items:
        if mold is not None and (mode != "wall" or molding is not None):
            area = area & ~mold                           # 몰딩은 따로 처리하니 벽에서 뺌
        if not area.any():
            continue
        if isinstance(mat, str):
            first_hex = first_hex or mat
            if wall_method_for(room, area, method) == "filter":          # 민무늬 벽: 원래 결을 그대로 곱함
                out = paint_fast(out, area, mat, detail, edge, pattern_remove=0, norm_mask=masks["wall"],
                                 shade_from=room["image"])    # 빛은 늘 원본에서 읽음 → 면을 칠하는 순서가 결과를 바꾸지 않음
            else:                                                        # 무늬 있는 벽: 바닥재처럼 덮고 빛만 가져옴
                painted = wall_lighting(room["image"], masks["wall"], solid_layer(mat, out.shape),
                                        wall_keep, wall_spec, wall_exposure, pattern)
                out = blend_region(out, area, painted, edge)
        else:
            laid = mat if (wall_laid is not None and mat is wall_laid) else wall_laid_for(room, mat, wall_tile)
            first_hex = first_hex or avg_hex(laid)
            painted = wall_lighting(room["image"], masks["wall"], laid, wall_keep, wall_spec, wall_exposure, pattern)
            out = blend_region(out, area, painted, edge)   # 빛은 벽 전체 기준, 칠하기는 이 면에만
    if mold is not None and items:
        default = None if mode in ("wall", "keep") else mode
        parts = []                                            # 천장 몰딩과 걸레받이를 **따로** 칠할 수 있어요
        if isinstance(molding, dict):                         # {"molding_top": 재료, "molding_bottom": 재료}
            for key in ("molding_top", "molding_bottom"):
                m = (room.get("moldings") or {}).get(key)
                mat = molding.get(key, default)
                if m is not None and m.any() and mat is not None:
                    parts.append((m, mat))
        else:
            mat = molding if molding is not None else default
            if mat is not None:
                parts.append((mold, mat))
        for m, mat in parts:
            if isinstance(mat, str):
                out = paint_fast(out, m, mat, detail, edge, pattern_remove=0, shade_from=room["image"])   # 몰딩은 늘 필터 (민무늬라 결이 살아야 예쁨)
            else:
                laid = wall_laid_for(room, mat, wall_tile)
                out = blend_region(out, m, wall_lighting(room["image"], masks["wall"], laid, wall_keep, wall_spec, wall_exposure, pattern), edge)

    ceil = ceiling_color(first_hex, ceiling_hex) if (ceiling_hex != "none") else None
    if ceil and masks["ceiling"].any():
        out = paint_fast(out, masks["ceiling"], ceil, detail, edge, pattern_remove=pattern, shade_from=room["image"])
    if floor_laid is None and floor is not None and not isinstance(floor, str):
        floor_laid = floor_perspective(out.shape[:2], masks["floor"], floor, FLOOR_TILE, HORIZON, FOV, ROTATE)
    base = floor_laid if floor_laid is not None else floor
    if base is not None and masks["floor"].any():
        painted = light_floor(room, out, base, keep, spec, reflect, rough, exposure)       # 반사에는 새 벽이 비침
        out = blend_region(out, masks["floor"], painted, edge)
    return out

def apply_masks(room, masks, moldings=None):
    """마스크를 직접 바꾼 뒤 벽 모서리·벽 면·몰딩을 다시 계산해요 (분할과 마스크 정리는 다시 안 함).
    moldings를 주면 그걸 그대로 쓰고(사용자가 붓으로 고친 경우), 안 주면 다시 찾아요."""
    room["masks"] = masks
    room["wall_split"] = tuple(WALL_SPLIT) if WALL_SPLIT else estimate_wall_split(masks)
    if moldings is not None:
        room["moldings"], room["molding_info"] = moldings, {k: {"found": True, "edited": True} for k in moldings}
    elif MOLDING_FIND:
        room["moldings"], room["molding_info"] = find_moldings(masks, room["image"])
    else:
        room["moldings"], room["molding_info"] = {}, {}
    room["wall_faces"], room["wall_split_found"] = split_wall_faces(masks, room["wall_split"])
    room.pop("laid", None)
    room.pop("_wall_laid", None)
    return room

def set_wall_split(room, split):
    """벽 모서리를 직접 정해요. split=(x1, x2)면 그 자리로, None이면 자동으로 다시 찾음."""
    global WALL_SPLIT
    WALL_SPLIT = tuple(int(x) for x in split) if split else None
    return apply_masks(room, room["masks"], room.get("moldings") or None)

# 5단계: 바닥재·벽지 목록 만들기 + CLIP 벡터 (파일 이름은 아무거나 OK)
import os, io

FLOOR_EXTS = (".jpg", ".jpeg", ".png", ".webp")
_floor_cache = {}       # 바닥재 id → 읽어 둔 사진 (한 번만 읽음)

# ---- CLIP: 사진/글을 같은 공간의 숫자(벡터)로 바꿔서 서로 얼마나 닮았는지 비교 ----
_clip_cache = {}
def get_clip():
    """CLIP 모델을 처음 한 번만 불러와요."""
    if "m" not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor
        print(f"  CLIP 불러오는 중: {CLIP_MODEL}")
        _clip_cache["m"] = (CLIPProcessor.from_pretrained(CLIP_MODEL), CLIPModel.from_pretrained(CLIP_MODEL).eval().to(DEVICE))
    return _clip_cache["m"]

def _to_vectors(out):
    """CLIP 출력을 길이 1로 맞춘 numpy 벡터로 (transformers 버전에 따라 출력 모양이 달라서 둘 다 처리)."""
    if not torch.is_tensor(out):
        out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
    v = out.detach().cpu().numpy().astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def embed_images(images, batch=16):
    """PIL 사진 리스트 → (장수, 차원) 벡터"""
    processor, model = get_clip()
    outs = []
    for i in range(0, len(images), batch):
        inputs = processor(images=[im.convert("RGB") for im in images[i:i + batch]], return_tensors="pt")
        with torch.no_grad():
            outs.append(_to_vectors(model.get_image_features(pixel_values=inputs["pixel_values"].to(DEVICE))))
    return np.concatenate(outs)

def embed_texts(texts):
    """글 리스트 → (개수, 차원) 벡터"""
    processor, model = get_clip()
    inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model.get_text_features(input_ids=inputs["input_ids"].to(DEVICE), attention_mask=inputs["attention_mask"].to(DEVICE))
    return _to_vectors(out)

def avg_hex(path_or_array):
    """바닥재 사진의 평균 색을 #RRGGBB로."""
    img = Image.fromarray(path_or_array) if isinstance(path_or_array, np.ndarray) else Image.open(path_or_array).convert("RGB")
    small = np.array(img.resize((32, 32), Image.BILINEAR)).reshape(-1, 3).mean(axis=0).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*small)

def _read_floor_raw(info):
    """바닥재 한 장을 배열로 읽어요 (path 또는 url)."""
    if info.get("path"):
        return load_texture(info["path"])
    import requests
    r = requests.get(info["url"], timeout=20); r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    s = 1024 / max(img.size)
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return np.array(img)

def make_floor_vector(info):
    """[백엔드용] 바닥재 한 장({"path": ...} 또는 {"url": ...})의 CLIP 벡터를 리스트로 돌려줘요. 바닥재를 등록할 때 한 번 계산해서 DB에 저장해 두면 돼요."""
    return embed_images([Image.fromarray(_read_floor_raw(info))])[0].tolist()

def _finish_library(entries):
    """읽어 둔 바닥재들에 평균 색과 CLIP 벡터를 채워요. 벡터가 이미 있는 건 그대로 쓰고, 없는 것만 한꺼번에 계산."""
    need = [fid for fid, e in entries.items() if e.get("vec") is None]
    if need:
        print(f"  바닥재 {len(need)}개를 CLIP으로 벡터화 중...")
        for fid, v in zip(need, embed_images([Image.fromarray(entries[fid]["raw"]) for fid in need])):
            entries[fid]["vec"] = v
    library = {}
    for fid, e in entries.items():
        v = np.asarray(e["vec"], np.float32)
        library[fid] = {"path": e.get("path"), "url": e.get("url"), "name": e.get("name"),
                        "avg_hex": avg_hex(e["raw"]), "vec": v / (np.linalg.norm(v) + 1e-9)}
    return library

def solid_swatch(hex_code, size=224):
    """색상 코드 → 그 색으로 꽉 찬 작은 견본 사진. 사진 없이 색만 있는 벽지·페인트를 CLIP으로 고르려고 만들어요."""
    return np.full((size, size, 3), np.array(hex_to_rgb(hex_code), np.uint8), np.uint8)

def _build_library(items, label="바닥재", prefix=""):
    """[백엔드 연결용] 목록을 받아 라이브러리를 만들어요 (바닥재·벽지 공용).
    items: [{"id": 아무 이름이나 번호, "url": "https://..." / "path": "/파일/경로" / "hex": "#RRGGBB", "vec": [...] (선택), "name": "퓨어화이트" (선택)}]
    사진(url·path)이 없고 **색상 코드(hex)만 있어도 돼요** — 그 색으로 꽉 찬 견본을 만들어 CLIP에 보여줘요.
    사진과 hex를 같이 주면 사진을 씁니다.
    vec(CLIP 벡터)을 같이 주면 계산을 건너뛰어 빨라요 (make_floor_vector로 미리 만들어 DB에 저장해 둔 값)."""
    entries = {}
    for it in items:
        fid = str(it["id"])
        info = {"path": it.get("path"), "url": it.get("url")}
        if not (info["path"] or info["url"]) and not it.get("hex"):
            print(f"  {label} '{fid}'에 path·url·hex가 모두 없어 건너뜀"); continue
        try:
            _floor_cache.pop(prefix + fid, None)
            raw = solid_swatch(it["hex"]) if not (info["path"] or info["url"]) else _read_floor_raw(info)
            entries[fid] = {**info, "raw": raw, "vec": it.get("vec"), "name": it.get("name")}
        except Exception as e:
            print(f"  {label} '{fid}'를 읽지 못해 건너뜀: {type(e).__name__}: {str(e)[:80]}")
    if not entries:
        raise RuntimeError(f"쓸 수 있는 {label}가 하나도 없어요")
    library = _finish_library(entries)
    print(f"{label} {len(library)}개 등록")
    return library

def set_floor_library(items):
    """[백엔드 연결용] 바닥재 목록을 통째로 바꿔요."""
    global FLOOR_LIBRARY
    FLOOR_LIBRARY = _build_library(items, "바닥재", "")
    return FLOOR_LIBRARY

def set_wall_library(items):
    """[백엔드 연결용] 벽지 목록을 통째로 바꿔요."""
    global WALL_LIBRARY
    WALL_LIBRARY = _build_library(items, "벽지", "wall:")
    return WALL_LIBRARY

def load_floor_library(folder=None, label="바닥재", fallback=True):
    """사진 폴더를 읽어 {id(파일 이름): {"path", "avg_hex", "vec"}} 목록을 만들어요 (바닥재·벽지 공용).
    fallback=True인데 폴더가 없거나 비어 있으면 가짜 대리석 하나로 대신해서 구조만 테스트할 수 있게 함."""
    folder = folder or FLOOR_FOLDER
    if not os.path.isdir(folder):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print(f"  (Drive 연결 안 됨: {type(e).__name__})")
    entries = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            if ext.lower() in FLOOR_EXTS:
                path = os.path.join(folder, name)
                try:
                    entries[stem] = {"path": path, "raw": load_texture(path), "vec": None}
                except Exception as e:
                    print(f"  바닥재 '{name}'을 읽지 못해 건너뜀: {type(e).__name__}")
    if not entries:
        print(f"⚠️ {label} 폴더가 비어 있거나 없음: {folder}")
        if not fallback:
            return {}
        print("   → 가짜 대리석(sample_marble) 하나로 테스트합니다. 진짜 사진을 넣으면 자동으로 사용돼요.")
        _floor_cache["sample_marble"] = sample_marble()
        entries["sample_marble"] = {"path": None, "raw": _floor_cache["sample_marble"], "vec": None}
    library = _finish_library(entries)
    print(f"{label} {len(library)}개: " + ", ".join(library))
    return library

def get_floor_texture(floor_id, library, prefix=""):
    """id로 사진(배열)을 가져와요 (처음 한 번만 읽음). prefix: 바닥재와 벽지의 이름이 겹쳐도 섞이지 않게."""
    key = prefix + str(floor_id)
    if key not in _floor_cache:
        tex = _read_floor_raw(library[floor_id])
        if FLOOR_MIRROR_TILE:
            top = np.concatenate([tex, tex[:, ::-1]], axis=1)
            tex = np.concatenate([top, top[::-1]], axis=0)
        _floor_cache[key] = tex
    return _floor_cache[key]

FLOOR_LIBRARY = load_floor_library()
# ---- 벽지 목록 (바닥재와 같은 방식: 사진 → 평균 색 + CLIP 벡터) ----
WALL_LIBRARY = load_floor_library(WALL_FOLDER, label="벽지", fallback=False)

def use_wall_library():
    """이번 실행에서 벽지 사진을 쓸지(True) 색상 코드를 쓸지(False)."""
    if WALL_USE_LIBRARY == "auto":
        return bool(WALL_LIBRARY)
    return bool(WALL_USE_LIBRARY)

def get_wall_texture(wall_id):
    """벽지 id로 사진(배열)을 가져와요."""
    return get_floor_texture(wall_id, WALL_LIBRARY, prefix="wall:")

# 6단계: 레퍼런스 분석 함수 (OpenAI)
import os, json, re, time, base64

HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

def get_key(name, tries=3):
    """Colab 보안 비밀 → 환경변수 순서로 키를 찾아요. 없으면 None (그 API는 건너뜀).
    이 서버에서는 Colab이 없으니 **환경변수**(RunPod의 Environment Variables)만 봐요."""
    tries = 0                      # 이 서버는 Colab이 아니라 환경변수(RunPod Environment Variables)만 봐요
    for attempt in range(tries):
        try:
            from google.colab import userdata
            key = userdata.get(name)
            if isinstance(key, str) and key.strip():
                return key.strip()
            break                                   # 읽기는 됐는데 값이 비어 있음 → 다시 해도 소용없음
        except Exception as e:
            transient = "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower()
            print(f"  ({name}) Colab 보안 비밀을 못 불러옴 ({attempt + 1}/{tries}): {type(e).__name__}")
            if not transient:
                print("   → 🔑에 이름이 정확한지, '노트북 액세스'가 켜져 있는지 확인")
                break
            time.sleep(2)
    key = os.environ.get(name)
    if isinstance(key, str) and key.strip():
        return key.strip()
    print(f"  ({name}) 키를 찾지 못해 이 API는 건너뜀 (RunPod의 Environment Variables에 넣어 주세요)")
    return None

_client = None
def get_client():
    """OpenAI 클라이언트를 한 번만 만들어 재사용. 키가 없으면 None. (키를 못 찾은 경우는 기억하지 않아서 다시 부르면 또 찾아봐요)"""
    global _client
    if _client is None:
        key = get_key("OPENAI_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        _client = OpenAI(api_key=key)
    return _client


ANALYSIS_PROMPT = """너는 실내 인테리어 컬러·소재 코디네이터야.
첨부한 레퍼런스 사진들(1장일 수도 있어)은 한 사용자가 좋아하는 방이야. 이 사진들을 분석해서
(1) 사용자의 인테리어 취향 키워드를 뽑고,
(2) 사용자 방에 적용할 시안을 __N__가지 제안해줘. 시안 하나는 '벽 색 1개 + 바닥재 느낌 1개 + 몰딩 색 1개'야.

규칙:
- 키워드는 사진들의 공통된 분위기·색감·소재·질감에서 뽑아. keywords_ko는 한국어 5~8개, keywords_en은 같은 뜻의 영어 5~8개(소문자).
- 시안 __N__가지는 모두 레퍼런스의 취향과 어울려야 하지만 서로 확실히 달라야 해. 같은 분위기 안에서 밝기와 깊이를 다르게 하고(밝은 / 중간 / 어두운 톤 등), 바닥재 재질이나 톤도 서로 다르게 해줘.
- 벽 색과 바닥재는 서로 자연스럽게 어울려야 해. 벽 색은 가구가 아니라 벽의 색이야.
- wall_keywords_en: 벽(벽지)의 느낌도 영어 단어 2~4개로 말해줘. 재질, 무늬, 톤 순서로 (예: ["plain", "warm beige", "matte"], ["linen texture", "soft grey"]). 무늬가 없는 깔끔한 벽이면 "plain"을 넣어줘.
- 바닥재는 레퍼런스 사진 속 바닥의 재질·톤과 비슷하게 하되, 문장이 아니라 영어 단어 2~4개로 말해줘. 재질, 톤, 질감 순서로 (예: ["light oak", "wood plank", "matte"], ["white marble", "polished"]). floor_hex는 그 바닥재의 대표 색.
- 밝은 톤이라도 거의 흰색(아주 밝은 #F2 이상)은 피하고, 연한 베이지·그레이처럼 색이 살짝 느껴지는 색을 써. 벽이 하얗게 날아가 보이면 안 돼.
- hex는 반드시 #RRGGBB 형식. color_name은 한국어로 짧게. name은 시안 이름(한국어, 짧게). tone은 light | mid | dark 중 하나.
- reason과 why는 **사용자에게 이 마감재를 추천하는 이유**야. 기술 설명이 아니야.
  ❌ 절대 쓰지 마: "자연스럽게 보이도록 ~했다", "연결이 자연스럽게", "경계가 ~", "톤을 맞춰 합성했다"
     처럼 **합성·이미지 처리 이야기**. 사용자는 그걸 궁금해하지 않아.
  ⭕ 대신: "이 색이 당신이 좋아하는 분위기와 왜 맞는지", "이 조합이 방에서 어떤 느낌을 줄지"를 말해.
- **마감재로 바꿀 수 있는 것만 근거로 써.** 벽 색, 바닥 재질·톤, 몰딩 색 같은 것.
  ❌ 자연광·창문 위치·방 구조·가구 배치처럼 **인테리어 마감으로 못 바꾸는 것**을 근거로 들지 마
     ("창으로 빛이 들어와서 따뜻해요" 같은 말은 쓰지 마). 레퍼런스를 볼 때도 **색과 재질** 위주로 읽어.
- reason과 why의 **말투**: 앱 화면에서 사용자에게 직접 말하듯 **부드러운 존댓말(해요체)**로 써.
  "~하시는 것 같아요", "~해 보여요", "~하면 좋을 것 같아요"처럼. 사용자의 취향을 **읽어주듯** 말해줘.
  예: "따뜻한 계열을 좋아하시는 것 같아요. 고르신 사진마다 오크 바닥에 베이지 패브릭이 반복되거든요."
  설명 대상은 **사용자**야. 개발자나 동료에게 보고하는 글이 아니야.
  ❌ 쓰지 마: "~이다", "~했다", "~합니다"체 / "사용자는 ~을 선호한다" 같은 보고서 문장 / "당신"이라는 호칭.
- **글은 세 덩어리로 나눠서 줘** (앱에서 띄어서 보여줄 거야):
  ① reason  = 레퍼런스 분석 요약. "사진에서 ~가 반복적으로 나타났어요" 같은 한두 문장.
  ② picks   = 시안마다, 마감재별로 **한 줄씩**. part는 "벽지" / "바닥재" / "몰딩" 세 가지로 쓰고,
               name은 그 마감재 이름(한국어), why는 "~를 선택해 레퍼런스의 ~를 반영했어요" 형태 한 줄.
  ③ why     = 시안마다 마무리 한 문장. 세 마감재를 합쳤을 때 방이 어떤 느낌이 될지.
  예시 ①: "사진에서 베이지·크림 계열의 색상과 자연스러운 우드 질감이 반복적으로 나타났어요."
  예시 ②: [{"part": "벽지", "name": "밝은 웜 아이보리", "why": "레퍼런스의 부드러운 색감을 반영했어요"},
            {"part": "바닥재", "name": "내추럴 오크", "why": "자연스러운 우드 질감을 반영했어요"},
            {"part": "몰딩", "name": "소프트 화이트", "why": "벽과 톤을 맞춰 깔끔하게 정돈했어요"}]
  예시 ③: "밝은 벽지와 따뜻한 우드 바닥을 함께 써서 편안하고 정돈된 방이 돼요."
- reason은 한국어 1~3문장이고, **반드시 이 구조**로 써:
  레퍼런스 사진에서 **실제로 보이는 색과 재질**을 구체적으로 — 벽 색, 바닥의 재질·톤,
  반복해서 나오는 포인트 색, 가구·패브릭의 색감 같은 것.
  그리고 picks의 why에서는 **reason에 쓴 그 단어를 그대로 다시 써서** 이어줘
  ("레퍼런스의 **베이지·크림 색감**을 반영했어요"처럼). 관찰 → 선택이 한 줄로 이어져야 해.
  ❌ 나쁜 예: "뉴트럴 그레이와 블랙 포인트가 공통적이다. 벽은 부드러운 웜 그레이로 정돈감을 줬다."
     (앞의 '뉴트럴 그레이·블랙 포인트'와 뒤의 '웜 그레이·정돈감'이 서로 이어지지 않음)
  ⭕ 좋은 예: "차분한 무채색을 좋아하시는 것 같아요 — 고르신 사진마다 뉴트럴 그레이 벽에 블랙 포인트,
     라이트 우드가 반복돼요. 그래서 벽은 그 그레이를 살짝 따뜻하게 푼 웜 그레이로 골랐어요. 회색만 쓰면
     차가워 보이기 쉬운데 이 정도면 차분함은 그대로면서 방이 삭막해 보이지 않아요. 바닥은 사진 속
     라이트 우드와 같은 톤이라 벽과 부딪히지 않고, 걸레받이만 한 톤 진하게 가서 바닥 색이 또렷하게 보여요."
  '정돈감', '세련됨', '분위기를 완성한다' 같은 **추상어는 그 자체로 쓰지 마.** 쓰려면 앞에서 관찰한 것과 묶어서 써.
  효과를 말할 때도 **근거를 붙여.** "따뜻해 보이게 했다"처럼 어디서 나온 판단인지 모를 말은 쓰지 마 —
  "사진의 웜 베이지 패브릭과 같은 계열이라 따뜻하게 보여요"처럼 **무엇 때문에 그런지**를 같이 써.
  그리고 네가 위에 적은 keywords_ko·style·warmth 값과 **모순되면 안 돼.**
  warmth가 neutral·cool인데 이유에 "따뜻하게"라고 쓰면 틀린 글이야. 그 시안이 정말 따뜻한 쪽이면 warmth를 warm으로 적어.
  벽·바닥·몰딩 **셋 다** 왜 그 색인지가 보이게 해줘.
  사용자가 **직접 고른 값(선택값)도 함께 주어졌다면**, 사진에서 본 것과 선택값을 **어떻게 함께 반영했는지** 한 문장으로 밝혀줘.
  선택값과 사진이 어긋나면(예: 선택은 '웜톤'인데 사진은 차가운 그레이) 그 점을 솔직히 말하고 어느 쪽을 얼마나 따랐는지 설명해.
  사진이 없고 선택값만 있으면, 선택값의 단어를 인용해서 쓰고 사진 이야기는 하지 마.
- 시안마다 why를 한국어 **한 문장**으로 넣어줘. 여기서도 **레퍼런스에서 본 것을 한 가지 집어서** 이 시안과 이어줘
  (예: "사진 속 블랙 프레임이 살아나도록 벽을 한 톤 밝게 잡았어요"). why도 같은 해요체로.

- keyword_weights: keywords_ko의 **모든 단어**마다 이 사용자 취향을 얼마나 뚜렷하게 나타내는지 0~1 숫자를 붙여줘.
  레퍼런스에서 여러 번, 크게 보일수록 높게. 가장 뚜렷한 단어는 1.0 근처, 살짝 보이는 정도면 0.3 근처.
  word는 keywords_ko에 쓴 단어를 **글자 그대로** 써.
- 시안마다 keywords에 그 시안이 반영한 keywords_ko 단어를 2~4개 골라 넣어줘 (역시 글자 그대로).
- style_ko는 style을 한국어로 짧게 (예: "내추럴", "모던").

아래 JSON 형식으로만 답해:
{
  "keywords_ko": ["", ""],
  "keyword_weights": [{"word": "", "weight": 0.0}],
  "keywords_en": ["", ""],
  "style": "스타일 이름(영어)",
  "style_ko": "스타일 이름(한국어)",
  "warmth": "warm | cool | neutral",
  "softness": "soft | hard | neutral",
  "proposals": [
    {"name": "", "tone": "light | mid | dark", "why": "", "keywords": ["", ""],
     "picks": [{"part": "벽지", "name": "", "why": ""},
               {"part": "바닥재", "name": "", "why": ""},
               {"part": "몰딩", "name": "", "why": ""}],
     "wall": {"color_name": "", "hex": ""},
     "wall_keywords_en": ["", ""],
     "floor_keywords_en": ["", ""],
     "floor_hex": "",
     "molding_top": {"color_name": "", "hex": ""},
     "molding_bottom": {"color_name": "", "hex": ""}}
  ],
  "reason": ""
}"""

def build_prompt():
    return ANALYSIS_PROMPT.replace("__N__", str(N_PROPOSALS))

def validate_analysis(data):
    """AI 응답이 약속한 형식인지 검사하고 값을 정리해요. 이상하면 ValueError → 다시 시도."""
    if not isinstance(data, dict):
        raise ValueError("JSON 객체(dict)가 아님")
    for key in ["keywords_ko", "keywords_en"]:
        items = data.get(key)
        if not isinstance(items, list) or len([k for k in items if isinstance(k, str) and k.strip()]) < 3:
            raise ValueError(f"'{key}'는 문자열 3개 이상의 리스트여야 함")
        data[key] = [k.strip() for k in items if isinstance(k, str) and k.strip()]
    data["keywords_en"] = [k.lower() for k in data["keywords_en"]]
    proposals = data.get("proposals")
    if not isinstance(proposals, list) or len(proposals) < 1:
        raise ValueError("'proposals'는 시안 1개 이상의 리스트여야 함")
    clean = []
    for i, p in enumerate(proposals[:N_PROPOSALS]):
        wall = p.get("wall") if isinstance(p, dict) else None
        if not isinstance(wall, dict) or not wall.get("color_name") or not HEX_PATTERN.match(str(wall.get("hex", ""))):
            raise ValueError(f"proposals[{i}].wall에 color_name과 #RRGGBB 형식의 hex가 필요함")
        words = p.get("floor_keywords_en")
        if isinstance(words, str):
            words = [w for w in words.split(",")]
        if not isinstance(words, list):
            raise ValueError(f"proposals[{i}].floor_keywords_en은 영어 단어 리스트여야 함")
        words = [w.strip().lower() for w in words if isinstance(w, str) and w.strip()][:6]
        if not words:
            raise ValueError(f"proposals[{i}].floor_keywords_en이 비어 있음")
        if not HEX_PATTERN.match(str(p.get("floor_hex", ""))):
            raise ValueError(f"proposals[{i}].floor_hex가 #RRGGBB 형식이 아님")
        capped_hex, capped = cap_luma(str(wall["hex"]))      # 너무 밝은 벽 색은 상한까지 낮춤 (WALL_MAX_LUMA)
        if capped:
            wall["hex_requested"], wall["hex"] = wall["hex"], capped_hex
        wall_words = p.get("wall_keywords_en")                    # 벽 느낌은 없어도 넘어감 (색상 이름으로 대신)
        if isinstance(wall_words, str):
            wall_words = wall_words.split(",")
        wall_words = [w.strip().lower() for w in wall_words if isinstance(w, str) and w.strip()][:6] if isinstance(wall_words, list) else []
        def _molding(key, fallback_hex, fallback_name):
            """몰딩 색 정리 — AI가 안 줬거나 형식이 틀리면 알아서 만들어요 (auto=True로 표시)."""
            v = p.get(key) if isinstance(p.get(key), dict) else {}
            if HEX_PATTERN.match(str(v.get("hex", ""))):
                return {"color_name": str(v.get("color_name") or "몰딩"), "hex": str(v["hex"]).upper(), "auto": False}
            return {"color_name": fallback_name, "hex": fallback_hex, "auto": True}
        light_wall = lighten_hex(wall["hex"], MOLDING_SUGGEST_LIGHTEN)
        molding_top = _molding("molding_top", light_wall, "벽보다 밝은 톤")       # 천장 몰딩: 벽과 어울리게
        floor_hex = p["floor_hex"] if HEX_PATTERN.match(str(p.get("floor_hex", ""))) else light_wall
        molding_bottom = _molding("molding_bottom", floor_hex, "바닥과 맞춘 톤")   # 걸레받이: 바닥과 어울리게
        if not isinstance(p.get("molding_top"), dict) and isinstance(p.get("molding"), dict):
            molding_top = _molding("molding", light_wall, "벽보다 밝은 톤")        # 옛 모양(molding 하나)도 받아줌
        tone = str(p.get("tone", "")).strip().lower()
        why = str(p.get("why") or "").strip()[:200]           # 시안 카드에 쓸 마무리 한 문장 (없어도 됨)
        picks = []                                            # 마감재별 한 줄 설명 (벽지 / 바닥재 / 몰딩)
        for item in (p.get("picks") if isinstance(p.get("picks"), list) else []):
            if not isinstance(item, dict):
                continue
            part, text = str(item.get("part") or "").strip(), str(item.get("why") or "").strip()[:200]
            if part and text:
                picks.append({"part": part, "name": str(item.get("name") or "").strip()[:60], "why": text})
        used = p.get("keywords")                              # 이 시안이 반영한 키워드 (그래프용, 없어도 됨)
        used = [w.strip() for w in used if isinstance(w, str) and w.strip()] if isinstance(used, list) else []
        clean.append({"name": str(p.get("name") or f"시안 {i + 1}"), "why": why, "picks": picks, "keywords": used,
                      "tone": tone if tone in ["light", "mid", "dark"] else "mid",
                      "wall": wall, "wall_keywords_en": wall_words, "floor_keywords_en": words, "floor_hex": p["floor_hex"],
                      "molding_top": molding_top, "molding_bottom": molding_bottom,
                      "molding": molding_top})      # molding은 옛 이름 (천장 몰딩과 같음)
    data["proposals"] = clean
    for key, allowed in [("warmth", ["warm", "cool", "neutral"]), ("softness", ["soft", "hard", "neutral"])]:
        value = str(data.get(key, "")).strip().lower()
        data[key] = value if value in allowed else "neutral"
    data.setdefault("style", "")
    data["style_ko"] = str(data.get("style_ko") or "").strip()[:30]
    if not data.get("reason"):
        raise ValueError("'reason' 항목이 없음")
    tidy_keyword_weights(data)                            # 가중치는 빠져도 재시도하지 않고 알아서 채워요
    return data

def tidy_keyword_weights(data):
    """keywords_ko 순서대로 [{"word", "weight"}]를 만들어요.
    AI가 준 값이 있으면 그 값(0~1로 자름), 없으면 순서로 채워요 (앞일수록 큼: 1.0 → 0.4).
    data["keyword_weights_source"]: "ai"(전부 AI 값) / "mixed"(일부만) / "order"(전부 순서로 채움)"""
    words = data["keywords_ko"]
    given = {}
    for item in (data.get("keyword_weights") if isinstance(data.get("keyword_weights"), list) else []):
        if isinstance(item, dict) and isinstance(item.get("word"), str):
            try:
                given[item["word"].strip()] = min(1.0, max(0.0, float(item.get("weight"))))
            except (TypeError, ValueError):
                pass
    out, from_ai = [], 0
    for i, w in enumerate(words):
        if w in given:
            weight, from_ai = given[w], from_ai + 1
        else:
            weight = 1.0 - 0.6 * i / max(1, len(words) - 1)
        out.append({"word": w, "weight": round(weight, 2)})
    data["keyword_weights"] = out
    data["keyword_weights_source"] = "ai" if from_ai == len(words) else ("order" if from_ai == 0 else "mixed")
    weight_of = {x["word"]: x["weight"] for x in out}
    top = [x["word"] for x in sorted(out, key=lambda x: -x["weight"])[:3]]
    for p in data["proposals"]:                          # 시안이 반영한 키워드: 목록에 있는 단어만, 없으면 상위 3개
        used = [w for w in p.get("keywords", []) if w in weight_of]
        p["keywords"] = list(dict.fromkeys(used)) or top
    return data

def build_graph(data):
    """취향 키워드 → 스타일 → 시안을 잇는 그래프 (프론트 약속 형식).
    nodes: {id, type(keyword|style|proposal), label, weight(0~1)}
    edges: {source, target, weight(0~1)}
    - 키워드 weight = AI가 매긴 값 / 시안 weight = 그 시안이 반영한 키워드 weight의 평균
    - 선: 키워드→스타일(키워드 weight), 스타일→시안(시안 weight), 키워드→시안(키워드 weight)"""
    kw = data.get("keyword_weights") or []
    kid = {x["word"]: f"kw_{i + 1}" for i, x in enumerate(kw)}
    nodes = [{"id": kid[x["word"]], "type": "keyword", "label": x["word"], "weight": x["weight"]} for x in kw]
    edges = []
    style_label = data.get("style_ko") or data.get("style") or ""
    if style_label:
        nodes.append({"id": "st_1", "type": "style", "label": style_label, "weight": 1.0})
        edges += [{"source": kid[x["word"]], "target": "st_1", "weight": x["weight"]} for x in kw]
    weight_of = {x["word"]: x["weight"] for x in kw}
    for i, p in enumerate(data.get("proposals") or []):
        pid = f"pr_{i + 1}"
        used = [w for w in p.get("keywords", []) if w in kid]
        pw = round(sum(weight_of[w] for w in used) / len(used), 2) if used else 0.0
        nodes.append({"id": pid, "type": "proposal", "label": p.get("name", f"시안 {i + 1}"), "weight": pw, "index": i})
        if style_label:
            edges.append({"source": "st_1", "target": pid, "weight": pw})
        edges += [{"source": kid[w], "target": pid, "weight": weight_of[w]} for w in used]
    return {"nodes": nodes, "edges": edges, "weight_source": data.get("keyword_weights_source", "order")}

def load_image_bytes(path, max_side=1024):
    """AI에 보내려고 JPEG 바이트로 변환 (확장자가 뭐든 같은 형식으로 통일)."""
    image = load_image_rgb(path, max_side)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()

def preference_text(prefs):
    """사용자가 고른 값({"tone": "웜톤", "texture": "우드"})을 프롬프트에 붙일 문장으로 바꿔요."""
    lines = [f"- {k}: {v}" for k, v in (prefs or {}).items() if str(v).strip()]
    if not lines:
        raise RuntimeError("선택값이 비어 있어요 (예: tone=웜톤, texture=우드)")
    return ("레퍼런스 사진 대신, 사용자가 화면에서 **직접 고른 값**이야. 이 취향에 맞춰 분석하고 시안을 제안해줘.\n"
            + "\n".join(lines)
            + "\n사진이 없으니 keywords_ko·keywords_en은 이 선택값에서 자연스럽게 넓혀서 뽑아줘.")

def _call_openai(reference_paths, preferences=None):
    content = [{"type": "text", "text": build_prompt()}]
    if preferences:
        content.append({"type": "text", "text": preference_text(preferences)})
    for path in reference_paths:
        b64 = base64.b64encode(load_image_bytes(path, SEND_MAX_SIDE)).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    response = get_client().chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"})        # temperature는 모델에 따라 못 쓰는 경우가 있어 지정하지 않음
    return response.choices[0].message.content


# 재시도해도 해결되지 않는 오류 (단어들, 안내 문구). 위에 있는 것부터 확인
FATAL_ERRORS = [
    (["insufficient_quota", "no credits", "billing", "exceeded your current quota"], "API 크레딧/결제 한도 문제. OpenAI 결제 페이지에서 크레딧을 충전해야 함 (ChatGPT Plus 같은 구독과는 별개)"),
    (["invalid_api_key", "incorrect api key", "401"], "API 키가 올바르지 않음. 🔑에 저장한 키를 다시 확인"),
    (["permission", "403"], "이 키로는 이 모델을 쓸 권한이 없음"),
    (["model_not_found", "404", "not found"], "모델 이름이 없거나 쓸 수 없음. 2단계의 MODEL 확인"),
]

def analyze_preferences(preferences, max_retry=3):
    """[AI 분석] 사진 없이 **선택값만**으로 키워드 + 시안을 만들어요 (앱의 '직접 선택' 화면).
    preferences: {"tone": "웜톤", "texture": "우드", ...} 처럼 자유롭게 (키 이름은 앱 마음대로)"""
    return analyze_reference([], max_retry=max_retry, preferences=preferences)

def analyze_reference(reference_paths, max_retry=3, preferences=None):
    """[AI 분석] 레퍼런스 사진들 → 키워드 + 시안(벽 색 + 바닥재 이름).
    - 형식이 틀리거나 없는 바닥재를 골랐으면 다시 시도, 서버 혼잡(429/500/503 등)이면 점점 더 오래 기다렸다가 다시 시도
    - 끝까지 실패하면 RuntimeError
    반환: (분석 결과 dict, 걸린 시간(초))"""
    if not reference_paths and not preferences:
        raise RuntimeError("레퍼런스 사진이나 선택값(preferences) 중 하나는 있어야 해요")
    reference_paths = list(reference_paths)[:MAX_REFERENCES]
    if get_client() is None:
        raise RuntimeError("OpenAI API 키가 없어요. Colab 🔑에 OPENAI_API_KEY를 저장하고 '노트북 액세스'를 켜주세요")
    started, last_error = time.time(), None
    for attempt in range(max_retry + 1):
        try:
            text = _call_openai(reference_paths, preferences)
            data = validate_analysis(json.loads(text))
            return data, time.time() - started
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            for words, hint in FATAL_ERRORS:      # 다시 시도해도 소용없는 오류는 바로 멈추고 이유를 알려줘요
                if any(w in msg for w in words):
                    raise RuntimeError(f"OpenAI 사용 불가 - {hint}")
            print(f"  (재시도 {attempt + 1}/{max_retry + 1}) {type(e).__name__}: {str(e)[:150]}")
            busy = any(w in msg for w in ["429", "500", "502", "503", "unavailable", "rate", "overloaded", "timeout"])
            if busy and attempt < max_retry:
                time.sleep(min(2 ** (attempt + 1), 20))
    raise RuntimeError(f"OpenAI 응답 실패: {last_error}")

# 7단계: 실행 함수 (방 준비 1번 + 분석 1번 → CLIP으로 바닥재 고르기 → 시안별 합성)
#         prepare_room: 분할 1번(작은 사진) + 마스크 정리·몰딩 찾기(큰 사진) / reclean: 분할은 그대로 두고 정리만 다시

def prepare_room(room_path, seg_max_side=None, out_max_side=None):
    """방 사진을 읽고 영역을 한 번만 나눠 둬요.
    분할은 작은 사진(SEG_MAX_SIDE)에서 하고, 합성은 큰 사진(OUTPUT_MAX_SIDE)에서 해요.
    → 분할 속도는 그대로면서 결과는 선명하게. 마스크는 큰 사진 크기로 키운 뒤 경계를 다시 맞춰요."""
    started = time.time()
    full = load_image_rgb(room_path, OUTPUT_MAX_SIDE if out_max_side is None else out_max_side)
    small = load_image_rgb(full, SEG_MAX_SIDE if seg_max_side is None else seg_max_side)
    raw = segment_room(small)
    seconds = time.time() - started
    room = {"image": np.array(full), "raw": raw, "raw_size": small.size}
    reclean(room)
    m = room["masks"]
    print(f"[영역 분할] {seconds:.1f}초 | 분할 {small.width}x{small.height} → 합성 {full.width}x{full.height}"
          f" | 벽 {m['wall'].mean():.0%}, 바닥 {m['floor'].mean():.0%}, 천장 {m['ceiling'].mean():.0%}")
    for name, d in (room.get("molding_info") or {}).items():
        print(f"  {name}: " + (f"두께 {d['thickness']}px, 가로의 {d['cols_ratio']:.0%}에서 보임" if d.get("found") else "못 찾음"))
    print("  벽 모서리:", f"x={room['wall_split']} (왼쪽·정면·오른쪽으로 나눠서 옆벽에 원근을 줌)" if room.get("wall_split")
          else "못 찾음 → 전체를 정면 벽으로 곧게 깖 (WALL_SPLIT에 직접 적을 수 있어요)")
    print("  벽 면:", ", ".join(room["wall_faces"]), "(왼쪽부터)")
    wall_report(room)                                   # 면마다 무늬 점수 → 덮어씌우기/필터 중 무엇을 쓸지
    return room

def straighten_labels(sm, classes, eps):
    """영역 윤곽을 직선 꼭짓점으로 단순화해서 다시 그려요 (딱딱한 직선 느낌).
    틈이 안 생기게 영역을 1픽셀씩 겹쳐 그리고, 겹치면 위 우선순위가 이겨요. 몰딩은 가늘어지지 않게 오차를 작게 잡아요."""
    H, W = sm.shape
    out = np.zeros((H, W), np.uint8)
    order = [k for k, n in classes.items() if n == "wall"] + [k for k, n in classes.items() if n in ("ceiling", "floor")] \
        + [k for k, n in classes.items() if n == "keep"] + [k for k, n in classes.items() if n not in ("wall", "ceiling", "floor", "keep")]
    for k in order:
        is_molding = classes[k] not in ("wall", "ceiling", "floor", "keep")
        m = (sm == k).astype(np.uint8)
        if not m.any():
            continue
        e = float(eps)
        if is_molding:                                          # 두께의 1/10 이하로 제한 → 가는 띠가 안 사라져요
            thick = 2.0 * float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max())
            e = min(e, max(1.0, thick * 0.1))
        cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        tmp = np.zeros((H, W), np.uint8)
        if hier is not None:
            for i, c in enumerate(cs):
                if cv2.contourArea(c) < 12:
                    continue
                a = cv2.approxPolyDP(c, e, True)
                cv2.fillPoly(tmp, [a], 1 if hier[0][i][3] == -1 else 0)   # 바깥은 칠하고 구멍은 비움
        if not is_molding:
            tmp = cv2.dilate(tmp, np.ones((3, 3), np.uint8), iterations=1)
        out[tmp == 1] = k
    return out

def smooth_partition(masks, moldings, keep, ratio):
    """영역 경계의 잔톱니·계단을 없애요. 영역마다 '투표'를 번지게 한 뒤, 픽셀마다 제일 센 쪽으로 다시 정해요.
    그래서 영역끼리 빈틈이나 겹침 없이 경계만 둥글어져요 (몰딩·창문 자리 포함).
    규칙은 붓으로 고친 뒤(_apply_label)와 같아요: 벽 마스크는 몰딩 자리까지 포함하고, 몰딩은 따로 돌려줘요.
    ratio: 흐림 정도 (이미지 긴 변 대비). → (masks, moldings, keep)"""
    H, W = masks["wall"].shape
    sigma = max(1.0, float(ratio) * max(H, W))
    label = np.zeros((H, W), np.uint8)
    classes = {}                                           # 클래스 번호 → 이름
    for name in ("ceiling", "wall", "floor"):              # 겹치면 나중 것이 이김 (_label_from_room과 같은 순서)
        if name in masks and masks[name].any():
            classes[len(classes) + 1] = name
            label[masks[name]] = len(classes)
    if keep is not None and keep.any():
        classes[len(classes) + 1] = "keep"
        label[keep] = len(classes)
    for key, m in (moldings or {}).items():                # 몰딩이 제일 위
        if m is not None and m.any():
            classes[len(classes) + 1] = key
            label[m] = len(classes)
    K = len(classes) + 1
    votes = np.empty((K, H, W), np.float32)
    for k in range(K):
        votes[k] = cv2.GaussianBlur((label == k).astype(np.float32), (0, 0), sigma)
    votes[0] -= 1e-4                                       # 정확히 비길 땐 '없음'보다 영역 쪽으로
    sm = votes.argmax(axis=0)
    if EDGE_STYLE == "straight":                           # 잔톱니를 없앤 뒤 직선 꼭짓점으로 단순화
        sm = straighten_labels(sm.astype(np.uint8), classes, max(1.0, EDGE_STRAIGHT * max(H, W)))
    by_name = {name: (sm == k) for k, name in classes.items()}
    new_moldings = {key: by_name[key] for key in (moldings or {}) if key in by_name and by_name[key].any()}
    wall = by_name.get("wall", np.zeros((H, W), bool))
    for key in new_moldings:
        wall = wall | new_moldings[key]                    # 벽 마스크는 몰딩 자리까지 (칠할 때 몰딩만 따로 뺌)
    new_masks = dict(masks)
    new_masks["wall"] = wall
    new_masks["floor"] = by_name.get("floor", np.zeros((H, W), bool))
    new_masks["ceiling"] = by_name.get("ceiling", np.zeros((H, W), bool))
    new_keep = by_name.get("keep", keep)                 # 창문·문 자리가 없으면 원래 값 그대로
    return new_masks, new_moldings, new_keep

def reclean(room, smooth=None, snap=None, band=None, min_area_ratio=None, find_molding=None, edge_smooth=None):
    """분할은 다시 하지 않고 마스크 정리만 다시 해요 (설정을 바꿔가며 바로 확인).
    작은 사진에서 정리 → 큰 사진 크기로 키우기 → 큰 사진에서 경계 맞추기 → 몰딩 찾기 순서."""
    image = room["image"]
    H, W = image.shape[:2]
    keep = room["raw"].get("keep")                       # 창문·문처럼 그대로 둘 곳 (벽에 섞이면 안 돼요)
    masks = {n: clean_mask(m, min_area_ratio, smooth) for n, m in room["raw"].items() if n != "keep"}
    if (masks["wall"].shape[0], masks["wall"].shape[1]) != (H, W):
        masks = {n: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for n, m in masks.items()}
    snap = EDGE_SNAP if snap is None else snap
    if snap:
        masks = snap_edges(masks, image, band)          # 큰 사진에서 맞춰야 가장자리가 또렷해요
    if keep is not None and keep.any():                  # 빈틈 메우다 삼켜진 창문·문을 다시 빼냄
        if keep.shape != (H, W):
            keep = cv2.resize(keep.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        masks = {n: (m & ~keep) for n, m in masks.items()}
    room["keep"] = keep
    room["masks"] = masks
    room["wall_split"] = tuple(WALL_SPLIT) if WALL_SPLIT else estimate_wall_split(masks)   # 벽 모서리 (옆벽 원근용)
    find_molding = MOLDING_FIND if find_molding is None else find_molding
    if find_molding:
        room["moldings"], room["molding_info"] = find_moldings_auto(masks, image)
    else:
        room["moldings"], room["molding_info"] = {}, {}
    edge_smooth = EDGE_SMOOTH if edge_smooth is None else edge_smooth
    if edge_smooth and edge_smooth > 0:                  # 경계의 잔톱니·계단 다듬기 (실패하면 다듬지 않고 그대로 써요)
        try:
            masks, room["moldings"], room["keep"] = smooth_partition(masks, room["moldings"], room["keep"], edge_smooth)
            room["masks"] = masks
            room["wall_split"] = tuple(WALL_SPLIT) if WALL_SPLIT else estimate_wall_split(masks)
        except Exception as e:
            print(f"⚠️ 경계 다듬기를 건너뛰었어요 ({type(e).__name__}: {str(e)[:80]})")
    room["wall_faces"], room["wall_split_found"] = split_wall_faces(masks, room["wall_split"])   # wall1, wall2, ...
    room.pop("laid", None)
    room.pop("_wall_laid", None)
    return room

FLOOR_QUERY = "a top-down photo of {} flooring"
WALL_QUERY = "a photo of {} wallpaper on a wall"

def reference_floor_vector(reference_paths):
    """레퍼런스 사진들에서 바닥 부분만 잘라 CLIP 벡터의 평균을 만들어요. 바닥이 안 보이는 사진은 건너뜀. 하나도 없으면 None."""
    vecs = []
    for path in reference_paths:
        try:
            image = load_image_rgb(path)
            mask = clean_mask(segment_room(image)["floor"])
            if mask.mean() < 0.02:
                continue
            arr = np.array(image).copy()
            arr[~mask] = arr[mask].mean(axis=0)                 # 바닥이 아닌 곳은 평균색으로 덮어서 바닥만 보이게
            ys, xs = np.where(mask)
            vecs.append(embed_images([Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])])[0])
        except Exception as e:
            print(f"  (레퍼런스 바닥 분석 건너뜀: {type(e).__name__}: {str(e)[:60]})")
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)

def pick_from_library(proposals, library, get_words, get_hex, set_result, query, ref_vec=None, top_k=3,
                      color_weight=None, unique=True):
    """시안마다 라이브러리(바닥재·벽지)에서 하나씩 골라요.
    점수 = CLIP 단어 유사도(후보들 사이에서 표준화) + (선택) 레퍼런스 사진과의 유사도 - 색 차이 × 가중치
    unique=True면 시안끼리 같은 것이 안 겹치도록 점수 높은 짝부터 확정 (부족하면 겹침 허용)."""
    color_weight = FLOOR_COLOR_WEIGHT if color_weight is None else color_weight
    ids = list(library)
    if not ids:
        return proposals
    V = np.stack([library[i]["vec"] for i in ids])
    T = embed_texts([query.format(", ".join(get_words(p))) for p in proposals])
    sim = T @ V.T
    z = (sim - sim.mean(axis=1, keepdims=True)) / (sim.std(axis=1, keepdims=True) + 1e-6)
    if ref_vec is not None:
        rs = V @ ref_vec
        z = z + FLOOR_REF_WEIGHT * ((rs - rs.mean()) / (rs.std() + 1e-6))[None, :]
    avg = np.array([hex_to_rgb(library[i]["avg_hex"]) for i in ids], np.float32)
    want = np.array([hex_to_rgb(get_hex(p)) for p in proposals], np.float32)
    dist = np.linalg.norm(want[:, None, :] - avg[None, :, :], axis=2) / 441.7      # 0(같은 색)~1
    score = z - color_weight * dist
    chosen, used = {}, set()
    if unique:
        for flat in np.argsort(-score, axis=None):
            i, j = divmod(int(flat), len(ids))
            if i in chosen or j in used:
                continue
            chosen[i] = j; used.add(j)
            if len(chosen) == len(proposals):
                break
    for i in range(len(proposals)):
        if i not in chosen:
            chosen[i] = int(score[i].argmax())
    for i, p in enumerate(proposals):
        set_result(p, ids[chosen[i]], [ids[j] for j in np.argsort(-score[i])[:top_k]])
    return proposals

def pick_floors(proposals, ref_vec=None, top_k=3):
    """시안마다 바닥재를 하나씩 골라 p["floor_id"]에 넣어요."""
    return pick_from_library(proposals, FLOOR_LIBRARY, lambda p: p["floor_keywords_en"], lambda p: p["floor_hex"],
                             lambda p, fid, cands: p.update(floor_id=fid, floor_candidates=cands),
                             FLOOR_QUERY, ref_vec, top_k)

def pick_walls(proposals, top_k=3):
    """시안마다 벽지를 하나씩 골라 p["wall_id"]에 넣어요 (벽지 폴더가 비어 있으면 아무것도 안 함)."""
    if not use_wall_library():
        return proposals
    return pick_from_library(proposals, WALL_LIBRARY,
                             lambda p: p.get("wall_keywords_en") or [p["wall"].get("color_name", "plain")],
                             lambda p: p["wall"]["hex"],
                             lambda p, wid, cands: p.update(wall_id=wid, wall_candidates=cands),
                             WALL_QUERY, None, top_k)

def make_design(room, reference_paths, detail=1.0):
    """레퍼런스 → 키워드 + 시안 N개 → CLIP으로 바닥재·벽지 고르기 → 시안마다 합성.
    벽은 벽지 폴더에 사진이 있으면 **벽지 사진을 깔고**, 없으면 색상 코드로 칠해요 (2단계 WALL_USE_LIBRARY).
    반환: {"analysis", "seconds", "images", "floor_ids"}"""
    analysis, seconds = analyze_reference(reference_paths)
    started = time.time()
    ref_vec = reference_floor_vector(list(reference_paths)[:MAX_REFERENCES]) if FLOOR_FROM_REFERENCE else None
    pick_floors(analysis["proposals"], ref_vec)
    pick_walls(analysis["proposals"])
    laid_cache, wall_cache = {}, {}                       # 같은 재료는 한 번만 깔기
    images, floor_ids = [], []
    for p in analysis["proposals"]:
        fid = p["floor_id"]
        if fid not in laid_cache:
            laid_cache[fid] = floor_perspective(room["image"].shape[:2], room["masks"]["floor"], get_floor_texture(fid, FLOOR_LIBRARY),
                                                FLOOR_TILE, HORIZON, FOV, ROTATE)
        wid, wall_material = p.get("wall_id"), p["wall"]["hex"]
        if wid:
            if wid not in wall_cache:
                wall_cache[wid] = lay_wall_texture(room["image"].shape[:2], room["masks"]["wall"], get_wall_texture(wid),
                                                   split=room.get("wall_split"))
            wall_material = None
        p["ceiling_hex"] = ceiling_color(avg_hex(wall_cache[wid]) if wid else p["wall"]["hex"])
        images.append(recolor(room, wall_material, None, detail, floor_laid=laid_cache[fid],
                              wall_laid=wall_cache[wid] if wid else None))
        floor_ids.append(fid)
    print(f"[OpenAI] 분석 {seconds:.1f}초 + 재료 고르기·합성 {len(images)}장 {time.time() - started:.1f}초")
    print("키워드:", ", ".join(analysis["keywords_ko"]))
    for i, p in enumerate(analysis["proposals"], 1):
        capped = f" (원래 {p['wall']['hex_requested']} → 너무 밝아서 낮춤)" if p["wall"].get("hex_requested") else ""
        wall_txt = f"벽지 {p['wall_id']}" if p.get("wall_id") else f"벽 {p['wall']['color_name']} {p['wall']['hex']}{capped}"
        ceil = f" / 천장 {p['ceiling_hex']}" if p.get("ceiling_hex") else ""
        print(f"  {i}. {p['name']} ({p['tone']}) | {wall_txt}{ceil} / 바닥재 {p['floor_id']}"
              f"  ← 바닥 '{', '.join(p['floor_keywords_en'])}'" +
              (f", 벽 '{', '.join(p.get('wall_keywords_en') or [])}' (벽지 후보: {', '.join(p.get('wall_candidates') or [])})" if p.get("wall_id") else ""))
    return {"analysis": analysis, "seconds": seconds, "images": images, "floor_ids": floor_ids}

# ---- 등록된 색 목록(팔레트) 안에서만 색을 고르게 하기 ----
COLOR_PALETTE = {}          # {"색 id": {"hex": "#RRGGBB", "name": "퓨어화이트", "lab": ...}}

def _hex_to_lab(hex_code):
    rgb = np.array(hex_to_rgb(hex_code), np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

def set_color_palette(colors):
    """[백엔드 연결용] 앱 DB에 등록된 색 목록을 넣어요 → 추천 색이 **이 목록 안에서만** 나와요.
    colors: [{"id": "c01", "hex": "#FFFFFF", "name": "퓨어화이트"}, ...]   (id·name은 선택)
    빈 목록을 주면 제한이 풀려서 AI가 만든 색을 그대로 써요."""
    global COLOR_PALETTE
    palette = {}
    for i, c in enumerate(colors or []):
        hx = str(c.get("hex", "")).strip()
        if not HEX_PATTERN.match(hx):
            print(f"  색 '{c.get('id', i)}'의 hex가 #RRGGBB 형식이 아니라 건너뜀"); continue
        cid = str(c.get("id") or hx)
        palette[cid] = {"hex": hx.upper(), "name": c.get("name"), "lab": _hex_to_lab(hx)}
    COLOR_PALETTE = palette
    print(f"색 팔레트 {len(palette)}개 등록" if palette else "색 팔레트 비움 (AI가 만든 색을 그대로 씀)")
    return palette

def snap_to_palette(hex_code):
    """색 하나를 팔레트에서 가장 비슷한 색으로 바꿔요 → (hex, 색 id). 팔레트가 비었으면 그대로 돌려줘요.
    사람 눈에 가까운 Lab 거리로 비교해요 (RGB로 재면 엉뚱한 색이 뽑혀요)."""
    if not COLOR_PALETTE or not HEX_PATTERN.match(str(hex_code)):
        return hex_code, None
    lab = _hex_to_lab(hex_code)
    cid = min(COLOR_PALETTE, key=lambda k: float(np.linalg.norm(COLOR_PALETTE[k]["lab"] - lab)))
    return COLOR_PALETTE[cid]["hex"], cid

MOLDING_TOP_K = 3           # 몰딩 후보 색 개수 (가까운 순)

def palette_top(hex_code, k=MOLDING_TOP_K):
    """팔레트에서 이 색과 가까운 색 id를 k개 (가까운 순). 팔레트가 비었거나 색이 이상하면 []."""
    if not COLOR_PALETTE or not HEX_PATTERN.match(str(hex_code)):
        return []
    lab = _hex_to_lab(hex_code)
    return sorted(COLOR_PALETTE, key=lambda c: float(np.linalg.norm(COLOR_PALETTE[c]["lab"] - lab)))[:k]

def snap_proposals(proposals):
    """시안들의 벽 색·몰딩 색을 등록된 색으로 바꿔요. 원래 AI가 제안한 색은 hex_suggested에 남겨둬요.
    같은 색이 여러 시안에 겹치면 그대로 둬요 (색이 24개뿐이면 겹칠 수 있어요).
    몰딩은 후보도 3개씩: p["molding_top_candidates"], p["molding_bottom_candidates"] (색 id, 가까운 순 — 첫 번째가 고른 색)"""
    if not COLOR_PALETTE:
        for p in proposals:
            p["molding_top_candidates"], p["molding_bottom_candidates"] = [], []
        return proposals
    for p in proposals:
        for key in ("molding_top", "molding_bottom"):           # 바꾸기 전 AI 색 기준으로 후보를 뽑아요
            part = p.get(key)
            p[f"{key}_candidates"] = palette_top(part.get("hex")) if isinstance(part, dict) else []
        for part in (p.get("wall"), p.get("molding_top"), p.get("molding_bottom")):
            if not isinstance(part, dict) or not part.get("hex"):
                continue
            snapped, cid = snap_to_palette(part["hex"])
            if cid and snapped.upper() != str(part["hex"]).upper():
                part["hex_suggested"] = part["hex"]          # AI가 원래 말한 색 (참고용)
            part["hex"], part["color_id"] = snapped, cid
            if cid and COLOR_PALETTE[cid].get("name"):
                part["color_name"] = COLOR_PALETTE[cid]["name"]
    return proposals

# 8단계: AI 서버 코드 (FastAPI) — 백엔드가 부르는 창구
import uuid, base64, threading, collections, contextlib
import requests as _requests
from typing import List, Optional, Dict, Any
from PIL import ImageOps
from fastapi import FastAPI, Header, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

# ---- 서버 설정 ----
AI_SERVER_KEY = get_key("AI_SERVER_KEY")      # 백엔드와 약속한 비밀 값 (요청 헤더 X-AI-Key)
if not AI_SERVER_KEY:
    print("⚠️ AI_SERVER_KEY가 없어 인증 없이 열립니다 (개발용). 백엔드와 정한 비밀 값을 Colab 🔑에 AI_SERVER_KEY로 저장하세요")
SERVER_MAX_SIDE = OUTPUT_MAX_SIDE   # 결과 사진의 긴 변 상한 (2단계 OUTPUT_MAX_SIDE)
PREVIEW_SIDE = 512                  # generate의 preview=true일 때 결과의 긴 변
MAX_ROOMS = 30                      # 서버가 기억하는 방 개수 (넘으면 오래된 것부터 지움)
POLY_EPS = 0.004                    # 다각형 꼭짓점 줄이는 정도 (둘레 대비). 키우면 꼭짓점이 적고 거칠어짐
POLY_MIN_AREA = 0.0005              # 이 비율보다 작은 조각은 다각형으로 안 보냄 (화면에서 못 잡는 점 방지)

# 영역 번호: 벽 면은 11부터 왼쪽 순서대로 (wall1=11, wall2=12, ...)
REGION_IDS = {"allWalls": 1, "floor": 2, "ceiling": 3, "moldingTop": 4, "moldingBottom": 5}
WALL_FACE_BASE = 11
MOLD_KEYS = (("molding_top", 4, "moldingTop"), ("molding_bottom", 5, "moldingBottom"))
REGION_COLORS = {1: (66, 133, 244), 2: (234, 67, 53), 3: (52, 168, 83), 4: (255, 193, 7), 5: (255, 193, 7)}

LOCK = threading.RLock()                 # 모델과 설정값을 한 번에 한 요청씩만 쓰게 함
ROOMS = collections.OrderedDict()        # room_id → 방 정보
EXTRA_TEX = collections.OrderedDict()    # 목록에 없이 url·base64로 바로 온 사진 (최근 20개)

class AIError(Exception):
    def __init__(self, status, code, message, details=None):
        self.status, self.code, self.message, self.details = status, code, message, details

def _err(status, code, message, details=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})

app = FastAPI(title="Imagine AI Server", version="2.0",
              description="방 사진 영역 분할(벽 면 분리 포함) + 벽지·바닥재 합성 + 레퍼런스 취향 분석. "
                          "인증: 헤더 X-AI-Key (/health 제외). 에러: {error:{code,message,details}}")

@app.exception_handler(AIError)
async def _ai_error_handler(request, exc):
    return _err(exc.status, exc.code, exc.message, exc.details)

@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc):
    return _err(422, "invalid_request", "요청 형식이 올바르지 않아요",
                [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"]} for e in exc.errors()])

def check_key(x_ai_key: Optional[str] = Header(None)):
    if AI_SERVER_KEY and x_ai_key != AI_SERVER_KEY:
        raise AIError(401, "unauthorized", "X-AI-Key 헤더가 없거나 올바르지 않아요")

# ---- 도우미 ----
@contextlib.contextmanager
def override_globals(**changes):
    """요청 한 건 동안만 설정값(2단계)을 바꿔요. None이면 그대로 둠. LOCK 안에서 써야 함."""
    g = globals()
    changes = {k: v for k, v in changes.items() if v is not None}
    old = {k: g[k] for k in changes}
    g.update(changes)
    try:
        yield
    finally:
        g.update(old)

def _download(url, what="이미지"):
    try:
        r = _requests.get(url, timeout=20); r.raise_for_status()
        return r.content
    except Exception as e:
        raise AIError(400, "download_failed", f"{what}을 내려받지 못했어요 ({type(e).__name__})", {"url": url})

def _pil_from_bytes(data, what="이미지"):
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")      # 휴대폰 사진 회전 반영
    except Exception:
        raise AIError(400, "invalid_image", f"{what}을 열 수 없어요 (jpg, png 같은 사진 파일이어야 해요)")

def _from_b64(text, what="사진"):
    try:
        raw = text.split(",", 1)[1] if text.startswith("data:") else text
        return base64.b64decode(raw, validate=False)
    except Exception:
        raise AIError(400, "invalid_image", f"{what}의 base64를 읽을 수 없어요")

def _resize_max(img, max_side):
    s = max_side / max(img.size)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS) if s < 1 else img

def _png(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()

def _get_room(room_id):
    with LOCK:
        room = ROOMS.get(room_id)
        if room is None:
            raise AIError(404, "room_not_found", "방 정보를 찾을 수 없어요. 서버가 다시 시작됐다면 POST /rooms로 사진을 다시 올려 주세요",
                          {"room_id": room_id})
        ROOMS.move_to_end(room_id)
        return room

def _face_order(room):
    return sorted((room.get("wall_faces") or {}), key=lambda n: int(n.replace("wall", "") or 0))

def mask_to_polygons(mask, scale=(1.0, 1.0), eps=None, min_area=None):
    """마스크 → [{"points": [[x,y],...], "holes": [[[x,y],...], ...]}]  (프론트의 꼭짓점 편집용)
    바깥 경계와 구멍(벽 안의 창문처럼 제외된 자리)을 같이 돌려줘요. 좌표는 **원본 사진 픽셀**."""
    eps = POLY_EPS if eps is None else eps
    min_area = POLY_MIN_AREA if min_area is None else min_area
    H, W = mask.shape
    sx, sy = scale
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None:
        return []
    def simplify(c):
        pts = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)
        return [[int(round(x * sx)), int(round(y * sy))] for x, y in pts]
    limit = min_area * H * W
    polys, index = [], {}
    for i, c in enumerate(contours):                      # 바깥 경계 (부모가 없는 것)
        if hierarchy[0][i][3] == -1 and cv2.contourArea(c) >= limit and len(c) >= 3:
            index[i] = len(polys)
            polys.append({"points": simplify(c), "holes": []})
    for i, c in enumerate(contours):                      # 구멍 (부모가 있는 것)
        parent = hierarchy[0][i][3]
        if parent in index and cv2.contourArea(c) >= limit and len(c) >= 3:
            polys[index[parent]]["holes"].append(simplify(c))
    return [p for p in polys if len(p["points"]) >= 3]

def _scale_to_original(room):
    H, W = room["masks"]["wall"].shape
    ow, oh = room["orig_size"]
    return (ow / W, oh / H)

def _regions_info(room):
    """영역 목록 (벽 전체 / 벽 면들 / 바닥 / 천장 / 몰딩)"""
    masks, faces, mold = room["masks"], room.get("wall_faces") or {}, room.get("moldings") or {}
    scale = _scale_to_original(room)
    ratio = lambda m: round(float(m.mean()), 4)
    item = lambda rid, name, m: {"region_id": rid, "name": name, "area_ratio": ratio(m),
                                 "polygons": mask_to_polygons(m, scale)}
    items = [item(1, "allWalls", masks["wall"]), item(2, "floor", masks["floor"]), item(3, "ceiling", masks["ceiling"])]
    for i, name in enumerate(_face_order(room)):
        items.append(item(WALL_FACE_BASE + i, name, faces[name]))
    for key, rid, name in MOLD_KEYS:
        if key in mold:
            items.append(item(rid, name, mold[key]))
    return items

def _wall_methods(room):
    """면마다 무늬 점수와 자동으로 고를 방식 → {"wall1": {"pattern_score": 0.31, "method": "overlay"}, ...}
    프론트에서 '이 벽은 무늬가 있어 덮어씌웁니다' 같은 안내에 쓸 수 있어요."""
    out = {}
    for name, m in (room.get("wall_faces") or {}).items():
        score = wall_pattern_score(room["image"], m)
        out[name] = {"pattern_score": round(float(score), 3),
                     "method": "overlay" if score >= WALL_PATTERN_T else "filter"}
    return out

def _room_summary(room_id, room):
    H, W = room["masks"]["wall"].shape
    ow, oh = room["orig_size"]
    split = room.get("wall_split")
    return {"room_id": room_id, "width": ow, "height": oh, "work_width": W, "work_height": H,
            "regions": _regions_info(room),
            "wall_faces": _face_order(room),
            "wall_split": [int(x * ow / W) for x in split] if split else None,     # 원본 사진 좌표
            "wall_split_found": bool(room.get("wall_split_found")),
            "wall_methods": _wall_methods(room),
            "edited": bool(room.get("edited"))}

def _label_from_room(room):
    """마스크들을 번호 지도 하나로 (1 벽, 2 바닥, 3 천장, 4·5 몰딩)"""
    masks, mold = room["masks"], room.get("moldings") or {}
    label = np.zeros(masks["wall"].shape, np.uint8)
    label[masks["ceiling"]] = 3
    label[masks["wall"]] = 1
    label[masks["floor"]] = 2
    for key, rid, _ in MOLD_KEYS:
        if key in mold:
            label[mold[key]] = rid
    return label

def _apply_label(room, label):
    """번호 지도 → 마스크·몰딩·벽 면 (붓으로 고친 뒤)"""
    masks = {"wall": np.isin(label, [1, 4, 5]), "floor": label == 2, "ceiling": label == 3}
    mold = {key: (label == rid) for key, rid, _ in MOLD_KEYS if (label == rid).any()}
    apply_masks(room, masks, moldings=mold)
    room["edited"] = True
    return room

# ---- 연결 확인 / 재료 목록 ----
@app.get("/health")
def health():
    """연결 확인 + **지금 뭐가 모자란지**. 백엔드·프론트가 에러를 만나면 먼저 여기를 보면 돼요."""
    ready = SEG_MODEL[0] in _seg_cache
    problems = []
    if not get_key("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY가 없어 /recommend(분석)가 503으로 실패해요 — Colab 🔑에 넣고 노트북 액세스를 켜세요")
    if list(FLOOR_LIBRARY) == ["sample_marble"]:
        problems.append("바닥재가 비어 있어요 — 백엔드가 POST /floors/sync를 먼저 호출해야 해요")
    if not WALL_LIBRARY:
        problems.append("벽지가 비어 있어요 — 벽을 색으로만 칠해요 (POST /walls/sync로 등록)")
    if not ready:
        problems.append("분할 모델이 아직 안 올라왔어요 — 첫 POST /rooms가 오래 걸릴 수 있어요 (모델 준비 중)")
    if not COLOR_PALETTE:
        problems.append("색 목록이 비어 있어요 — POST /colors/sync 를 하면 추천 색이 DB에 등록된 색으로만 나와요")
    return {"status": "ok", "ready": ready, "rooms": len(ROOMS), "floors": len(FLOOR_LIBRARY), "walls": len(WALL_LIBRARY),
            "colors": len(COLOR_PALETTE),
            "model": MODEL, "openai_key": bool(get_key("OPENAI_API_KEY")),
            "segmenter_loaded": ready, "clip_loaded": "m" in _clip_cache,
            "molding_source": MOLDING_SOURCE,
            "sam3": ("ready" if "model" in _sam_cache else ("unavailable" if "failed" in _sam_cache else "not_loaded")),
            "features": ["wall_faces", "wallpaper", "molding", "ceiling", "auto_wall_method", "polygons", "color_palette", "preferences"],
            "problems": problems}

@app.get("/selfcheck", dependencies=[Depends(check_key)])
def selfcheck():
    """OpenAI가 **진짜로** 되는지 한 번 찔러봐요 (아주 짧은 요청 1번, 비용 거의 0).
    프론트에서 '분석 안 돼요'가 뜨면 이걸 먼저 불러서 누구 문제인지 가르면 돼요."""
    key = get_key("OPENAI_API_KEY")
    if not key:
        return {"openai": "no_key", "message": "Colab 🔑에 OPENAI_API_KEY가 없어요 (노트북 액세스도 켜야 해요)"}
    try:
        from openai import OpenAI
        OpenAI(api_key=key).chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}])
        # max_tokens 같은 옵션은 일부러 안 줘요 — 모델마다 받는 옵션이 달라서, 그것 때문에 '모델 문제'로 잘못 보일 수 있어요
        return {"openai": "ok", "model": MODEL, "message": "분석까지 정상이에요. 그래도 프론트에서 에러가 나면 백엔드-프론트 쪽 문제예요"}
    except Exception as e:
        text = str(e)
        if "insufficient_quota" in text or "billing" in text or "429" in text:
            kind, msg = "no_credit", "OpenAI 크레딧이 없어요 — 결제 페이지에서 충전해야 분석이 돼요 (ChatGPT 구독과 별개)"
        elif "401" in text or "invalid_api_key" in text:
            kind, msg = "bad_key", "OPENAI_API_KEY 값이 올바르지 않아요"
        elif "unsupported parameter" in text.lower() or "max_tokens" in text:
            kind, msg = "bad_request", f"요청 옵션이 '{MODEL}' 모델과 안 맞아요 (키·크레딧 문제는 아니에요): {text[:120]}"
        elif "does not exist" in text or "model_not_found" in text:
            kind, msg = "bad_model", f"이 계정에서 '{MODEL}' 모델을 쓸 수 없어요 — 2단계 MODEL을 바꿔 주세요"
        else:
            kind, msg = "error", text[:200]
        return {"openai": kind, "model": MODEL, "message": msg}

class LibItem(BaseModel):
    id: str
    url: Optional[str] = None               # 재료 사진 주소
    hex: Optional[str] = None               # 사진이 없는 색상 재료(페인트·단색 벽지)는 색상 코드만 줘도 돼요
    name: Optional[str] = None              # 표시용 이름 (선택, 예: "퓨어화이트")
    vec: Optional[List[float]] = None       # CLIP 벡터 (선택). 미리 계산해 주면 서버가 빨리 켜짐

class LibSync(BaseModel):
    items: Optional[List[LibItem]] = None
    floors: Optional[List[LibItem]] = None  # 예전 이름도 받아줌
    walls: Optional[List[LibItem]] = None

def _sync(body, setter, label):
    items = body.items or body.floors or body.walls or []
    with LOCK:
        try:
            lib = setter([{"id": f.id, "url": f.url, "hex": f.hex, "name": f.name, "vec": f.vec} for f in items])
        except RuntimeError as e:
            raise AIError(422, "no_usable_materials", str(e))
        for room in ROOMS.values():
            room.pop("laid", None); room.pop("_wall_laid", None)
        return {"kind": label, "count": len(lib), "ids": list(lib),
                "skipped": [f.id for f in items if str(f.id) not in lib]}

@app.post("/floors/sync", dependencies=[Depends(check_key)])
def floors_sync(body: LibSync):
    """백엔드의 바닥재 목록을 통째로 등록해요 (기존 목록은 바뀜)."""
    return _sync(body, set_floor_library, "floor")

@app.post("/walls/sync", dependencies=[Depends(check_key)])
def walls_sync(body: LibSync):
    """백엔드의 벽지 목록을 통째로 등록해요. 등록하면 벽을 벽지 사진으로 합성할 수 있어요.
    {"walls": [{"id": "wall_0001", "hex": "#FFFFFF", "name": "퓨어화이트"},
               {"id": "wall_0002", "url": "https://...", "name": "플라워 패턴"}, ...]}
    **사진(url)이 있으면 그 무늬를 덮어씌우고, hex만 있으면 그 색으로 칠해요.** 둘 다 없는 항목만 건너뜁니다."""
    return _sync(body, set_wall_library, "wall")

class VectorBody(BaseModel):
    items: List[LibItem]

@app.post("/vectors", dependencies=[Depends(check_key)])
def vectors(body: VectorBody):
    """[백엔드용] 재료 사진 주소 → CLIP 벡터. 바닥재·벽지를 **등록할 때 한 번** 불러서 DB에 저장해 두면,
    나중에 /floors/sync · /walls/sync에 vec을 같이 보내서 서버 켤 때 계산을 건너뛸 수 있어요 (서버가 빨리 떠요)."""
    out, failed = [], []
    with LOCK:
        for it in body.items:
            try:
                out.append({"id": it.id, "vec": make_floor_vector({"path": None, "url": it.url})})
            except Exception as e:
                failed.append({"id": it.id, "reason": f"{type(e).__name__}"})
    if not out and failed:
        raise AIError(422, "no_usable_materials", "사진을 하나도 읽지 못했어요", failed)
    return {"count": len(out), "vectors": out, "failed": failed}

class ColorItem(BaseModel):
    hex: str
    id: Optional[str] = None
    name: Optional[str] = None

class ColorSync(BaseModel):
    colors: Optional[List[ColorItem]] = None
    items: Optional[List[ColorItem]] = None

@app.post("/colors/sync", dependencies=[Depends(check_key)])
def colors_sync(body: ColorSync):
    """앱 DB에 등록된 **색 목록**을 넣어요 → 추천 색이 이 목록 안에서만 나와요.
    {"colors": [{"id": "c01", "hex": "#FFFFFF", "name": "퓨어화이트"}, ...]}
    벽 색과 몰딩 색 둘 다 여기서 골라요. 빈 목록을 보내면 제한이 풀려 AI가 만든 색을 그대로 씁니다."""
    items = body.colors or body.items or []
    with LOCK:
        palette = set_color_palette([{"id": c.id, "hex": c.hex, "name": c.name} for c in items])
    return {"count": len(palette), "colors": [{"id": k, "hex": v["hex"], "name": v.get("name")} for k, v in palette.items()],
            "skipped": [c.id or c.hex for c in items if not HEX_PATTERN.match(str(c.hex))]}

@app.get("/colors", dependencies=[Depends(check_key)])
def colors_list():
    return {"count": len(COLOR_PALETTE),
            "colors": [{"id": k, "hex": v["hex"], "name": v.get("name")} for k, v in COLOR_PALETTE.items()]}

@app.get("/floors", dependencies=[Depends(check_key)])
def floors_list():
    return {"count": len(FLOOR_LIBRARY),
            "floors": [{"id": k, "name": v.get("name"), "avg_hex": v["avg_hex"], "kind": "texture" if (v.get("url") or v.get("path")) else "color"}
                       for k, v in FLOOR_LIBRARY.items()]}

@app.get("/walls", dependencies=[Depends(check_key)])
def walls_list():
    return {"count": len(WALL_LIBRARY),
            "walls": [{"id": k, "name": v.get("name"), "avg_hex": v["avg_hex"], "kind": "texture" if (v.get("url") or v.get("path")) else "color"}
                      for k, v in WALL_LIBRARY.items()]}

# ---- 방 만들기 (영역 분할 + 벽 면 나누기) ----
@app.post("/rooms", dependencies=[Depends(check_key)])
def create_room(image: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None),
                image_base64: Optional[str] = Form(None)):
    """방 사진 → 영역 분할 + 벽 면 나누기 + 몰딩 찾기.
    image(파일) / image_url(주소) / image_base64 중 하나를 보내요. 원본 그대로 보내면 서버가 알아서 줄여요."""
    if image is not None:
        data = image.file.read()
    elif image_url:
        data = _download(image_url, "방 사진")
    elif image_base64:
        data = _from_b64(image_base64, "방 사진")
    else:
        raise AIError(422, "invalid_request", "image(파일), image_url, image_base64 중 하나가 필요해요")
    original = _pil_from_bytes(data, "방 사진")
    started = time.time()
    with LOCK:
        full = _resize_max(original, SERVER_MAX_SIDE) if SERVER_MAX_SIDE else original
        small = _resize_max(full, SEG_MAX_SIDE)
        try:
            raw = segment_room(small)
        except Exception as e:
            raise AIError(500, "segmentation_failed", f"영역 분할에 실패했어요 ({type(e).__name__})")
        room = {"image": np.array(full), "raw": raw, "orig_size": original.size, "edited": False}
        reclean(room)
        if not any(m.any() for m in room["masks"].values()):
            raise AIError(422, "no_room_detected", "벽이나 바닥을 찾지 못했어요. 방 안쪽이 보이는 사진인지 확인해 주세요")
        room_id = uuid.uuid4().hex[:12]
        ROOMS[room_id] = room
        while len(ROOMS) > MAX_ROOMS:
            ROOMS.popitem(last=False)
        out = _room_summary(room_id, room)
    out["seconds"] = round(time.time() - started, 2)
    out["warnings"] = []
    if not out["wall_split_found"]:
        out["warnings"].append("벽 모서리를 못 찾아 벽을 하나(wall1)로 봤어요. 가구가 바닥을 가리면 자주 그래요. "
                               "POST /rooms/{room_id}/wall-split 로 직접 정할 수 있어요")
    if not (room.get("moldings") or {}):
        out["warnings"].append("몰딩을 못 찾았어요 (없는 집이거나 벽과 색이 비슷한 경우). moldingTop/moldingBottom은 무시돼요")
    return out

@app.get("/rooms/{room_id}", dependencies=[Depends(check_key)])
def room_info(room_id: str):
    return _room_summary(room_id, _get_room(room_id))

@app.delete("/rooms/{room_id}", dependencies=[Depends(check_key)])
def room_delete(room_id: str):
    with LOCK:
        if ROOMS.pop(room_id, None) is None:
            raise AIError(404, "room_not_found", "이미 지워졌거나 없는 방이에요", {"room_id": room_id})
    return {"deleted": room_id}

@app.get("/rooms/{room_id}/regions", dependencies=[Depends(check_key)])
def room_regions(room_id: str, kind: str = "overlay"):
    """영역 그림. kind=overlay: 색칠한 반투명 PNG / kind=ids: 영역 번호가 값인 회색 PNG (벽 면은 11부터).
    크기는 work_width x work_height."""
    room = _get_room(room_id)
    masks, faces, mold = room["masks"], room.get("wall_faces") or {}, room.get("moldings") or {}
    H, W = masks["wall"].shape
    label = np.zeros((H, W), np.uint8)
    label[masks["ceiling"]] = 3
    label[masks["floor"]] = 2
    for i, name in enumerate(_face_order(room)):
        label[faces[name]] = WALL_FACE_BASE + i
    for key, rid, _ in MOLD_KEYS:
        if key in mold:
            label[mold[key]] = rid
    if kind == "ids":
        return Response(_png(label), media_type="image/png")
    if kind != "overlay":
        raise AIError(422, "invalid_request", "kind는 overlay 또는 ids여야 해요")
    palette = dict(REGION_COLORS)
    for i in range(len(faces)):                                   # 벽 면마다 조금씩 다른 파랑
        palette[WALL_FACE_BASE + i] = (66, 133 - i * 30, 244 - i * 50)
    rgba = np.zeros((H, W, 4), np.uint8)
    for rid, color in palette.items():
        rgba[label == rid] = (*[max(0, c) for c in color], 120)
    return Response(_png(rgba), media_type="image/png")

# ---- 붓으로 영역 고치기 ----
class Stroke(BaseModel):
    region_id: int
    width: float = 20
    points: List[List[float]]

class PolyPiece(BaseModel):
    points: List[List[float]]
    holes: Optional[List[List[List[float]]]] = None

class RegionPoly(BaseModel):
    region_id: int
    points: Optional[List[List[float]]] = None        # 꼭짓점 [[x, y], ...] (원본 사진 픽셀)
    holes: Optional[List[List[List[float]]]] = None   # 안쪽 구멍들 (창문·가구 등)
    polygons: Optional[List[PolyPiece]] = None        # 조각이 여러 개일 때 (가구에 가려 둘로 나뉜 바닥 등)
    mode: str = "replace"                             # "replace"=그 영역을 이 다각형들로 다시 정함 / "cut"=그 안에서 그 영역만 지움

class EditsBody(BaseModel):
    strokes: Optional[List[Stroke]] = None     # 붓으로 칠하기 (먼저 적용)
    polygons: Optional[List[RegionPoly]] = None  # 꼭짓점 편집·누끼선 (붓 다음에 적용)
    regions: Optional[List[RegionPoly]] = None   # polygons의 옛 이름 (그대로 받아줘요)

@app.post("/rooms/{room_id}/edits", dependencies=[Depends(check_key)])
def room_edits(room_id: str, body: EditsBody):
    """영역을 고쳐요 — **붓(strokes)** 또는 **꼭짓점 다각형(regions)**. 둘 다 좌표는 원본 사진 픽셀 기준.
    strokes: [{"region_id": 1, "width": 40, "points": [[x,y], ...]}]  — 칠하듯이 덧그리기 (**먼저** 적용)
             region_id: 0 지우개 / 1 벽 / 2 바닥 / 3 천장 / 4 천장몰딩 / 5 걸레받이
    polygons: [{"region_id": 1, "mode": "replace"|"cut", "points": [[x,y], ...], "holes": [[[x,y], ...]]}] (**나중에** 적용)
             조각이 여러 개면 한 항목 안에 "polygons": [{"points","holes"}, ...] 로 넣어도 되고,
             같은 region_id를 여러 항목으로 보내도 돼요 (replace여도 서로 지우지 않아요)
             region_id 0은 '없음'이라 그 다각형 안을 비워요
             · replace (꼭짓점 편집): 그 영역을 비우고 → 다각형으로 채우고 → holes는 다시 비우기
             · cut (누끼선): 다각형 안에서 그 영역인 픽셀만 0으로
             mode를 안 주면 replace예요. 옛 이름 "regions"로 보내도 똑같이 동작해요.
    둘을 같이 보내도 되고, 응답의 regions에는 고쳐진 뒤의 polygons가 들어 있어요 (그대로 다시 그리면 됨).
    (벽 면 wall1~은 여기가 아니라 /wall-split 으로 나눠요)"""
    room = _get_room(room_id)
    strokes = body.strokes or []
    polys = (body.polygons or []) + (body.regions or [])      # regions는 옛 이름
    if not strokes and not polys:
        raise AIError(422, "invalid_request", "strokes 또는 polygons 중 하나는 있어야 해요")
    with LOCK:
        label = _label_from_room(room)
        H, W = label.shape
        ow, oh = room["orig_size"]
        sx, sy = W / ow, H / oh
        to_work = lambda pts: np.round(np.array([[p[0] * sx, p[1] * sy] for p in pts], np.float32)).astype(np.int32)
        for k, st in enumerate(strokes):
            if st.region_id not in (0, 1, 2, 3, 4, 5):
                raise AIError(422, "invalid_region", "region_id는 0(지우개), 1(벽), 2(바닥), 3(천장), 4(천장몰딩), 5(걸레받이) 중 하나여야 해요",
                              {"stroke": k, "region_id": st.region_id})
            pts = [p for p in st.points if len(p) >= 2]
            if not pts:
                continue
            ipts = to_work(pts).reshape(-1, 1, 2)
            thick = max(1, int(round(st.width * (sx + sy) / 2)))
            if len(ipts) == 1:
                cv2.circle(label, (int(ipts[0, 0, 0]), int(ipts[0, 0, 1])), max(1, thick // 2), int(st.region_id), -1)
            else:
                cv2.polylines(label, [ipts], False, int(st.region_id), thick)
        cleared = set()                                       # replace는 한 영역당 한 번만 비워요 (조각이 여러 개여도)
        for k, pg in enumerate(polys):                        # 붓 다음에 다각형 (프론트 약속 순서)
            if pg.region_id not in (0, 1, 2, 3, 4, 5):
                raise AIError(422, "invalid_region",
                              "polygons의 region_id는 0(없음), 1(벽), 2(바닥), 3(천장), 4(천장몰딩), 5(걸레받이)여야 해요 "
                              "(벽 면 wall1~은 POST /rooms/{room_id}/wall-split 으로 나눠요)",
                              {"polygon": k, "region_id": pg.region_id})
            if pg.mode not in ("replace", "cut"):
                raise AIError(422, "invalid_request", 'polygons의 mode는 "replace" 또는 "cut"이어야 해요',
                              {"polygon": k, "mode": pg.mode})
            pieces = [(p.points, p.holes) for p in (pg.polygons or [])]
            if pg.points:
                pieces.append((pg.points, pg.holes))
            pieces = [(pts, holes) for pts, holes in pieces if pts and len(pts) >= 3]
            if not pieces:
                raise AIError(422, "invalid_request", "다각형에는 꼭짓점이 3개 이상 필요해요 (points 또는 polygons)", {"polygon": k})
            rid = int(pg.region_id)
            if pg.mode == "replace" and rid not in cleared:   # ① 그 번호를 비우고 (한 번만)
                label[label == rid] = 0
                cleared.add(rid)
            for pts_raw, holes_raw in pieces:
                pts = to_work(pts_raw)
                holes = [to_work(h) for h in (holes_raw or []) if len(h) >= 3]
                if pg.mode == "replace":                      # ② 다각형을 채우고 ③ 구멍을 0으로
                    cv2.fillPoly(label, [pts], rid)
                    for h in holes:
                        cv2.fillPoly(label, [h], 0)
                else:                                         # cut(누끼선): 다각형 안에서 그 번호인 픽셀만 0으로
                    inside = np.zeros_like(label)
                    cv2.fillPoly(inside, [pts], 1)
                    for h in holes:
                        cv2.fillPoly(inside, [h], 0)
                    label[(inside == 1) & (label == rid)] = 0
        _apply_label(room, label)
        out = _room_summary(room_id, room)
    out["applied_strokes"], out["applied_polygons"] = len(strokes), len(polys)
    out["applied_regions"] = out["applied_polygons"]          # 옛 이름
    return out

class WallSplitBody(BaseModel):
    x: Optional[List[float]] = None        # 벽 모서리 두 곳 (원본 사진 가로 픽셀)
    auto: bool = False                     # true면 자동으로 다시 찾음

@app.post("/rooms/{room_id}/wall-split", dependencies=[Depends(check_key)])
def room_wall_split(room_id: str, body: WallSplitBody):
    """벽을 왼쪽/정면/오른쪽으로 나눌 모서리를 직접 정해요. {"x": [400, 1500]} 또는 {"auto": true}
    좌표는 원본 사진 픽셀 기준. 사용자가 화면에서 선을 끌어 정하게 할 때 쓰면 돼요."""
    room = _get_room(room_id)
    with LOCK:
        H, W = room["masks"]["wall"].shape
        ow, _ = room["orig_size"]
        if body.auto or not body.x:
            set_wall_split(room, None)
        else:
            if len(body.x) != 2:
                raise AIError(422, "invalid_request", "x는 모서리 두 곳이어야 해요. 예: {\"x\": [400, 1500]}")
            set_wall_split(room, [max(0, min(W - 1, int(v * W / ow))) for v in body.x])
        room["edited"] = True
        return _room_summary(room_id, room)

@app.post("/rooms/{room_id}/reset", dependencies=[Depends(check_key)])
def room_reset(room_id: str):
    """붓 수정과 모서리 지정을 처음 상태로 되돌려요."""
    room = _get_room(room_id)
    with LOCK:
        global WALL_SPLIT
        WALL_SPLIT = None
        reclean(room)
        room["edited"] = False
        return _room_summary(room_id, room)

# ---- 레퍼런스 분석 ----
class RecommendBody(BaseModel):
    reference_urls: Optional[List[str]] = None
    reference_base64: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None   # 사진 없이 선택값만으로 추천 {"tone": "웜톤", "texture": "우드"}
    n_proposals: Optional[int] = None       # 시안 개수 (기본: 2단계의 N_PROPOSALS, 1~5)

@app.post("/recommend", dependencies=[Depends(check_key)])
def recommend(body: RecommendBody):
    """레퍼런스 사진 1~5장 **또는** 선택값(preferences) → 취향 키워드 + 시안.
    사진 없이 쓰려면: {"preferences": {"tone": "웜톤", "texture": "우드"}} (키 이름은 앱 마음대로, 값은 한국어 그대로)
    둘 다 주면 사진을 보면서 선택값도 같이 고려해요.
    바닥재는 /floors/sync, 벽지는 /walls/sync로 등록한 목록에서 CLIP이 골라요.
    응답의 warmth / style / keywords_ko 는 백엔드가 시안과 함께 저장해 두면 '취향 태그' 화면에 바로 쓸 수 있어요."""
    sources = list(body.reference_urls or []) + list(body.reference_base64 or [])
    def _pref_value(v):                                   # 배열로 와도 되게 ({"keywords": ["모던", "우드"]})
        if isinstance(v, (list, tuple, set)):
            return ", ".join(str(x).strip() for x in v if str(x).strip())
        return str(v).strip()
    prefs = {k: _pref_value(v) for k, v in (body.preferences or {}).items()} if body.preferences else None
    prefs = {k: v for k, v in (prefs or {}).items() if v} or None
    if not sources and not prefs:
        raise AIError(422, "invalid_request",
                      'reference_urls·reference_base64(사진) 또는 preferences(선택값) 중 하나는 필요해요. '
                      '예: {"preferences": {"tone": "웜톤", "texture": "우드"}}')
    n = None if body.n_proposals is None else max(1, min(5, int(body.n_proposals)))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, src in enumerate(sources[:MAX_REFERENCES]):
            data = _download(src, "레퍼런스 사진") if str(src).startswith("http") else _from_b64(src, "레퍼런스 사진")
            path = os.path.join(tmp, f"ref{i}.jpg")
            _pil_from_bytes(data, "레퍼런스 사진").save(path, quality=92)
            paths.append(path)
        with LOCK, override_globals(N_PROPOSALS=n):
            try:
                analysis, seconds = analyze_reference(paths, preferences=prefs)
            except RuntimeError as e:
                msg = str(e)
                if "키가 없어요" in msg:
                    raise AIError(503, "llm_not_configured", msg)
                if msg.startswith("OpenAI 사용 불가"):
                    raise AIError(502, "llm_unavailable", msg)
                raise AIError(502, "llm_failed", msg)
            ref_vec = reference_floor_vector(paths) if FLOOR_FROM_REFERENCE else None
            pick_floors(analysis["proposals"], ref_vec)
            pick_walls(analysis["proposals"])
            snap_proposals(analysis["proposals"])            # 벽·몰딩 색을 등록된 색 목록 안으로
    analysis["graph"] = build_graph(analysis)                # 키워드 그래프 (nodes / edges, 가중치 0~1)
    analysis["seconds"] = round(seconds, 2)
    analysis["warnings"] = []
    if list(FLOOR_LIBRARY) == ["sample_marble"]:
        analysis["warnings"].append("바닥재 목록이 비어 임시 가짜 바닥재만 쓰고 있어요. 먼저 POST /floors/sync를 호출해 주세요")
    if not WALL_LIBRARY:
        analysis["warnings"].append("벽지 목록이 비어 벽은 색상 코드로만 제안해요. 벽지를 쓰려면 POST /walls/sync를 호출해 주세요")
    if not COLOR_PALETTE:
        analysis["warnings"].append("색 목록이 비어 AI가 만든 색을 그대로 써요 (DB에 없는 색이 나올 수 있어요) — POST /colors/sync로 등록하면 그 안에서만 골라요")
    return analysis

# ---- 시안 만들기 ----
class GenOptions(BaseModel):
    wall_method: Optional[str] = None      # "auto"(기본) / "overlay"(항상 덮어씌움) / "filter"(항상 필터)
    detail: float = 1.0                    # 필터 방식에서 원래 벽 결을 반영하는 정도 (0~1.3)
    pattern: Optional[float] = None        # 덮어씌울 때 옛 무늬를 지우는 정도 (0~1)
    wall_tile: Optional[float] = None      # 벽지 무늬 한 칸의 가로 크기 (이미지 가로 대비)
    wall_keep: Optional[float] = None      # 벽: 원래 빛 유지 (0~1.6)
    wall_spec: Optional[float] = None      # 벽: 반짝임
    wall_exposure: Optional[float] = None  # 벽지 밝기
    keep: Optional[float] = None           # 바닥: 원래 빛 유지
    spec: Optional[float] = None           # 바닥: 하이라이트
    reflect: Optional[float] = None        # 바닥: 반사 (0~1)
    rough: Optional[float] = None          # 바닥: 반사 흐림
    exposure: Optional[float] = None       # 바닥재 밝기
    horizon: Optional[float] = None        # 수평선 높이 (0~1, 위가 0)
    fov: Optional[float] = None
    rotate: Optional[float] = None         # 바닥 무늬 회전(도)
    tile: Optional[float] = None           # 바닥재 한 칸 크기
    ceiling: Optional[str] = None          # "auto" / "none" / "#RRGGBB"
    molding: Optional[str] = None          # "wall"(벽과 같게) / "keep"(원래대로) / "#RRGGBB"

class GenerateBody(BaseModel):
    materials: Dict[str, Any]              # {"wall1": {"id": "wp_01"}, "floor": {"id": "fl_02"}, "allWalls": {"hex": "#F3EBDD"}}
    options: GenOptions = GenOptions()
    preview: bool = False                  # true면 결과를 작게(긴 변 512px) 줄여서 보냄
    quality: int = 90                      # JPEG 품질 (50~100)

def _as_ref(value, key):
    if isinstance(value, str):
        return {"hex": value} if HEX_PATTERN.match(value) else {"id": value}
    if isinstance(value, dict):
        ref = {k: value.get(k) for k in ("id", "url", "hex", "texture_base64")}   # 그 밖의 키는 조용히 무시
        if ref["hex"] is not None and not HEX_PATTERN.match(str(ref["hex"])):
            raise AIError(422, "invalid_material", f"{key}: hex는 #RRGGBB 형식이어야 해요", {"hex": ref["hex"]})
        try:                                        # repeat: 정면 벽(바닥은 화면 맨 아래) 가로 폭에 무늬가 몇 번 반복되는지
            rep = float(value.get("repeat")) if value.get("repeat") is not None else None
            ref["repeat"] = min(20.0, max(0.5, rep)) if rep is not None and np.isfinite(rep) and rep > 0 else None
        except (TypeError, ValueError):
            ref["repeat"] = None                    # 숫자가 아니면 무시
        return ref
    raise AIError(422, "invalid_material", f"{key}: 재료는 id/hex 문자열이나 {{id, url, hex, texture_base64}} 객체여야 해요")

def library_color(ref, library):
    """등록된 재료가 **사진 없는 색상 재료**면 그 색(hex)을 돌려줘요 (페인트·단색 벽지). 아니면 None."""
    fid = ref.get("id")
    entry = library.get(str(fid)) if fid else None
    if entry and not (entry.get("url") or entry.get("path")):
        return entry["avg_hex"]
    return None

def _texture_from(ref, library, prefix, what):
    """재료 참조 → 사진 배열 (없으면 None). id는 등록된 목록에서, url·base64는 그때그때 내려받아 캐시."""
    fid = ref.get("id")
    if fid and str(fid) in library:
        if library_color(ref, library):            # 색상 재료는 사진이 아니라 색으로 칠해요
            return None
        return get_floor_texture(str(fid), library, prefix=prefix)
    raw = None
    if ref.get("texture_base64"):
        raw = _from_b64(ref["texture_base64"], what)
    elif ref.get("url"):
        key = prefix + "url:" + ref["url"]
        if key in EXTRA_TEX:
            return EXTRA_TEX[key]
        raw = _download(ref["url"], what)
    if raw is None:
        return None
    img = _resize_max(_pil_from_bytes(raw, what), 1024)
    tex = np.array(img)
    key = prefix + (("url:" + ref["url"]) if ref.get("url") else ("b64:" + str(hash(ref["texture_base64"]))))
    EXTRA_TEX[key] = tex
    while len(EXTRA_TEX) > 20:
        EXTRA_TEX.popitem(last=False)
    return tex

def _wall_material(ref, key, warnings):
    """벽/몰딩 재료 → 벽지 사진(배열) 또는 색상 코드. 벽지 사진이 있으면 사진을 우선 씀.
    사진(url·base64)을 못 불러왔는데 hex가 같이 왔으면, 오류 대신 hex 색으로 칠하고 경고만 남겨요."""
    try:
        tex = _texture_from(ref, WALL_LIBRARY, "wall:", "벽지 사진")
    except AIError as e:
        if ref.get("hex") and e.code in ("download_failed", "invalid_image"):
            warnings.append(f"{key}: 무늬 사진을 못 불러와서 hex 색으로 칠했어요 ({e.message})")
            return ref["hex"]
        raise
    if tex is not None:
        return tex
    if ref.get("hex"):
        return ref["hex"]
    color = library_color(ref, WALL_LIBRARY)       # 등록된 색상 벽지(hex만 있는 것)
    if color:
        return color
    cid = str(ref.get("id")) if ref.get("id") is not None else None
    if cid and cid in COLOR_PALETTE:               # /colors/sync로 등록된 색 id (몰딩 후보 id 등)
        return COLOR_PALETTE[cid]["hex"]
    raise AIError(422, "material_unresolved",
                  f"{key}: 벽 재료를 알 수 없어요. 등록된 벽지 id, url, texture_base64, 또는 hex 중 하나가 필요해요 "
                  f"(벽지는 POST /walls/sync로 먼저 등록)", {"material": ref.get("id")})

@app.post("/rooms/{room_id}/generate", dependencies=[Depends(check_key)])
def room_generate(room_id: str, body: GenerateBody):
    """방에 재료를 입혀요 → 합성 사진(base64).
    materials의 영역 이름: allWalls / wall1, wall2, ... / floor / ceiling / moldingTop / moldingBottom
    값은 {"id": 등록한 재료} 또는 {"hex": "#RRGGBB"} 또는 {"url": ...} 또는 {"texture_base64": ...}
    allWalls와 wall1~을 같이 주면 **개별 면이 우선**이고, 안 준 면은 allWalls로 칠해요."""
    room = _get_room(room_id)
    started = time.time()
    with LOCK:
        warnings, applied = [], {}
        faces = _face_order(room)
        all_ref, per_face, floor_ref, ceil_ref, mold_refs = None, {}, None, None, {}
        for key, raw in body.materials.items():
            ref = _as_ref(raw, key)
            if key == "allWalls":
                all_ref = ref
            elif key in faces:
                per_face[key] = ref
            elif re.fullmatch(r"wall\d+", key):
                warnings.append(f"{key}: 이 사진에는 그 벽 면이 없어요 (지금 면: {', '.join(faces) or '없음'})")
            elif key == "floor":
                floor_ref = ref
            elif key == "ceiling":
                ceil_ref = ref
            elif key in ("moldingTop", "moldingBottom"):
                mold_refs[key] = ref
            else:
                raise AIError(422, "unknown_region", f"알 수 없는 영역 이름이에요: {key}",
                              {"allowed": ["allWalls"] + faces + ["floor", "ceiling", "moldingTop", "moldingBottom"]})
        # 벽: 면별로 재료 정하기 (안 준 면은 allWalls)
        walls = {}
        for name in faces:
            ref = per_face.get(name) or all_ref
            if ref:
                walls[name] = _wall_material(ref, name, warnings)
                applied[name] = ref.get("id") or ref.get("hex") or "url"
        if all_ref and not per_face:
            applied = {"allWalls": all_ref.get("id") or all_ref.get("hex") or "url",
                       **{k: v for k, v in applied.items() if k not in faces}}
        # 몰딩
        molding_mat = None
        if mold_refs:
            found = room.get("moldings") or {}
            if not found:
                warnings.append("몰딩을 못 찾은 사진이라 moldingTop/moldingBottom은 무시했어요")
            else:
                molding_mat = {}
                for key, part in (("moldingTop", "molding_top"), ("moldingBottom", "molding_bottom")):
                    ref = mold_refs.get(key)
                    if ref is None:
                        continue
                    if part not in found:
                        warnings.append(f"이 사진에서는 {key}를 못 찾아서 그 부분은 건너뛰었어요")
                        continue
                    molding_mat[part] = _wall_material(ref, key, warnings)
                    applied[key] = ref.get("id") or ref.get("hex") or "url"
                if not molding_mat:
                    molding_mat = None
        # 바닥
        floor_tex, floor_hex = None, None
        if floor_ref:
            try:
                floor_tex = _texture_from(floor_ref, FLOOR_LIBRARY, "", "바닥재 사진")
            except AIError as e:                       # 사진을 못 불러왔는데 hex가 있으면 그 색으로
                if not (floor_ref.get("hex") and e.code in ("download_failed", "invalid_image")):
                    raise
                warnings.append(f"floor: 무늬 사진을 못 불러와서 hex 색으로 칠했어요 ({e.message})")
                floor_tex = None
            if floor_tex is None:
                floor_hex = floor_ref.get("hex") or library_color(floor_ref, FLOOR_LIBRARY)
                if not floor_hex:
                    raise AIError(422, "material_unresolved",
                                  f"floor: 바닥재 '{floor_ref.get('id')}'를 찾을 수 없어요. POST /floors/sync로 등록하거나 url·hex를 주세요",
                                  {"material": floor_ref.get("id")})
            applied["floor"] = floor_ref.get("id") or floor_ref.get("hex") or "url"
        if not walls and floor_tex is None and floor_hex is None:
            raise AIError(422, "nothing_to_apply", "적용할 수 있는 재료가 없어요 (allWalls, wall1~, floor 중 하나가 필요해요)")
        masks = room["masks"]
        if walls and not masks["wall"].any():
            warnings.append("벽 영역이 비어 있어 벽 재료를 적용하지 못했어요")
        if (floor_tex is not None or floor_hex) and not masks["floor"].any():
            warnings.append("바닥 영역이 비어 있어 바닥재를 적용하지 못했어요")
        o = body.options
        # repeat(무늬 반복 횟수) → 무늬 한 칸 크기. options.wall_tile / options.tile을 직접 주면 그게 우선
        wall_tile, floor_tile = o.wall_tile, o.tile
        reps = {r["repeat"] for r in list(per_face.values()) + [all_ref] if r and r.get("repeat")}
        if wall_tile is None and reps:
            rep = (all_ref or {}).get("repeat") or sorted(reps)[0]
            wall_tile = 1.0 / rep
            applied["wallRepeat"] = rep
            if len(reps) > 1:
                warnings.append(f"벽 repeat는 한 값만 적용돼요 ({rep}회). 면마다 다른 반복 횟수는 아직 지원하지 않아요")
        if floor_tile is None and floor_ref and floor_ref.get("repeat"):
            floor_tile = 1.0 / floor_ref["repeat"]
            applied["floorRepeat"] = floor_ref["repeat"]
        if o.wall_method not in (None, "auto", "overlay", "filter"):
            raise AIError(422, "invalid_request", 'options.wall_method는 "auto", "overlay", "filter" 중 하나여야 해요',
                          {"wall_method": o.wall_method})
        ceiling_hex = o.ceiling
        if ceil_ref:
            ceiling_hex = ceil_ref.get("hex") or ceiling_hex
            applied["ceiling"] = ceiling_hex
        used_methods = {name: ("overlay" if not isinstance(mat, str) else wall_method_for(room, room["wall_faces"][name], o.wall_method))
                        for name, mat in walls.items()}          # 면마다 실제로 쓴 방식 (프론트 안내용)
        with override_globals(HORIZON=o.horizon, FOV=o.fov, ROTATE=o.rotate, FLOOR_TILE=floor_tile):
            laid = None
            if floor_tex is not None and masks["floor"].any():
                cache_key = (id(floor_tex), HORIZON, FOV, ROTATE, FLOOR_TILE)
                cache = room.setdefault("laid", {})
                if cache_key not in cache:
                    cache[cache_key] = floor_perspective(room["image"].shape[:2], masks["floor"], floor_tex,
                                                         FLOOR_TILE, HORIZON, FOV, ROTATE)
                laid = cache[cache_key]
            out = recolor(room, None, floor_hex, o.detail, floor_laid=laid, walls=walls, method=o.wall_method,
                          ceiling_hex=ceiling_hex, molding=molding_mat, pattern=o.pattern, wall_tile=wall_tile,
                          keep=o.keep, spec=o.spec, reflect=o.reflect, rough=o.rough, exposure=o.exposure,
                          wall_keep=o.wall_keep, wall_spec=o.wall_spec, wall_exposure=o.wall_exposure)
    img = Image.fromarray(out)
    if body.preview:
        img = _resize_max(img, PREVIEW_SIDE)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=max(50, min(100, body.quality)))
    return {"room_id": room_id, "image_base64": base64.b64encode(buf.getvalue()).decode(), "mime": "image/jpeg",
            "width": img.width, "height": img.height, "applied": applied, "wall_methods": used_methods, "warnings": warnings,
            "seconds": round(time.time() - started, 2)}

def warm_up():
    """서버를 켤 때 분할 모델·CLIP을 미리 올려둬요 → **첫 요청이 타임아웃으로 죽지 않게.**
    (안 하면 첫 POST /rooms에서 모델을 받느라 수십 초~몇 분 걸려서 프론트가 '분석 안 돼요'로 보여요)"""
    def run():
        try:
            with LOCK:
                t = time.time(); get_segmenter(); get_clip()
                print(f"✅ 모델 준비 완료 ({time.time() - t:.0f}초) — 이제 첫 요청도 몇 초면 돼요")
        except Exception as e:
            print(f"⚠️ 모델 미리 올리기 실패 ({type(e).__name__}) — 첫 요청 때 다시 시도해요")
    threading.Thread(target=run, daemon=True).start()



@app.get("/")
def root():
    """서버가 살아있는지 브라우저로 바로 확인하는 자리."""
    return {"service": "imagine-ai", "docs": "/docs", "health": "/health", "selfcheck": "/selfcheck"}


@app.on_event("startup")
def _warm_on_start():
    """uvicorn으로 띄워도 모델을 미리 올려요 (run.sh가 이 경로예요)."""
    warm_up()


# ---- 직접 실행할 때 ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f"🚀 http://0.0.0.0:{port}  (명세: /docs, 연결 확인: /health, 문제 점검: /selfcheck)")
    uvicorn.run(app, host="0.0.0.0", port=port)
