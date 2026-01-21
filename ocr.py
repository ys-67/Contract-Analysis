import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def imread_unicode(path: str) -> np.ndarray:
    """
    Windows 등에서 한글 경로 대응.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def imwrite_unicode(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    ok, buf = cv2.imencode(ext if ext else ".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode image to {path}")
    buf.tofile(path)

def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    cur_long = max(h, w)
    if cur_long <= long_side:
        return img
    scale = long_side / float(cur_long)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def upscale(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def illumination_correction(gray: np.ndarray) -> np.ndarray:
    """
    조명 불균일(그림자/밝은 스팟) 보정: divide 방식
    """
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    blur = np.clip(blur, 1, 255).astype(np.uint8)
    corrected = cv2.divide(gray, blur, scale=255)
    return corrected

def clahe(gray: np.ndarray) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return c.apply(gray)

def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

def unsharp_mask(gray: np.ndarray, amount: float = 1.2, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return sharp

def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def binarize_adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

def deskew(gray_or_bin: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    텍스트가 기울어진 스캔에 대해 deskew.
    - 입력이 grayscale/bi-level 모두 가능
    - 반환: (회전된 이미지, 회전각도)
    """
    if len(gray_or_bin.shape) == 3:
        gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bin

    # 텍스트를 전경으로 만들기 위해 반전 + 이진화
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(th > 0))
    if coords.size < 200:  # 텍스트가 거의 없으면 스킵
        return gray_or_bin, 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # minAreaRect 각도 보정 규칙
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = gray_or_bin.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray_or_bin, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle

# -----------------------------
# OCR result handling
# -----------------------------
@dataclass
class OCRCandidateResult:
    name: str
    score: float
    mean_conf: float
    num_tokens: int
    text: str
    raw: Any

def flatten_ocr_result(result: Any) -> Tuple[str, List[float]]:
    """
    PaddleOCR ocr() 반환 구조에서 (전체 텍스트, confidence 리스트)를 추출
    """
    texts = []
    confs = []
    if not result:
        return "", confs

    # PaddleOCR: result = [ [ [box], (text, conf) ], ... ]  (이미지 1장일 때)
    # 또는 result[0]에 라인들이 들어있음.
    lines = result[0] if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list) else result

    for line in lines:
        if not line or len(line) < 2:
            continue
        info = line[1]
        if isinstance(info, (list, tuple)) and len(info) >= 2:
            txt, conf = info[0], info[1]
            if txt is None:
                continue
            texts.append(str(txt))
            try:
                confs.append(float(conf))
            except Exception:
                pass

    full_text = "\n".join(texts).strip()
    return full_text, confs

def compute_score(text: str, confs: List[float]) -> Tuple[float, float, int]:
    """
    후보 결과 중 가장 좋은 것을 선택하기 위한 점수.
    - 평균 confidence 중심
    - 텍스트 길이가 너무 짧으면(검출 실패) 패널티
    """
    if not text:
        return 0.0, 0.0, 0

    mean_conf = float(np.mean(confs)) if confs else 0.0
    n = len(text)

    # 길이 기반 완만한 가중치(너무 짧은 결과 방지)
    length_bonus = min(1.0, np.log1p(n) / 6.0)  # 0~1
    score = mean_conf * (0.75 + 0.25 * length_bonus)

    # 숫자/특수문자만 잔뜩이면 약간 패널티(원하면 제거 가능)
    alpha_ratio = sum(ch.isalnum() for ch in text) / max(1, n)
    score *= (0.85 + 0.15 * alpha_ratio)

    return score, mean_conf, n

# -----------------------------
# Main
# -----------------------------
def build_candidates(bgr: np.ndarray, max_side: int) -> List[Tuple[str, np.ndarray]]:
    """
    다양한 전처리 후보 생성.
    """
    bgr_r = resize_long_side(bgr, max_side)

    gray = to_gray(bgr_r)
    gray_dn = denoise(gray)
    gray_illum = illumination_correction(gray_dn)
    gray_clahe = clahe(gray_illum)
    gray_sharp = unsharp_mask(gray_clahe, amount=1.0, sigma=1.0)

    otsu = binarize_otsu(gray_sharp)
    adapt = binarize_adaptive(gray_sharp)

    # deskew는 원본/그레이/이진 모두 시도
    gray_deskew, a1 = deskew(gray_sharp)
    otsu_deskew, a2 = deskew(otsu)
    adapt_deskew, a3 = deskew(adapt)

    # 업스케일(작은 글자에 유리한 경우가 많음)
    bgr_up = upscale(bgr_r, 2.0)
    gray_up = upscale(gray_sharp, 2.0)
    otsu_up = upscale(otsu, 2.0)
    adapt_up = upscale(adapt, 2.0)

    candidates = [
        ("00_original", bgr_r),
        ("01_gray_sharp", gray_sharp),
        ("02_otsu", otsu),
        ("03_adaptive", adapt),
        ("04_gray_deskew", gray_deskew),
        ("05_otsu_deskew", otsu_deskew),
        ("06_adapt_deskew", adapt_deskew),
        ("07_bgr_up2x", bgr_up),
        ("08_gray_up2x", gray_up),
        ("09_otsu_up2x", otsu_up),
        ("10_adapt_up2x", adapt_up),
    ]
    return candidates

def run_ocr_best(
    image_path: str,
    outdir: str,
    lang: str = "korean",
    use_gpu: bool = False,
    max_side: int = 2400,
    det_limit_side_len: int = 2400,
) -> OCRCandidateResult:
    ensure_dir(outdir)

    bgr = imread_unicode(image_path)
    candidates = build_candidates(bgr, max_side=max_side)

    # PaddleOCR 초기화 (정확도 우선 세팅)
    ocr = PaddleOCR(
        lang=lang,
        use_angle_cls=True,       # 90/180 회전 등 보정 (정확도에 매우 중요)
        show_log=False,
        use_gpu=use_gpu,
        det_limit_side_len=det_limit_side_len,  # 큰 문서 이미지에서 검출 성능 개선
        det_db_thresh=0.3,        # 검출 민감도 (낮추면 더 많이 잡지만 노이즈 증가 가능)
        det_db_box_thresh=0.5,    # 박스 필터
        det_db_unclip_ratio=1.6,  # 박스 확장 (좁게 잘리는 케이스 개선)
        rec_batch_num=8,
        max_text_length=80
    )

    best: OCRCandidateResult = OCRCandidateResult(
        name="none", score=0.0, mean_conf=0.0, num_tokens=0, text="", raw=None
    )

    debug_dir = os.path.join(outdir, "debug_candidates")
    ensure_dir(debug_dir)

    for name, img in candidates:
        # 후보 이미지 저장(원하면 주석 처리)
        save_path = os.path.join(debug_dir, f"{name}.png")
        if img.ndim == 2:
            imwrite_unicode(save_path, img)
        else:
            imwrite_unicode(save_path, img)

        # PaddleOCR 입력은 path 또는 ndarray 모두 가능
        try:
            result = ocr.ocr(img, cls=True)
        except Exception:
            # 일부 버전에서 grayscale/binary ndarray 처리 문제 있을 수 있어 BGR로 강제 변환
            if img.ndim == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            result = ocr.ocr(img_bgr, cls=True)

        text, confs = flatten_ocr_result(result)
        score, mean_conf, n = compute_score(text, confs)

        if score > best.score:
            best = OCRCandidateResult(
                name=name, score=score, mean_conf=mean_conf, num_tokens=n, text=text, raw=result
            )

    # 결과 저장
    out_txt = os.path.join(outdir, "result_best.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(best.text + "\n")

    out_json = os.path.join(outdir, "result_best.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_candidate": best.name,
                "score": best.score,
                "mean_conf": best.mean_conf,
                "num_chars": best.num_tokens,
                "text": best.text,
                "raw": best.raw,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 박스 시각화 이미지 저장(최적 후보)
    try:
        # draw_ocr는 PIL 기반
        # raw 구조에서 boxes/texts/scores 뽑기
        lines = best.raw[0] if isinstance(best.raw, list) and len(best.raw) == 1 else best.raw
        boxes, txts, scores = [], [], []
        for line in lines:
            if not line or len(line) < 2:
                continue
            boxes.append(line[0])
            txts.append(line[1][0])
            scores.append(line[1][1])

        # 원본을 시각화 대상으로(가독성 좋음)
        img_for_vis = imread_unicode(image_path)
        img_for_vis = resize_long_side(img_for_vis, max_side)

        vis = draw_ocr(
            Image.fromarray(cv2.cvtColor(img_for_vis, cv2.COLOR_BGR2RGB)),
            boxes,
            txts,
            scores,
            font_path=None  # 시스템 폰트로도 대개 충분. 필요하면 한글 ttf 경로 지정.
        )
        vis_bgr = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        imwrite_unicode(os.path.join(outdir, "result_best_vis.png"), vis_bgr)
    except Exception:
        # 시각화 실패는 텍스트 결과와 무관하므로 무시
        pass

    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="image.jpg")
    parser.add_argument("--outdir", type=str, default="ocr_out")
    parser.add_argument("--lang", type=str, default="korean")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max_side", type=int, default=2400)
    parser.add_argument("--det_limit_side_len", type=int, default=2400)
    args = parser.parse_args()

    use_gpu = False
    if args.gpu:
        try:
            import paddle
            use_gpu = bool(paddle.is_compiled_with_cuda())
        except Exception:
            use_gpu = False

    best = run_ocr_best(
        image_path=args.image,
        outdir=args.outdir,
        lang=args.lang,
        use_gpu=use_gpu,
        max_side=args.max_side,
        det_limit_side_len=args.det_limit_side_len,
    )

    print("=== BEST OCR RESULT ===")
    print(f"Best candidate: {best.name}")
    print(f"Score: {best.score:.4f}, mean_conf: {best.mean_conf:.4f}, num_chars: {best.num_tokens}")
    print("\n[TEXT]\n")
    print(best.text)

if __name__ == "__main__":
    main()
