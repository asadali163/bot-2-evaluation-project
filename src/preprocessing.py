"""Turning a scanned or photographed drawing into a clean black-and-white ink mask.

Developed and tested in notebooks/circle/02_circle_preprocessing.ipynb, which explains every step with examples;
the frame removal was improved for large drawings in notebooks/square/01_square_preprocessing.ipynb.
"""
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold

BORDER = 0.08             # outer band of the image where the box frame can be
HIGH_RATIO = 0.20         # a stroke must reach 20% of the image's strongest ink somewhere...
LOW_RATIO = 0.06          # ...and is then followed down to 6% of it
MIN_INK_FRACTION = 0.003  # a drawing covers ~1-7% of the image: less means the ink was missed,
MAX_INK_FRACTION = 0.15   # more means paper texture was picked up -> colour-only fallback
OUTPUT_SIZE = 512         # saved masks are 512 x 512 pixels


def load_image(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, transparency channel dropped


def fill_white_patches(image):
    """Fill the white anonymisation patches of grey photos from the colours around them (inpainting)."""
    paper = np.median(image.reshape(-1, 3), axis=0)  # most pixels are paper
    if paper.mean() > 235:  # white scan: patches are invisible, nothing to do
        return image
    patch = (image.min(axis=2) >= 248).astype(np.uint8)  # pure white pixels
    patch = cv2.dilate(patch, np.ones((9, 9), np.uint8))  # also cover the patch edges
    if patch.sum() == 0:
        return image
    return cv2.inpaint(image, patch, 5, cv2.INPAINT_TELEA)  # fill from the surrounding colours


def ink_strength(image, colour_only=False):
    """0 = paper, 1 = strong ink: the larger of darkness and colourfulness relative to the local paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Local paper brightness: a max-filter removes the thin dark strokes, a median filter smooths the result
    paper_gray = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    paper_gray = cv2.medianBlur(paper_gray.astype(np.uint8), 31).astype(np.float32)
    darkness = np.clip(1 - gray / np.maximum(paper_gray, 1), 0, 1)

    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    paper_saturation = cv2.medianBlur(saturation, 31)
    colourfulness = np.clip((saturation.astype(np.float32) - paper_saturation) / 255, 0, 1)

    if colour_only:
        return colourfulness
    return np.maximum(darkness, colourfulness)


def threshold_ink(strength):
    """Hysteresis threshold relative to the image's strongest ink, kept above the paper's noise level."""
    h, w = strength.shape
    m = int(BORDER * min(h, w))
    inner = strength[m:-m, m:-m]  # ignore the border, where the dark frame is
    strongest = np.percentile(inner, 99.9)
    noise = np.percentile(inner, 90)  # 90% of pixels are paper, so this is the paper's noise level
    high = max(HIGH_RATIO * strongest, 4 * noise, 0.04)
    low = max(LOW_RATIO * strongest, 2.5 * noise)
    return apply_hysteresis_threshold(strength, low, high).astype(np.uint8)


def remove_frame(mask, band=0.08, span_band=0.12, reach=0.06, tolerance=0.03):
    """Erase the box frame (developed in notebooks/square/01_square_preprocessing.ipynb).

    Per image side: if a frame line spanning (almost) the whole image is found, only straight lines at its position
    are erased; straight lines further inside belong to the drawing (e.g. the sides of a large square). Otherwise
    every long, nearly horizontal/vertical line in the outer band is erased (the original Circle rule).
    Note: the Circle masks in Data/processed/Circle were made with the original rule (9 of 572 masks differ slightly).
    """
    h, w = mask.shape
    side = min(h, w)
    lines = cv2.HoughLinesP(mask * 255, rho=1, theta=np.pi / 360, threshold=100,
                            minLineLength=int(0.35 * side), maxLineGap=int(0.02 * side))
    frame = np.zeros_like(mask)
    if lines is None:
        return mask, frame

    # Candidate frame lines per image side: (distance from that image edge, spans the image?, line)
    candidates = {"top": [], "bottom": [], "left": [], "right": []}
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if min(angle, 180 - angle) < 8:  # nearly horizontal
            spans = min(x1, x2) < reach * w and max(x1, x2) > (1 - reach) * w
            for name, distance in (("top", max(y1, y2)), ("bottom", h - min(y1, y2))):
                if distance < (span_band if spans else band) * h:
                    candidates[name].append((distance, spans, (x1, y1, x2, y2)))
        elif abs(angle - 90) < 8:  # nearly vertical
            spans = min(y1, y2) < reach * h and max(y1, y2) > (1 - reach) * h
            for name, distance in (("left", max(x1, x2)), ("right", w - min(x1, x2))):
                if distance < (span_band if spans else band) * w:
                    candidates[name].append((distance, spans, (x1, y1, x2, y2)))

    for found in candidates.values():
        spanning = [distance for distance, spans, _ in found if spans]
        limit = max(spanning) + tolerance * side if spanning else np.inf  # no spanning frame: erase all candidates
        for distance, _, (x1, y1, x2, y2) in found:
            if distance <= limit:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=int(0.012 * side) + 3)
    return mask * (1 - frame), frame


def remove_border_marks(mask, band=0.05):
    """Remove ink pieces lying (almost) entirely in the outer border band."""
    h, w = mask.shape
    b = int(band * min(h, w))
    in_band = np.ones_like(mask)
    in_band[b:-b, b:-b] = 0
    cleaned = mask.copy()
    n_pieces, labels = cv2.connectedComponents(mask)
    for i in range(1, n_pieces):
        piece = labels == i
        if in_band[piece].mean() > 0.9:
            cleaned[piece] = 0
    return cleaned


def remove_stray_marks(mask):
    """Keep the drawing; drop specks and small pieces far from it (e.g. the examiner's notes)."""
    n_pieces, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_pieces <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    main = 1 + np.argmax(areas[1:])  # label 0 is the background
    distance_to_main = cv2.distanceTransform((labels != main).astype(np.uint8), cv2.DIST_L2, 5)
    side = min(mask.shape)

    keep = np.zeros(n_pieces, bool)
    keep[main] = True
    for i in range(1, n_pieces):
        if i == main or areas[i] < 0.01 * areas[main]:  # specks (<1% of the drawing) are always dropped
            continue
        large = areas[i] >= 0.25 * areas[main]
        close = distance_to_main[labels == i].min() <= 0.04 * side
        keep[i] = large or close
    return keep[labels].astype(np.uint8)


def seal_pores(mask, radius=3):
    """Small closing: seals pores and hollow thick strokes (bridges gaps of ~6 px only)."""
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk)


def extract_ink(image, frame_removal=True):
    """Return (ink mask at the original resolution, whether the colour-only fallback was used).

    frame_removal=False skips the removal of the box frame (used by the Square's safety fallback,
    src/square_preprocessing.py); the default behaviour is unchanged for all shapes.
    """
    image = fill_white_patches(image)
    find_frame = remove_frame if frame_removal else (lambda m: (m, np.zeros_like(m)))
    mask, _ = find_frame(threshold_ink(ink_strength(image)))

    used_fallback = not (MIN_INK_FRACTION <= mask.mean() <= MAX_INK_FRACTION)
    if used_fallback:
        mask, _ = find_frame(threshold_ink(ink_strength(image, colour_only=True)))

    mask = remove_stray_marks(remove_border_marks(mask))
    mask = seal_pores(mask)
    return mask, used_fallback


def pad_to_square(image):
    """Place an image in the centre of a square of zeros (keeps the drawing's true proportions)."""
    h, w = image.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side) + image.shape[2:], image.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = image
    return square


def normalise_size(mask, size=OUTPUT_SIZE):
    """Pad to a square and resize; a pixel is ink if at least 25% of it was ink."""
    small = cv2.resize(pad_to_square(mask).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (small >= 0.25).astype(np.uint8)
