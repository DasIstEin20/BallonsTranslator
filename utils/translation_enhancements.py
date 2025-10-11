from __future__ import annotations

import copy
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from qtpy.QtGui import QFont, QFontMetricsF

from utils.config import pcfg
from utils import shared
from utils.io_utils import text_is_empty
from utils.logger import logger as LOGGER
from utils.textblock import TextBlock, sort_regions


def perform_smart_bubble_split(
    textblocks: Sequence[TextBlock],
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> Tuple[bool, List[TextBlock]]:
    """Split textblocks with multiple columns into smaller regions.

    Returns a tuple ``(changed, blocks)``.
    """

    if not textblocks:
        return False, list(textblocks)

    changed = False
    result: List[TextBlock] = []
    for blk in textblocks:
        segments = _split_block(blk, mask)
        if len(segments) > 1:
            changed = True
            result.extend(segments)
        else:
            result.append(segments[0])

    if changed:
        result = sort_regions(result)

    return changed, result


def _split_block(block: TextBlock, mask: np.ndarray | None) -> List[TextBlock]:
    by_lines = _split_block_by_lines(block)
    if by_lines:
        LOGGER.debug(
            '[SmartBubbleSplit] Split block %s into %d parts via line clustering.',
            block.xyxy,
            len(by_lines),
        )
        return by_lines

    by_mask = _split_block_by_mask(block, mask)
    if by_mask:
        LOGGER.debug(
            '[SmartBubbleSplit] Split block %s into %d parts via mask components.',
            block.xyxy,
            len(by_mask),
        )
        return by_mask

    return [block]


def _split_block_by_lines(block: TextBlock) -> List[TextBlock] | None:
    if len(block.lines) <= 1:
        return None

    lines = block.lines_array(dtype=np.float32)
    centers = lines.mean(axis=1)
    axis = 1 if block.vertical else 0
    span = max(
        1.0,
        (block.xyxy[3] - block.xyxy[1]) if block.vertical else (block.xyxy[2] - block.xyxy[0]),
    )
    order = np.argsort(centers[:, axis])

    gap_threshold = max(span * 0.35, 24.0)
    groups: List[List[int]] = [[int(order[0])]]
    for idx in order[1:]:
        prev_idx = groups[-1][-1]
        gap = abs(float(centers[int(idx), axis] - centers[int(prev_idx), axis]))
        if gap > gap_threshold:
            groups.append([int(idx)])
        else:
            groups[-1].append(int(idx))

    if len(groups) <= 1:
        return None

    clones = [_clone_block(block, line_indices=group) for group in groups]
    return clones


def _split_block_by_mask(block: TextBlock, mask: np.ndarray | None) -> List[TextBlock] | None:
    if mask is None:
        return None

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in block.xyxy]
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    sub_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    if sub_mask.size == 0:
        return None

    binary = (sub_mask > 0).astype(np.uint8)
    if binary.sum() < 16:
        return None

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(binary)

    boxes: List[Tuple[int, int, int, int]] = []
    min_area = max(16, int(binary.shape[0] * binary.shape[1] * 0.02))
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        if xs.size < min_area:
            continue
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    if len(boxes) <= 1:
        return None

    if block.vertical:
        boxes.sort(key=lambda b: b[1])
        span = max(1, block.xyxy[3] - block.xyxy[1])
    else:
        boxes.sort(key=lambda b: b[0])
        span = max(1, block.xyxy[2] - block.xyxy[0])

    gap_threshold = span * 0.1
    separated = False
    previous_edge = None
    for box in boxes:
        start = box[1] if block.vertical else box[0]
        end = box[3] if block.vertical else box[2]
        if previous_edge is not None and start - previous_edge > gap_threshold:
            separated = True
            break
        previous_edge = end

    if not separated:
        return None

    clones = []
    for box in boxes:
        rel_box = (
            box[0] + crop_x1 - x1,
            box[1] + crop_y1 - y1,
            box[2] + crop_x1 - x1,
            box[3] + crop_y1 - y1,
        )
        clones.append(_clone_block(block, bbox=rel_box))
    return clones


def _clone_block(block: TextBlock, line_indices: Iterable[int] | None = None, bbox: Tuple[int, int, int, int] | None = None) -> TextBlock:
    new_block = copy.deepcopy(block)
    new_block.translation = ''
    new_block.rich_text = ''
    new_block.region_mask = None
    new_block.region_inpaint_dict = None

    if line_indices is not None:
        indices = list(line_indices)
        new_block.lines = [copy.deepcopy(block.lines[idx]) for idx in indices]
        if block.text:
            new_block.text = [block.text[idx] if idx < len(block.text) else '' for idx in indices]
        if block.distance is not None:
            if isinstance(block.distance, np.ndarray):
                new_block.distance = block.distance[indices]
            else:
                new_block.distance = [block.distance[idx] for idx in indices]
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, x2 - 1))
        y1 = max(0, min(y1, y2 - 1))
        abs_x1 = int(round(block.xyxy[0] + x1))
        abs_y1 = int(round(block.xyxy[1] + y1))
        abs_x2 = int(round(block.xyxy[0] + x2))
        abs_y2 = int(round(block.xyxy[1] + y2))
        rect = [[abs_x1, abs_y1], [abs_x2, abs_y1], [abs_x2, abs_y2], [abs_x1, abs_y2]]
        new_block.lines = [rect]
        new_block.text = []
        new_block.distance = None

    new_block.adjust_bbox(with_bbox=False)
    return new_block


def auto_format_fit_hook(
    translations: List[str] | None = None,
    textblocks: Sequence[TextBlock] | None = None,
    translator=None,
    **_: object,
) -> None:
    if not translations or not textblocks or not getattr(pcfg, 'auto_format_fit', False):
        return

    tolerance = float(getattr(pcfg, 'fit_tolerance', 0.9) or 0.9)
    min_font = int(getattr(pcfg, 'min_font_size', 12) or 12)
    max_font = int(getattr(pcfg, 'max_font_size', 28) or 28)

    for idx, (blk, text) in enumerate(zip(textblocks, translations)):
        formatted = _auto_format_block(blk, text, tolerance, min_font, max_font)
        if formatted is None:
            continue
        formatted_text, font_size, spacing, shrink = formatted
        translations[idx] = formatted_text
        blk.translation = formatted_text
        blk.font_size = font_size
        blk.line_spacing = spacing
        LOGGER.debug(
            '[AutoFormatFit] bbox=%s size=%.2f spacing=%.2f lines=%d shrink=%s',
            blk.xyxy,
            font_size,
            spacing,
            formatted_text.count('\n') + 1,
            shrink,
        )


def _auto_format_block(
    block: TextBlock,
    text: str,
    tolerance: float,
    min_font: int,
    max_font: int,
) -> Tuple[str, float, float, bool] | None:
    if text_is_empty(text):
        return None
    if getattr(block, 'vertical', False):
        return None

    width = max(1.0, float(block.xyxy[2] - block.xyxy[0]))
    height = max(1.0, float(block.xyxy[3] - block.xyxy[1]))

    font_family = block.font_family or shared.DEFAULT_FONT_FAMILY
    base_size = block.font_size if block.font_size and block.font_size > 0 else block._detected_font_size
    if not base_size or base_size <= 0:
        base_size = max_font
    base_size = float(min(max(base_size, min_font), max_font))

    formatted = _try_layout_within_bounds(text, font_family, base_size, width, height, tolerance, min_font, max_font)
    if formatted is None:
        return None

    lines, metrics, font_size, shrink_applied = formatted
    formatted_text = '\n'.join(lines)
    total_height = metrics.lineSpacing() * max(len(lines), 1)
    spacing_ratio = (height * tolerance) / max(total_height, 1.0)
    base_spacing = getattr(block, 'line_spacing', 1.0)
    spacing = min(base_spacing, spacing_ratio)
    spacing = float(max(0.8, min(spacing, 1.5)))

    return formatted_text, font_size, spacing, shrink_applied


def _try_layout_within_bounds(
    text: str,
    font_family: str,
    base_size: float,
    width: float,
    height: float,
    tolerance: float,
    min_font: int,
    max_font: int,
) -> Tuple[List[str], QFontMetricsF, float, bool] | None:
    best_result: Tuple[List[str], QFontMetricsF, float, bool] | None = None

    for size in range(int(round(base_size)), int(max_font) + 1):
        lines, metrics = _wrap_text_for_width(text, font_family, size, width)
        if _fits_bounds(lines, metrics, width, height, tolerance):
            return lines, metrics, float(size), False
        if best_result is None or size == int(round(base_size)):
            best_result = (lines, metrics, float(size), False)

    for size in range(int(round(base_size)) - 1, min_font - 1, -1):
        lines, metrics = _wrap_text_for_width(text, font_family, size, width)
        if _fits_bounds(lines, metrics, width, height, tolerance):
            return lines, metrics, float(size), False
        if best_result is None:
            best_result = (lines, metrics, float(size), False)

    if best_result is None:
        return None

    lines, metrics, size, _ = best_result
    max_line_width = max((metrics.horizontalAdvance(line) for line in lines), default=0.0)
    total_height = metrics.lineSpacing() * max(len(lines), 1)
    if max_line_width <= 0 or total_height <= 0:
        return lines, metrics, size, True

    ratio_w = width / max_line_width
    ratio_h = height / total_height
    shrink_ratio = min(ratio_w, ratio_h) * tolerance
    adjusted_size = max(6.0, float(size) * shrink_ratio)
    adjusted_size = min(adjusted_size, float(max_font))

    lines, metrics = _wrap_text_for_width(text, font_family, adjusted_size, width)
    return lines, metrics, adjusted_size, True


def _wrap_text_for_width(text: str, font_family: str, font_size: float, width: float) -> Tuple[List[str], QFontMetricsF]:
    font = QFont(font_family)
    font.setPointSizeF(font_size)
    metrics = QFontMetricsF(font)
    lines: List[str] = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.rstrip()
        if not paragraph:
            lines.append('')
            continue
        wrapped = _wrap_paragraph(paragraph, metrics, width)
        lines.extend(wrapped)
    return lines, metrics


def _wrap_paragraph(paragraph: str, metrics: QFontMetricsF, width: float) -> List[str]:
    words = paragraph.split()
    if not words:
        return ['']

    line = words[0]
    lines = []
    for word in words[1:]:
        candidate = f'{line} {word}' if line else word
        if metrics.horizontalAdvance(candidate) <= width:
            line = candidate
            continue
        lines.append(line)
        segments = _split_word(word, metrics, width)
        line = segments.pop() if segments else ''
        lines.extend(segments)

    if metrics.horizontalAdvance(line) > width:
        segments = _split_word(line, metrics, width)
        if segments:
            lines.extend(segments[:-1])
            line = segments[-1]

    if line:
        lines.append(line)
    return lines


def _split_word(word: str, metrics: QFontMetricsF, width: float) -> List[str]:
    if metrics.horizontalAdvance(word) <= width or width <= 1:
        return [word]

    segments: List[str] = []
    current = ''
    for char in word:
        candidate = current + char
        if metrics.horizontalAdvance(candidate) <= width or not current:
            current = candidate
        else:
            segments.append(current)
            current = char
    if current:
        segments.append(current)
    return segments


def _fits_bounds(lines: List[str], metrics: QFontMetricsF, width: float, height: float, tolerance: float) -> bool:
    max_line_width = max((metrics.horizontalAdvance(line) for line in lines), default=0.0)
    total_height = metrics.lineSpacing() * max(len(lines), 1)
    return max_line_width <= width * tolerance and total_height <= height * tolerance

