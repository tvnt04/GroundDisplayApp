"""
space_studio_intro.py
~~~~~~~~~~~~~~~~~~~~~~
Full-screen animated splash screen for a space-related application.

Look & feel
-----------
Black background with a subtle twinkling starfield, a plain white X mark
during the zoom-in and pause beats, a soft blurred glow that only appears
once the wordmark starts fading in beside it, an orbital accent ring, and
a "warp flash" finish (radiating light streaks + white-out) instead of a
flat fade to white. The wordmark fades in place -- no sliding.

Animation timeline (5 seconds total)
-------------------------------------
0.00 - 0.36  Oversized X zooms in and spins to centre
0.36 - 0.42  X rests at final size (pause beat)
0.42 - 0.60  Wordmark fades into view (no slide), glow ramps in with it
0.60 - 0.74  Complete wordmark holds, orbit ring rotates slowly
0.74 - 1.00  X exits the same way it entered, reversed: grows oversized
             and drifts back up and away (no spin), finishing with warp
             streaks + white flash

Customize your brand name via _BRAND_LEFT / _BRAND_GLYPH / _BRAND_RIGHT
below -- everything else scales automatically.
"""

from __future__ import annotations

import math
import random
import sys

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, pyqtProperty
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsBlurEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QWidget,
)


# ---------------------------------------------------------------------------
# Brand text -- edit these three to rebrand the splash screen
# ---------------------------------------------------------------------------
_BRAND_LEFT      = "Display"
_BRAND_GLYPH     = "X"
_BRAND_RIGHT     = "Studio"

# ---------------------------------------------------------------------------
# Timing boundaries (normalised 0..1 over the 5-second animation)
# ---------------------------------------------------------------------------
_T_ZOOM_END      = 0.36
_T_PAUSE_END     = 0.42
_T_FADE_END      = 0.60
_T_HOLD_END      = 0.74

_DURATION_MS     = 5_000

# ---------------------------------------------------------------------------
# Font presets -- change _FONT_PRESET to switch the wordmark's look.
# Each entry is (family fallback list, QFont.Weight, letter-spacing px).
# ---------------------------------------------------------------------------
_FONT_PRESETS = {
    # Squared-off, technical sci-fi look (needs Orbitron installed for the
    # full effect; falls back gracefully if it isn't).
    "orbitron":        (["Orbitron", "Eurostile", "Century Gothic", "Segoe UI"], QFont.DemiBold, 3.0),
    # Clean geometric sans, feels like a modern space-agency wordmark.
    "geometric_sans":  (["Century Gothic", "Futura", "Segoe UI Semibold", "Arial"], QFont.Bold, 2.0),
    # Thin, refined, premium-tech feel (Apple-keynote adjacent).
    "elegant_thin":    (["Helvetica Neue Light", "Segoe UI Light", "Arial"], QFont.Thin, 4.0),
    # Monospaced, mission-control / terminal read-out feel.
    "technical_mono":  (["JetBrains Mono", "Consolas", "Courier New"], QFont.Medium, 1.5),
    # Tall, confident condensed display face.
    "condensed_bold":  (["Bahnschrift", "Segoe UI Semibold", "Arial Narrow"], QFont.Bold, 1.0),
}
_FONT_PRESET     = "orbitron"   # <- try: geometric_sans / elegant_thin / technical_mono / condensed_bold

_FONT_SCALE      = 0.034   # fraction of min(width, height)
_FONT_SIZE_MIN   = 20      # px

_X_HEIGHT_RATIO  = 0.95    # X size relative to text cap-height
_GAP_RATIO       = 0.75    # spacing around X relative to text height
_STROKE_RATIO    = 0.13    # pen width relative to X size
_STROKE_MIN      = 2       # px
_GLOW_RADIUS_RATIO = 0.55  # blur radius relative to X size

_ENTRY_Y_OFFSET  = 0.08    # initial Y offset (fraction of height) for zoom entry

_WHITE_START     = 0.55    # within phase-5 when the white overlay begins
_STREAK_START    = 0.15    # within phase-5 when warp streaks begin

_STAR_COUNT      = 220
_STAR_SEED       = 42

# ---------------------------------------------------------------------------
# Palette -- strictly black + white, with one soft ice-blue reserved only
# for the glow falloff (set _C_GLOW = _C_WHITE if you want pure monochrome).
# ---------------------------------------------------------------------------
_C_BLACK         = QColor(0, 0, 0)
_C_BG_BOTTOM     = QColor(6, 6, 8)     # near-black, gives the sky faint depth
_C_WHITE         = QColor(255, 255, 255)
_C_TEXT          = QColor(235, 235, 235)
_C_GLOW          = QColor(210, 225, 255)   # the one permitted accent, glow-only


class SpaceStudioIntro(QWidget):
    """Frameless, full-screen space-themed intro animation widget."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setStyleSheet("background-color: black;")

        self._progress: float = 0.0
        self._stars: list[tuple[float, float, float, float, float]] = []
        self._stars_size: tuple[int, int] | None = None

        # Cache for the blurred X glow -- regenerated only when size/rotation
        # change meaningfully, since a Gaussian blur pass isn't free per-frame.
        self._glow_cache_key: tuple | None = None
        self._glow_cache_img: QImage | None = None

        self._anim = QPropertyAnimation(self, b"progress")
        self._anim.setDuration(_DURATION_MS)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Linear)  # phases handle their own easing
        self._anim.valueChanged.connect(self.update)
        self._anim.finished.connect(self._on_finished)

    # ------------------------------------------------------------------
    # Qt property -- required for QPropertyAnimation
    # ------------------------------------------------------------------

    def _get_progress(self) -> float:
        return self._progress

    def _set_progress(self, value: float) -> None:
        self._progress = value
        self.update()

    progress = pyqtProperty(float, fget=_get_progress, fset=_set_progress)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Resize to the primary screen, show, and begin the animation."""
        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())

        self.show()
        self.raise_()
        self.activateWindow()
        self._anim.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_finished(self) -> None:
        QTimer.singleShot(150, QApplication.quit)

    def _brand_font(self) -> QFont:
        """Return the scaled brand font for the current widget size."""
        families, weight, letter_spacing = _FONT_PRESETS[_FONT_PRESET]
        size = max(_FONT_SIZE_MIN, int(min(self.width(), self.height()) * _FONT_SCALE))
        font = QFont()
        if hasattr(font, "setFamilies"):
            font.setFamilies(families)
        else:  # pragma: no cover - older Qt fallback
            font.setFamily(families[0])
        font.setPixelSize(size)
        font.setWeight(weight)
        font.setLetterSpacing(QFont.AbsoluteSpacing, letter_spacing)
        return font

    def _final_x_size(self) -> float:
        """X glyph size in pixels, matched to the brand-font cap-height."""
        return QFontMetrics(self._brand_font()).height() * _X_HEIGHT_RATIO

    def _ensure_stars(self, w: int, h: int) -> None:
        if self._stars_size == (w, h) and self._stars:
            return
        rng = random.Random(_STAR_SEED)
        self._stars = [
            (
                rng.uniform(0, w),                     # x
                rng.uniform(0, h),                      # y
                rng.uniform(0.5, 1.8),                   # radius
                rng.uniform(0, 2 * math.pi),             # twinkle phase
                rng.uniform(0.6, 1.6),                   # twinkle speed
            )
            for _ in range(_STAR_COUNT)
        ]
        self._stars_size = (w, h)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        p = self._progress
        final_x = self._final_x_size()

        self._draw_background(painter, w, h, p)

        # --- Phase 1: oversized X zooms in and spins (0.00 - _T_ZOOM_END) ---
        if p < _T_ZOOM_END:
            t        = p / _T_ZOOM_END
            zoom_t   = _ease_out_quart(t)
            huge     = max(w, h) * 5.0
            size     = huge - (huge - final_x) * zoom_t
            rotation = 180.0 * (1.0 - _ease_out_cubic(t))
            start_y  = cy - h * _ENTRY_Y_OFFSET
            y        = start_y + (cy - start_y) * zoom_t
            self._draw_x(painter, cx, y, size, rotation, 1.0, glow_opacity=0.0)
            return

        # --- Phase 2: X rests at centre (_T_ZOOM_END - _T_PAUSE_END) ---
        if p < _T_PAUSE_END:
            self._draw_x(painter, cx, cy, final_x, 0.0, 1.0, glow_opacity=0.0)
            return

        # --- Phase 3: wordmark fades in (_T_PAUSE_END - _T_FADE_END) ---
        # The glow ramps in together with the wordmark, not before.
        if p < _T_FADE_END:
            t = _ease_in_out_cubic((p - _T_PAUSE_END) / (_T_FADE_END - _T_PAUSE_END))
            self._draw_orbit_ring(painter, cx, cy, final_x, t, p)
            self._draw_x(painter, cx, cy, final_x, 0.0, 1.0, glow_opacity=t)
            self._draw_brand(painter, cx, cy, final_x, t)
            return

        # --- Phase 4: wordmark holds (_T_FADE_END - _T_HOLD_END) ---
        if p < _T_HOLD_END:
            self._draw_orbit_ring(painter, cx, cy, final_x, 1.0, p)
            self._draw_x(painter, cx, cy, final_x, 0.0, 1.0, glow_opacity=1.0)
            self._draw_brand(painter, cx, cy, final_x, 1.0)
            return

        # --- Phase 5: X exits the same way it entered, reversed (_T_HOLD_END - 1.00) ---
        # Mirrors phase 1's grow + spin + drift. Rotation uses the same
        # ease_out_cubic curve the entry uses (not an accelerating one), so
        # the spin feels identical in character, not exaggerated.
        expand_t = _ease_in_cubic((p - _T_HOLD_END) / (1.0 - _T_HOLD_END))
        zoom_t   = _ease_in_quart(expand_t)
        huge     = max(w, h) * 5.0
        size     = final_x + (huge - final_x) * zoom_t
        rotation = 180.0 * _ease_out_cubic(expand_t)
        start_y  = cy - h * _ENTRY_Y_OFFSET
        y        = cy + (start_y - cy) * zoom_t

        orbit_fade = max(0.0, 1.0 - expand_t * 3.0)
        if orbit_fade > 0:
            self._draw_orbit_ring(painter, cx, cy, final_x, orbit_fade, p)

        text_opacity = max(0.0, 1.0 - expand_t * 2.2)
        if text_opacity > 0:
            self._draw_brand(painter, cx, cy, final_x, text_opacity)

        if expand_t > _STREAK_START:
            streak_t = min(1.0, (expand_t - _STREAK_START) / (1.0 - _STREAK_START))
            self._draw_warp_streaks(painter, cx, y, w, h, streak_t)

        # Glow fades out together with the wordmark as the X pulls away.
        self._draw_x(painter, cx, y, size, rotation, 1.0, glow_opacity=text_opacity)

        if expand_t > _WHITE_START:
            white_t = _ease_in_cubic((expand_t - _WHITE_START) / (1.0 - _WHITE_START))
            painter.fillRect(self.rect(), QColor(255, 255, 255, int(255 * white_t)))

    # ------------------------------------------------------------------
    # Background: gradient sky + nebula glow + starfield
    # ------------------------------------------------------------------

    def _draw_background(self, painter: QPainter, w: int, h: int, p: float) -> None:
        # Strictly black -- a faint radial lift toward the centre gives the
        # sky a touch of depth without introducing another hue.
        grad = QRadialGradient(w / 2, h * 0.4, max(w, h) * 0.9)
        grad.setColorAt(0.0, _C_BG_BOTTOM)
        grad.setColorAt(1.0, _C_BLACK)
        painter.fillRect(0, 0, w, h, grad)

        self._ensure_stars(w, h)
        t = p * _DURATION_MS / 1000.0
        painter.save()
        for sx, sy, r, phase, speed in self._stars:
            twinkle = 0.55 + 0.45 * math.sin(t * speed + phase)
            alpha = max(30, min(235, int(180 * twinkle)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, alpha))
            painter.drawEllipse(QRectF(sx - r, sy - r, r * 2, r * 2))
        painter.restore()

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _render_glow_source(self, size: float, core_width: float, canvas: int) -> QImage:
        """Render a crisp, unblurred X onto a transparent canvas."""
        img = QImage(canvas, canvas, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        gp = QPainter(img)
        gp.setRenderHint(QPainter.Antialiasing)
        gp.translate(canvas / 2, canvas / 2)
        half = size / 2
        # Slightly thicker than the final core so the blurred result reads
        # as a soft halo rather than disappearing at low opacity.
        pen = QPen(_C_WHITE, core_width * 1.15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        gp.setPen(pen)
        gp.drawLine(int(-half), int(-half), int(half), int(half))
        gp.drawLine(int(half), int(-half), int(-half), int(half))
        gp.end()
        return img

    def _blurred(self, img: QImage, radius: float) -> QImage:
        """Apply a true Gaussian blur via QGraphicsBlurEffect for a smooth,
        continuous falloff (avoids the banded/ringed look of stacked strokes)."""
        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        effect = QGraphicsBlurEffect()
        effect.setBlurRadius(radius)
        effect.setBlurHints(QGraphicsBlurEffect.QualityHint)
        item.setGraphicsEffect(effect)
        scene.addItem(item)

        result = QImage(img.size(), QImage.Format_ARGB32_Premultiplied)
        result.fill(Qt.transparent)
        rp = QPainter(result)
        rp.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, img.width(), img.height())
        scene.render(rp, rect, rect)
        rp.end()
        return result

    def _glow_image(self, size: float, core_width: float) -> QImage:
        """Cached blurred glow, generated at a small reference size and
        scaled cheaply afterward -- keeps memory/CPU bounded regardless of
        how large the X is drawn on screen (important during the zoom and
        expand phases, where the glyph can be several times the screen
        size)."""
        blur_radius = max(6.0, size * _GLOW_RADIUS_RATIO)
        canvas = int(size * 1.8 + blur_radius * 4)
        key = (round(size), round(core_width))
        if self._glow_cache_key == key and self._glow_cache_img is not None:
            return self._glow_cache_img

        source = self._render_glow_source(size, core_width, canvas)
        blurred = self._blurred(source, blur_radius)
        self._glow_cache_key = key
        self._glow_cache_img = blurred
        return blurred

    def _draw_x(
        self,
        painter: QPainter,
        x: float,
        y: float,
        size: float,
        rotation: float,
        opacity: float,
        glow_opacity: float = 0.0,
    ) -> None:
        """Draw the X glyph centred on (x, y).

        The core line is always drawn crisp. The soft blurred glow is
        separate and only shows when `glow_opacity` > 0 -- by design this
        stays at 0 while the X is alone (zoom-in, resting pause) and only
        ramps up once the wordmark beside it starts appearing.

        The glow is rendered once at a small reference size (the resting
        logo size) and then stretched with a cheap painter transform, so
        this stays fast even when `size` is huge (zoom-in / expand phases)."""
        if glow_opacity > 0:
            reference_size = self._final_x_size()
            reference_core = max(_STROKE_MIN, reference_size * _STROKE_RATIO)
            scale = size / reference_size if reference_size > 0 else 1.0

            glow_img = self._glow_image(reference_size, reference_core)
            painter.save()
            painter.translate(x, y)
            painter.rotate(rotation)
            painter.scale(scale, scale)
            painter.setOpacity(glow_opacity)
            if _C_GLOW != _C_WHITE:
                tinted = QImage(glow_img)
                tint = QPainter(tinted)
                tint.setCompositionMode(QPainter.CompositionMode_SourceIn)
                tint.fillRect(tinted.rect(), _C_GLOW)
                tint.end()
                painter.drawImage(int(-tinted.width() / 2), int(-tinted.height() / 2), tinted)
            else:
                painter.drawImage(int(-glow_img.width() / 2), int(-glow_img.height() / 2), glow_img)
            painter.restore()

        # --- Crisp core line on top, drawn fresh so it stays sharp at any size ---
        painter.save()
        painter.translate(x, y)
        painter.rotate(rotation)
        core_color = QColor(_C_WHITE)
        core_color.setAlpha(int(255 * opacity))
        core_width = max(_STROKE_MIN, size * _STROKE_RATIO)
        painter.setPen(
            QPen(core_color, core_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        half = size / 2
        painter.drawLine(int(-half), int(-half), int(half), int(half))
        painter.drawLine(int(half), int(-half), int(-half), int(half))
        painter.restore()

    def _draw_orbit_ring(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        x_size: float,
        opacity: float,
        p: float,
    ) -> None:
        """A slim rotating ellipse orbiting the X mark, sci-fi accent."""
        if opacity <= 0:
            return
        painter.save()
        painter.translate(cx, cy)
        painter.rotate((p * 360.0 * 0.6) % 360.0)

        radius_x = x_size * 1.35
        radius_y = x_size * 0.55

        color = QColor(_C_WHITE)
        color.setAlpha(int(110 * opacity))
        pen = QPen(color, max(1.0, x_size * 0.012))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(-radius_x, -radius_y, radius_x * 2, radius_y * 2))

        # A small "satellite" dot travelling along the ring
        dot_angle = (p * 360.0 * 1.4) % 360.0
        rad = math.radians(dot_angle)
        dot_x = radius_x * math.cos(rad)
        dot_y = radius_y * math.sin(rad)
        dot_color = QColor(_C_WHITE)
        dot_color.setAlpha(int(220 * opacity))
        painter.setPen(Qt.NoPen)
        painter.setBrush(dot_color)
        dot_r = max(1.5, x_size * 0.035)
        painter.drawEllipse(QRectF(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2))

        painter.restore()

    def _draw_warp_streaks(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        w: int,
        h: int,
        t: float,
    ) -> None:
        """Radiating light streaks bursting outward, like a hyperspace jump."""
        painter.save()
        rng = random.Random(7)
        max_len = math.hypot(w, h) * 0.6 * t
        count = 26
        for i in range(count):
            angle = (2 * math.pi / count) * i + rng.uniform(-0.06, 0.06)
            length = max_len * rng.uniform(0.6, 1.0)
            inner = length * 0.25
            x1 = cx + math.cos(angle) * inner
            y1 = cy + math.sin(angle) * inner
            x2 = cx + math.cos(angle) * length
            y2 = cy + math.sin(angle) * length
            alpha = int(150 * t * rng.uniform(0.5, 1.0))
            color = QColor(_C_WHITE)
            color.setAlpha(alpha)
            pen = QPen(color, rng.uniform(1.2, 2.6), Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.restore()

    def _draw_brand(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        x_size: float,
        opacity: float,
    ) -> None:
        """Draw the full wordmark ("Display" + X + "Studio") at the given opacity."""
        painter.save()

        font    = self._brand_font()
        metrics = QFontMetrics(font)
        painter.setFont(font)

        left_w  = metrics.horizontalAdvance(_BRAND_LEFT)
        right_w = metrics.horizontalAdvance(_BRAND_RIGHT)
        text_h  = metrics.height()

        logo_size   = text_h * _X_HEIGHT_RATIO
        gap         = text_h * _GAP_RATIO
        total_width = left_w + gap + logo_size + gap + right_w

        left     = cx - total_width / 2
        baseline = cy - (metrics.ascent() + metrics.descent()) / 2 + metrics.ascent()
        alpha    = int(255 * opacity)

        # Subtle dark drop shadow for depth -- stays within the black/white
        # palette instead of adding a colored glow behind the text.
        shadow_color = QColor(0, 0, 0, int(alpha * 0.6))
        painter.setPen(shadow_color)

        painter.drawText(int(left) + 1, int(baseline) + 2, _BRAND_LEFT)
        studio_x = left + left_w + gap + logo_size + gap
        painter.drawText(int(studio_x) + 1, int(baseline) + 2, _BRAND_RIGHT)

        painter.setPen(QColor(_C_TEXT.red(), _C_TEXT.green(), _C_TEXT.blue(), alpha))
        painter.drawText(int(left), int(baseline), _BRAND_LEFT)
        painter.drawText(int(studio_x), int(baseline), _BRAND_RIGHT)

        painter.restore()


# ---------------------------------------------------------------------------
# Easing functions
# ---------------------------------------------------------------------------

def _ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def _ease_out_quart(t: float) -> float:
    return 1 - (1 - t) ** 4


def _ease_in_quart(t: float) -> float:
    return t ** 4


def _ease_in_cubic(t: float) -> float:
    return t ** 3


def _ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app   = QApplication(sys.argv)
    intro = SpaceStudioIntro()
    intro.start()
    sys.exit(app.exec_())