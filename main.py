"""
Real-time Hand VFX System with Neon Effects
Webcam application with glowing neon effects around hand movements.

Dependencies: OpenCV, MediaPipe, NumPy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration class for all visual effects parameters"""

    # Visual Effects
    glow_intensity: int = 6          # Number of glow layers (kept small to avoid fat bars)
    line_thickness: int = 2
    smoothing_factor: float = 0.3

    # Performance
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5

    # Effects toggles
    enable_particles: bool = True
    enable_trails: bool = True
    enable_aura: bool = True
    enable_fingertip_pulse: bool = True
    enable_beam: bool = True
    enable_power_orb: bool = True
    enable_grayscale_bg: bool = False
    enable_screen_shake: bool = True
    enable_finger_laser: bool = True
    enable_glitch: bool = True

    # Cinematic VFX
    cinematic_darkening: float = 0.25
    bloom_intensity: float = 0.3
    pulse_speed: float = 3.0

    # Beam effect
    beam_fade_speed: float = 4.0

    # Neon line blend weight (higher = more visible lines)
    beam_blend_alpha: float = 0.85

    # Fingertip Join Effect
    join_effect_distance_threshold: float = 60.0
    join_effect_intensity: float = 1.0
    join_effect_max_size: int = 40
    join_effect_fade_speed: float = 8.0

    # Magic Shield Effect
    enable_shield: bool = True
    shield_radius: int = 120
    shield_rotation_speed: float = 2.0
    shield_glow_intensity: float = 0.6
    shield_color: Tuple[int, int, int] = (0, 140, 255)  # BGR orange

    # Skeleton glow falloff (lower = thinner/softer outer glow)
    skeleton_glow_alpha: float = 0.35

    # Aura enable/intensity
    aura_alpha: float = 0.08

    # Colors (BGR format for OpenCV)
    colors: dict = field(default_factory=lambda: {
        'primary':   (0, 255, 255),      # Cyan
        'secondary': (255, 0, 255),      # Magenta
        'accent':    (255, 255, 0),      # Yellow
        'energy':    (0, 255, 0),        # Green
        'purple':    (128, 0, 255),      # Purple
        'pink':      (255, 105, 180),    # Pink

        # Beam core colors
        'beam_core':      (255, 255, 255),
        'beam_highlight': (200, 220, 255),

        # Per-finger neon colors (BGR)
        'thumb_line':  (20, 80, 255),    # Orange-red
        'index_line':  (255, 255, 0),    # Cyan
        'middle_line': (220, 0, 255),    # Magenta
        'ring_line':   (255, 60, 140),   # Purple-blue
        'pinky_line':  (0, 220, 255),    # Yellow-green
    })

# ============================================================================
# HAND TRACKING
# ============================================================================

class HandTracker:
    """Handles hand detection using MediaPipe"""

    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            model_complexity=1
        )

        # Hand skeleton connections
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm base
        ]

    def detect_hands(self, frame: np.ndarray) -> Optional[List[List[Tuple[int, int]]]]:
        """Detect hands and return pixel landmark positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        h, w = frame.shape[:2]
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))
            all_landmarks.append(landmarks)

        return all_landmarks

# ============================================================================
# NEON RENDERING
# ============================================================================

class NeonRenderer:
    """Handles neon skeleton and aura rendering"""

    def __init__(self, config: Config):
        self.config = config
        self.pulse_time = 0.0
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def update(self, dt: float):
        self.pulse_time += dt

    def draw_neon_line_on_overlay(self, overlay: np.ndarray,
                                   pt1: Tuple[int, int], pt2: Tuple[int, int],
                                   color: Tuple[int, int, int],
                                   base_thickness: int = 2):
        """
        Draw a neon glow line onto an overlay image.
        Uses only colored layers — no black, no dark outlines.
        Outer layers are dim, inner core is bright.
        """
        n = self.config.glow_intensity  # e.g. 6

        for i in range(n, 0, -1):
            # i=n is outermost, i=1 is innermost core
            thickness = base_thickness + (i - 1) * 2
            # Alpha: very faint outer, solid bright inner
            alpha = self.config.skeleton_glow_alpha * (1.0 - (i - 1) / n)
            alpha = max(0.0, min(1.0, alpha))
            if i == 1:
                alpha = 0.9  # Core line: nearly full brightness

            glow_color = tuple(int(c * alpha) for c in color)
            cv2.line(overlay, pt1, pt2, glow_color, thickness, cv2.LINE_AA)

    def draw_hand_skeleton(self, frame: np.ndarray,
                            landmarks: List[Tuple[int, int]],
                            color: Tuple[int, int, int]):
        """
        Draw hand skeleton using overlay blending so lines look glowing,
        not painted. Blends onto frame in-place.
        """
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for c0, c1 in self.connections:
            if c0 < len(landmarks) and c1 < len(landmarks):
                pt1 = (int(landmarks[c0][0]), int(landmarks[c0][1]))
                pt2 = (int(landmarks[c1][0]), int(landmarks[c1][1]))
                self.draw_neon_line_on_overlay(overlay, pt1, pt2, color,
                                               self.config.line_thickness)

        # Blend skeleton overlay onto frame: additive-style blending
        # cv2.add clamps at 255 which gives the glow-on-dark look
        cv2.add(frame, overlay, dst=frame)

    def draw_hand_aura(self, frame: np.ndarray,
                       landmarks: List[Tuple[int, int]],
                       color: Tuple[int, int, int]):
        """
        Draw a soft pulsing aura ring around the palm center.
        Uses overlay blending — no dark blobs.
        """
        if not self.config.enable_aura or len(landmarks) < 21:
            return

        palm_indices = [0, 5, 9, 13, 17]
        palm_x = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
        palm_y = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))

        pulse = math.sin(self.pulse_time * self.config.pulse_speed) * 0.15 + 1.0
        base_radius = int(35 * pulse)

        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw soft rings — only on overlay, blended additively
        num_rings = 5
        for i in range(num_rings):
            radius = base_radius + i * 7
            # Fades out toward outer rings
            alpha = self.config.aura_alpha * (1.0 - i / num_rings)
            ring_color = tuple(int(c * alpha) for c in color)
            cv2.circle(overlay, (palm_x, palm_y), radius, ring_color, 2, cv2.LINE_AA)

        cv2.add(frame, overlay, dst=frame)

# ============================================================================
# BEAM / FINGERTIP CONNECTION EFFECT
# ============================================================================

class BeamEffect:
    """Renders RGB neon finger-to-finger connection lines between two hands"""

    def __init__(self, config: Config):
        self.config = config
        self.beam_alpha = 0.0
        self.target_alpha = 0.0
        self.time = 0.0
        self.hand1_landmarks: Optional[List[Tuple[int, int]]] = None
        self.hand2_landmarks: Optional[List[Tuple[int, int]]] = None

        # Tracking alpha for each finger's join effect
        self.join_alphas = {idx: 0.0 for idx in [4, 8, 12, 16, 20]}

        # Fingertip index → color
        self.finger_colors = {
            4:  config.colors['thumb_line'],
            8:  config.colors['index_line'],
            12: config.colors['middle_line'],
            16: config.colors['ring_line'],
            20: config.colors['pinky_line'],
        }

    def update(self, dt: float, landmarks_list):
        """Update fade state and store landmark data"""
        self.time += dt

        current_hand_count = len(landmarks_list) if landmarks_list else 0
        self.target_alpha = 1.0 if current_hand_count >= 2 else 0.0

        alpha_diff = self.target_alpha - self.beam_alpha
        self.beam_alpha += alpha_diff * dt * self.config.beam_fade_speed
        self.beam_alpha = max(0.0, min(1.0, self.beam_alpha))

        if landmarks_list and len(landmarks_list) >= 2:
            self.hand1_landmarks = landmarks_list[0]
            self.hand2_landmarks = landmarks_list[1]
            
            # Update join effect alphas based on distance
            hand1 = self.hand1_landmarks
            hand2 = self.hand2_landmarks
            for idx in self.join_alphas.keys():
                if idx < len(hand1) and idx < len(hand2):
                    x1, y1 = hand1[idx]
                    x2, y2 = hand2[idx]
                    dist = math.hypot(x2 - x1, y2 - y1)
                    
                    target = 1.0 if dist < self.config.join_effect_distance_threshold else 0.0
                    j_diff = target - self.join_alphas[idx]
                    new_alpha = self.join_alphas[idx] + j_diff * dt * self.config.join_effect_fade_speed
                    self.join_alphas[idx] = max(0.0, min(1.0, new_alpha))
                else:
                    self._fade_out_join_alpha(idx, dt)
        else:
            self.hand1_landmarks = None
            self.hand2_landmarks = None
            for idx in self.join_alphas.keys():
                self._fade_out_join_alpha(idx, dt)

    def _fade_out_join_alpha(self, idx: int, dt: float):
        j_diff = 0.0 - self.join_alphas[idx]
        new_alpha = self.join_alphas[idx] + j_diff * dt * self.config.join_effect_fade_speed
        self.join_alphas[idx] = max(0.0, min(1.0, new_alpha))

    def _draw_neon_line(self, overlay: np.ndarray,
                        pt1: Tuple[int, int], pt2: Tuple[int, int],
                        color: Tuple[int, int, int]):
        """
        Draw a single neon fingertip connection line onto overlay.
        Multiple passes with decreasing thickness for soft glow.
        No black border — only colored layers.
        """
        # Outer soft glow layers - upgraded for richer, more premium spread
        glow_passes = [
            (40, 0.06),   # furthest spread, very soft bloom
            (28, 0.12),   # wide halo
            (18, 0.20),   # mid halo
            (10, 0.40),   # inner bright halo
            (5,  0.75),   # thick core
            (2,  1.00),   # intense center
        ]
        for thickness, alpha_mult in glow_passes:
            layer_color = tuple(int(c * alpha_mult) for c in color)
            cv2.line(overlay, pt1, pt2, layer_color, thickness, cv2.LINE_AA)

    def _draw_join_effect(self, overlay: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int], alpha: float):
        """
        Draw a concentrated energy orb/burst at the contact area.
        Combines a pulsing bright core with a delicate magic cross-flare.
        """
        intensity = self.config.join_effect_intensity * alpha
        base_size = self.config.join_effect_max_size
        
        # Orb pulse dynamic sizing (fast subtle pulsing jitter)
        pulse = math.sin(self.time * 20.0) * 0.1 + 1.0
        size = int(base_size * pulse * alpha)
        
        if size <= 0:
            return
            
        # Energy burst glowing layers
        passes = [
            (size, 0.10, color),                 # faint far glow
            (int(size * 0.65), 0.35, color),     # bright corona
            (int(size * 0.35), 0.70, color),     # intense primary core
            (int(size * 0.15) + 1, 1.0, (255, 255, 255)) # tiny pure white center spark
        ]
        
        for radius, a_mult, c in passes:
            if radius <= 0:
                continue
            effective_a = min(1.0, a_mult * intensity)
            layer_color = tuple(int(ch * effective_a) for ch in c)
            cv2.circle(overlay, center, radius, layer_color, -1, cv2.LINE_AA)
            
        # Draw a sleek magical 4-point star flash (lens flare style)
        star_len = int(size * 1.8)
        if star_len > 0:
            star_color = tuple(int(ch * 0.5 * intensity) for ch in color)
            # Horizontal flare
            cv2.line(overlay, (center[0] - star_len, center[1]), (center[0] + star_len, center[1]), star_color, 2, cv2.LINE_AA)
            cv2.line(overlay, (center[0] - int(star_len*0.5), center[1]), (center[0] + int(star_len*0.5), center[1]), (255,255,255), 1, cv2.LINE_AA)
            # Vertical flare
            cv2.line(overlay, (center[0], center[1] - star_len), (center[0], center[1] + star_len), star_color, 2, cv2.LINE_AA)
            cv2.line(overlay, (center[0], center[1] - int(star_len*0.5)), (center[0], center[1] + int(star_len*0.5)), (255,255,255), 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw fingertip-to-fingertip neon connections and join effects.
        Returns the modified frame (IMPORTANT: caller must use return value).
        """
        if self.beam_alpha <= 0.01:
            return frame
        if self.hand1_landmarks is None or self.hand2_landmarks is None:
            return frame

        h, w = frame.shape[:2]
        
        # Overlay for connection lines
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Dedicated overlay for join effects (for 100% additive sparks)
        join_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        has_join_effect = False

        fingertip_indices = [4, 8, 12, 16, 20]
        hand1 = self.hand1_landmarks
        hand2 = self.hand2_landmarks

        for idx in fingertip_indices:
            if idx < len(hand1) and idx < len(hand2):
                pt1 = (int(hand1[idx][0]), int(hand1[idx][1]))
                pt2 = (int(hand2[idx][0]), int(hand2[idx][1]))
                color = self.finger_colors.get(idx, self.config.colors['beam_core'])
                
                # Render beam neon line
                self._draw_neon_line(overlay, pt1, pt2, color)
                
                # Render join effect if fingers are close
                join_alpha = self.join_alphas.get(idx, 0.0)
                if join_alpha > 0.01:
                    has_join_effect = True
                    mx = int((pt1[0] + pt2[0]) / 2)
                    my = int((pt1[1] + pt2[1]) / 2)
                    self._draw_join_effect(join_overlay, (mx, my), color, join_alpha)

        # Blend connection lines softly
        # beam_blend_alpha controls how visible the lines are
        effective_alpha = self.config.beam_blend_alpha * self.beam_alpha
        cv2.addWeighted(overlay, effective_alpha, frame, 1.0, 0, dst=frame)
        
        # Add join effects directly on top (bright additive blend without muting)
        if has_join_effect:
            cv2.add(frame, join_overlay, dst=frame)

        return frame

# ============================================================================
# MAGIC SHIELD EFFECT
# ============================================================================

class MagicShieldEffect:
    """Dr Strange style magical circular shield effect"""

    def __init__(self, config: Config):
        self.config = config
        self.time = 0.0
        self.current_alpha = 0.0
        self.fade_speed = 8.0
        self.last_center = None

    def update(self, dt: float, landmarks_list):
        self.time += dt
        
        target_alpha = 0.0
        
        # Activate ONLY when exactly one hand is detected
        if landmarks_list and len(landmarks_list) == 1:
            landmarks = landmarks_list[0]
            
            # Simple open palm gesture check:
            # Check if fingertips are further from wrist than PIP joints
            wrist = (landmarks[0][0], landmarks[0][1])
            is_open = True
            for tip_idx, pip_idx in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                tip = (landmarks[tip_idx][0], landmarks[tip_idx][1])
                pip = (landmarks[pip_idx][0], landmarks[pip_idx][1])
                dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
                dist_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
                if dist_tip < dist_pip:
                    is_open = False
                    break
                    
            if is_open:
                target_alpha = 1.0
                # Update last known center safely
                palm_indices = [0, 5, 9, 13, 17]
                cx = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
                cy = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))
                self.last_center = (cx, cy)
                
        alpha_diff = target_alpha - self.current_alpha
        self.current_alpha += alpha_diff * dt * self.fade_speed
        self.current_alpha = max(0.0, min(1.0, self.current_alpha))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enable_shield or self.current_alpha <= 0.01:
            return frame
        if self.last_center is None:
            return frame

        h, w = frame.shape[:2]
        shield_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Mild flicker on glow intensity
        flicker = 1.0 + math.sin(self.time * 30.0) * 0.05
        intensity = self.config.shield_glow_intensity * self.current_alpha * flicker
        
        # Subtle scale pulsing
        pulse = math.sin(self.time * 5.0) * 0.02 + 1.0
        base_radius = int(self.config.shield_radius * pulse)
        
        color = self.config.shield_color
        
        angle_outer = self.time * self.config.shield_rotation_speed
        angle_inner = -self.time * self.config.shield_rotation_speed * 1.5
        
        self._draw_magic_circle(shield_overlay, self.last_center, base_radius, angle_outer, angle_inner, color, intensity)
        
        cv2.add(frame, shield_overlay, dst=frame)
        return frame

    def _draw_magic_circle(self, overlay: np.ndarray, center, radius, angle_outer, angle_inner, color, intensity):
        """Draw circular magic shield elements onto overlay with neon glow technique"""
        # Multi-layer drawing for neon glow
        thick_passes = [
            (24, 0.05),
            (16, 0.15),
            (8,  0.40),
            (3,  0.80),
            (1,  1.0)
        ]
        
        cx, cy = center
        
        for thickness, alpha in thick_passes:
            layer_alpha = alpha * intensity
            if layer_alpha <= 0: continue
            
            # gradient color effect: core is closer to white/bright yellow
            if thickness <= 3:
                r_c = min(255, color[0] + 100)
                g_c = min(255, color[1] + 100)
                b_c = min(255, color[2] + 100)
                c = (r_c, g_c, b_c)
            else:
                c = color
                
            layer_color = tuple(int(ch * layer_alpha) for ch in c)
            
            # 1. Main Outer Ring
            cv2.circle(overlay, center, radius, layer_color, thickness, cv2.LINE_AA)
            cv2.circle(overlay, center, max(1, radius - 15), layer_color, max(1, thickness - 1), cv2.LINE_AA)
            
            # 2. Outer Octagon
            if radius > 15:
                pts_oct = []
                for i in range(8):
                    theta = angle_outer + i * (math.pi / 4)
                    x = int(cx + (radius - 15) * math.cos(theta))
                    y = int(cy + (radius - 15) * math.sin(theta))
                    pts_oct.append((x, y))
                for i in range(8):
                    cv2.line(overlay, pts_oct[i], pts_oct[(i + 1) % 8], layer_color, max(1, thickness-1), cv2.LINE_AA)
            
            # 3. Inner Rotating Squares (giving that layered look)
            inner_r = max(5, radius - 45)
            pts_sq = []
            for i in range(4):
                theta = angle_inner + i * (math.pi / 2)
                x = int(cx + inner_r * math.cos(theta))
                y = int(cy + inner_r * math.sin(theta))
                pts_sq.append((x, y))
            for i in range(4):
                cv2.line(overlay, pts_sq[i], pts_sq[(i + 1) % 4], layer_color, thickness, cv2.LINE_AA)
                
            # 4. Connecting Radial Lines from Inner to Outer
            for i in range(12):
                theta = angle_outer + i * (math.pi / 6)
                x1 = int(cx + inner_r * math.cos(theta))
                y1 = int(cy + inner_r * math.sin(theta))
                x2 = int(cx + max(1, radius - 15) * math.cos(theta))
                y2 = int(cy + max(1, radius - 15) * math.sin(theta))
                cv2.line(overlay, (x1,y1), (x2,y2), layer_color, max(1, thickness-1), cv2.LINE_AA)
                
            # 5. Glowing Core (soft blend orb + small dot)
            core_r = int(25 * (1.0 + 0.1 * math.sin(self.time * 10.0)))
            if thickness > 3:
                cv2.circle(overlay, center, core_r, layer_color, -1, cv2.LINE_AA)
            elif thickness == 1:
                cv2.circle(overlay, center, 8, layer_color, -1, cv2.LINE_AA)
                
            # 6. Outer Runes/Arcs
            # Drawing disjoint arcs via cv2.ellipse
            arc_radius = int(radius + 20)
            axes = (arc_radius, arc_radius)
            for i in range(4):
                theta_mid = angle_inner * 1.5 + i * (math.pi / 2)
                start_rad = theta_mid - 0.2
                end_rad = theta_mid + 0.2
                cv2.ellipse(overlay, center, axes, 0, math.degrees(start_rad), math.degrees(end_rad), layer_color, thickness, cv2.LINE_AA)

# ============================================================================
# PARTICLE SYSTEM
# ============================================================================

class ParticleSystem:
    def __init__(self, config: Config):
        self.config = config
        self.particles = []

    def emit(self, x, y, color, amount=2):
        for _ in range(amount):
            vx = random.uniform(-3, 3)
            vy = random.uniform(-8, -2)
            life = random.uniform(0.5, 1.2)
            size = random.uniform(2, 6)
            self.particles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'life': life, 'max_life': life, 'color': color, 'size': size})

    def update_and_draw(self, dt, frame):
        if not getattr(self.config, 'enable_particles', True):
            return frame
        
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        alive_particles = []
        for p in self.particles:
            p['life'] -= dt
            if p['life'] > 0:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['vy'] += 5.0 * dt  # mild gravity
                
                alpha = p['life'] / p['max_life']
                c = tuple(int(ch * alpha) for ch in p['color'])
                radius = int(p['size'] * alpha)
                if radius > 0:
                    # Draw a glowing particle (core + bloom)
                    cv2.circle(overlay, (int(p['x']), int(p['y'])), radius + 2, tuple(int(ch * 0.3) for ch in c), -1, cv2.LINE_AA)
                    cv2.circle(overlay, (int(p['x']), int(p['y'])), radius, c, -1, cv2.LINE_AA)
                alive_particles.append(p)
                
        self.particles = alive_particles
        cv2.add(frame, overlay, dst=frame)
        return frame

# ============================================================================
# SHOCKWAVE SYSTEM (SNAP GESTURE)
# ============================================================================

class ShockwaveSystem:
    def __init__(self, config: Config):
        self.config = config
        self.shockwaves = []

    def emit(self, x, y, color):
        self.shockwaves.append({
            'x': x, 'y': y, 
            'radius': 10.0, 
            'max_radius': 800.0, 
            'speed': 1500.0, 
            'color': color,
            'thickness': 25.0
        })

    def update_and_draw(self, dt, frame):
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        alive = []
        for s in self.shockwaves:
            s['radius'] += s['speed'] * dt
            s['thickness'] = max(1.0, s['thickness'] - 40.0 * dt)
            alpha = max(0.0, 1.0 - (s['radius'] / s['max_radius']))
            
            if alpha > 0.01:
                c = tuple(int(ch * alpha) for ch in s['color'])
                cv2.circle(overlay, (int(s['x']), int(s['y'])), int(s['radius']), c, int(s['thickness']), cv2.LINE_AA)
                
                # Inner bright flash flash
                if s['radius'] < 150:
                    c_inner = tuple(int(ch * alpha * 0.8) for ch in (255, 255, 255))
                    cv2.circle(overlay, (int(s['x']), int(s['y'])), int(s['radius']), c_inner, -1, cv2.LINE_AA)
                    
                alive.append(s)
                
        self.shockwaves = alive
        cv2.add(frame, overlay, dst=frame)
        return frame

# ============================================================================
# POWER ORB EFFECT (TWO HANDS)
# ============================================================================

class PowerOrbEffect:
    """Intense energy orb forming between two hands with lightning arcs"""

    def __init__(self, config: Config):
        self.config = config
        self.time = 0.0
        self.current_alpha = 0.0
        self.fade_speed = 3.0
        self.last_center = None
        self.p1 = None
        self.p2 = None
        self.dist = 0.0
        self.shake_amount = 0.0

    def update(self, dt: float, landmarks_list):
        self.time += dt
        
        target_alpha = 0.0
        
        if landmarks_list and len(landmarks_list) >= 2:
            target_alpha = 1.0
            
            # calculate palm centers
            p1 = self._get_palm_center(landmarks_list[0])
            p2 = self._get_palm_center(landmarks_list[1])
            self.last_center = (
                int((p1[0] + p2[0]) / 2),
                int((p1[1] + p2[1]) / 2)
            )
            self.p1 = p1
            self.p2 = p2
            self.dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
        alpha_diff = target_alpha - self.current_alpha
        self.current_alpha += alpha_diff * dt * self.fade_speed
        self.current_alpha = max(0.0, min(1.0, self.current_alpha))

    def _get_palm_center(self, landmarks):
        palm_indices = [0, 5, 9, 13, 17]
        cx = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
        cy = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))
        return cx, cy

    def draw(self, frame: np.ndarray) -> np.ndarray:
        self.shake_amount = 0.0
        if not getattr(self.config, 'enable_power_orb', True) or self.current_alpha <= 0.01 or self.last_center is None:
            return frame

        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Orb intensity based on distance (closer = more intense)
        dist_factor = max(0.0, min(1.0, 1.0 - (self.dist / (w * 0.8))))
        intensity = self.current_alpha * (0.3 + 0.7 * dist_factor)
        
        if dist_factor > 0.85:
            self.shake_amount = (dist_factor - 0.85) * 60.0 * self.current_alpha
            
        # High energy cyan/blue color (BGR)
        color = (255, 200, 50) 
        
        # Draw lightning arcs between palms and orb
        self._draw_lightning(overlay, self.p1, self.last_center, color, intensity)
        self._draw_lightning(overlay, self.p2, self.last_center, color, intensity)
        
        # Draw Orb
        self._draw_orb(overlay, self.last_center, color, intensity)
        
        cv2.add(frame, overlay, dst=frame)
        return frame

    def _draw_lightning(self, overlay, start, end, color, intensity):
        if intensity <= 0.1: return
        
        num_arcs = int(2 + 4 * intensity)
        for arc in range(num_arcs):
            num_segments = random.randint(5, 10)
            pts = [start]
            for i in range(1, num_segments):
                t = i / num_segments
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                
                # Add jitter based on distance and intensity
                jitter = 40.0 * intensity * math.sin(t * math.pi)
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
                pts.append((int(x), int(y)))
            pts.append(end)
            
            alpha = intensity * random.uniform(0.5, 1.0)
            c = tuple(int(ch * alpha) for ch in color)
            
            thickness = random.randint(1, 3)
            # Draw lines
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i+1], c, thickness + 2, cv2.LINE_AA)
                cv2.line(overlay, pts[i], pts[i+1], (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

    def _draw_orb(self, overlay, center, color, intensity):
        # Base size pulses with time and intensity
        pulse = math.sin(self.time * 20.0) * 0.1 + 1.0
        size = int((80 * intensity + 20) * pulse)
        if size <= 0: return
        
        passes = [
            (int(size * 2.0), 0.05, color),
            (int(size * 1.2), 0.15, color),
            (size, 0.4, color),
            (int(size * 0.5), 0.8, color),
            (int(size * 0.2), 1.0, (255, 255, 255))
        ]
        
        for radius, alpha_mult, c in passes:
            if radius <= 0: continue
            layer_color = tuple(int(ch * alpha_mult * intensity) for ch in c)
            cv2.circle(overlay, center, radius, layer_color, -1, cv2.LINE_AA)
            
        # Draw dynamic rotating rings around orb
        for i in range(3):
            angle = self.time * (100.0 + i * 50.0) * (1 if i % 2 == 0 else -1)
            r_x = int(size * (1.5 + i * 0.5))
            r_y = int(size * (0.3 + i * 0.2))
            
            c_ring = tuple(int(ch * 0.6 * intensity) for ch in color)
            cv2.ellipse(overlay, center, (r_x, r_y), angle, 0, 360, c_ring, 2 + i, cv2.LINE_AA)

# ============================================================================
# FINGER LASER EFFECT (FINGER GUN GESTURE)
# ============================================================================

class FingerLaserEffect:
    def __init__(self, config: Config):
        self.config = config
        self.laser_alpha = {}

    def _is_finger_gun(self, landmarks):
        if len(landmarks) < 21: return False
        
        def dist(p1, p2):
            return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            
        wrist = landmarks[0]
        
        # Index extended
        if dist(landmarks[8], wrist) < dist(landmarks[6], wrist): return False
        
        # Others closed
        for tip, pip in [(12, 10), (16, 14), (20, 18)]:
            if dist(landmarks[tip], wrist) > dist(landmarks[pip], wrist): return False
            
        return True

    def draw(self, frame: np.ndarray, landmarks_list, dt) -> np.ndarray:
        if not getattr(self.config, 'enable_finger_laser', True): return frame
        
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        if landmarks_list is None:
            landmarks_list = []
            
        for i, landmarks in enumerate(landmarks_list):
            if i not in self.laser_alpha: self.laser_alpha[i] = 0.0
            
            target_alpha = 1.0 if self._is_finger_gun(landmarks) else 0.0
            diff = target_alpha - self.laser_alpha[i]
            self.laser_alpha[i] = max(0.0, min(1.0, self.laser_alpha[i] + diff * dt * 10.0))
            
            alpha = self.laser_alpha[i]
            if alpha > 0.01:
                p_pip = landmarks[6]
                p_tip = landmarks[8]
                
                dx = p_tip[0] - p_pip[0]
                dy = p_tip[1] - p_pip[1]
                length = math.hypot(dx, dy)
                if length > 0.1:
                    dir_x = dx / length
                    dir_y = dy / length
                    
                    laser_length = 2000 
                    end_x = int(p_tip[0] + dir_x * laser_length)
                    end_y = int(p_tip[1] + dir_y * laser_length)
                    
                    start = (int(p_tip[0]), int(p_tip[1]))
                    end = (end_x, end_y)
                    
                    color = (0, 50, 255) # Intense Red Laser
                    
                    passes = [
                        (30, 0.1),
                        (15, 0.3),
                        (6, 0.7),
                        (2, 1.0)
                    ]
                    for thickness, a_mult in passes:
                        c = tuple(int(ch * alpha * a_mult) for ch in color)
                        if thickness <= 2: c = tuple(int(ch * alpha) for ch in (255,200,200))
                        cv2.line(overlay, start, end, c, thickness, cv2.LINE_AA)
                        
                    flash_radius = int(random.uniform(15, 30) * alpha)
                    cv2.circle(overlay, start, flash_radius, tuple(int(ch*alpha*0.5) for ch in color), -1, cv2.LINE_AA)
                    cv2.circle(overlay, start, flash_radius//3, (255,255,255), -1, cv2.LINE_AA)
                    
        cv2.add(frame, overlay, dst=frame)
        return frame

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HandVFXApp:
    """Main application class"""

    def __init__(self):
        self.config = Config()
        self.hand_tracker = HandTracker(self.config)
        self.neon_renderer = NeonRenderer(self.config)
        self.beam_effect = BeamEffect(self.config)
        self.power_orb_effect = PowerOrbEffect(self.config)
        self.shield_effect = MagicShieldEffect(self.config)
        self.particle_system = ParticleSystem(self.config)
        self.shockwave_system = ShockwaveSystem(self.config)
        self.finger_laser_effect = FingerLaserEffect(self.config)
        self.trail_canvas = None
        self.is_pinching = {}
        self.glitch_time = 0.0
        self.last_time = time.time()
        self.frame_count = 0

        # Camera init with macOS fallback
        print("[VFX] Opening webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[VFX] Default backend failed, trying DirectShow...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("[VFX] ERROR: Could not open webcam on index 0, trying index 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("[VFX] ERROR: No webcam found. Exiting.")
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[VFX] Webcam opened successfully.")

        cv2.namedWindow("Hand VFX System", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Hand VFX System", cv2.WND_PROP_TOPMOST, 1)
        print("[VFX] Window created.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return it with VFX applied"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.neon_renderer.update(dt)

        landmarks_list = self.hand_tracker.detect_hands(frame)
        
        # Grayscale / Cyberpunk Background Filter
        if getattr(self.config, 'enable_grayscale_bg', False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame = cv2.convertScaleAbs(frame, alpha=0.4, beta=0)

        # Handle trail canvas decay
        if getattr(self.config, 'enable_trails', True):
            if self.trail_canvas is None or self.trail_canvas.shape != frame.shape:
                self.trail_canvas = np.zeros_like(frame)
            else:
                self.trail_canvas = cv2.addWeighted(self.trail_canvas, 0.85, np.zeros_like(self.trail_canvas), 0, 0)

        if landmarks_list:
            hand_colors = [
                self.config.colors['primary'],
                self.config.colors['secondary'],
            ]
            for hand_idx, landmarks in enumerate(landmarks_list):
                color = hand_colors[hand_idx % len(hand_colors)]
                self.neon_renderer.draw_hand_skeleton(frame, landmarks, color)
                self.neon_renderer.draw_hand_aura(frame, landmarks, color)
                
                # Draw to trail canvas
                if getattr(self.config, 'enable_trails', True) and self.trail_canvas is not None:
                    dim_color = tuple(int(c * 0.4) for c in color)
                    self.neon_renderer.draw_hand_skeleton(self.trail_canvas, landmarks, dim_color)
                
                # Detect Pinch / Snap
                if len(landmarks) > 12:
                    thumb = landmarks[4]
                    middle = landmarks[12]
                    dist = math.hypot(thumb[0] - middle[0], thumb[1] - middle[1])
                    
                    was_pinching = self.is_pinching.get(hand_idx, False)
                    is_pinching_now = dist < 30.0
                    
                    if is_pinching_now and not was_pinching:
                        # SNAP!
                        self.shockwave_system.emit(int(thumb[0]), int(thumb[1]), color)
                        # Add a huge shake
                        self.power_orb_effect.shake_amount = max(getattr(self.power_orb_effect, 'shake_amount', 0), 25.0)
                        self.glitch_time = 0.4
                        
                    self.is_pinching[hand_idx] = is_pinching_now
                
                # Emit particles from fingertips
                if getattr(self.config, 'enable_particles', True):
                    for tip_idx in [4, 8, 12, 16, 20]:
                        if tip_idx < len(landmarks):
                            x, y = int(landmarks[tip_idx][0]), int(landmarks[tip_idx][1])
                            self.particle_system.emit(x, y, color, amount=1)

        # Update and draw particles
        frame = self.particle_system.update_and_draw(dt, frame)
        
        # Draw trails
        if getattr(self.config, 'enable_trails', True) and self.trail_canvas is not None:
            cv2.add(frame, self.trail_canvas, dst=frame)
            
        # Draw shockwaves
        frame = self.shockwave_system.update_and_draw(dt, frame)

        # Update and draw beam connections
        if self.config.enable_beam:
            self.beam_effect.update(dt, landmarks_list)
            frame = self.beam_effect.draw(frame)  # MUST use returned frame
            
        # Update and draw power orb
        if getattr(self.config, 'enable_power_orb', True):
            self.power_orb_effect.update(dt, landmarks_list)
            frame = self.power_orb_effect.draw(frame)

        # Update and draw magic shield
        self.shield_effect.update(dt, landmarks_list)
        frame = self.shield_effect.draw(frame)
        
        # Update and draw finger laser
        frame = self.finger_laser_effect.draw(frame, landmarks_list, dt)

        # Cinematic tint overlay
        if self.config.cinematic_darkening > 0:
            tint = np.full_like(frame, (10, 5, 20), dtype=np.uint8)
            cv2.addWeighted(frame, 1.0 - self.config.cinematic_darkening,
                            tint, self.config.cinematic_darkening, 0, dst=frame)

        # Subtle bloom
        if self.config.bloom_intensity > 0:
            bloom = cv2.GaussianBlur(frame, (21, 21), 0)
            cv2.addWeighted(frame, 1.0, bloom, self.config.bloom_intensity, 0, dst=frame)

        # Screen shake
        shake_amt = 0.0
        if getattr(self.config, 'enable_power_orb', True) and getattr(self.config, 'enable_screen_shake', True):
            shake_amt = getattr(self.power_orb_effect, 'shake_amount', 0.0)
            
        if shake_amt > 1.0:
            dx = random.randint(-int(shake_amt), int(shake_amt))
            dy = random.randint(-int(shake_amt), int(shake_amt))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
            
        # RGB Split / Chromatic Aberration Glitch
        if getattr(self.config, 'enable_glitch', True) and getattr(self, 'glitch_time', 0) > 0:
            self.glitch_time -= dt
            shift = int(random.uniform(5, 25) * (self.glitch_time / 0.4))
            if shift > 0:
                b, g, r = cv2.split(frame)
                M_b = np.float32([[1, 0, shift], [0, 1, 0]])
                b = cv2.warpAffine(b, M_b, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REFLECT)
                M_r = np.float32([[1, 0, -shift], [0, 1, 0]])
                r = cv2.warpAffine(r, M_r, (r.shape[1], r.shape[0]), borderMode=cv2.BORDER_REFLECT)
                frame = cv2.merge((b, g, r))

        self.draw_info(frame)
        return frame

    def draw_info(self, frame: np.ndarray):
        """Draw HUD text"""
        lines = [
            "Hand VFX  |  q: quit",
            f"Aura: {'ON' if self.config.enable_aura else 'OFF'} (a)  "
            f"Beam: {'ON' if self.config.enable_beam else 'OFF'} (b)  "
            f"Shield: {'ON' if self.config.enable_shield else 'OFF'} (s)  "
            f"Orb: {'ON' if getattr(self.config, 'enable_power_orb', True) else 'OFF'} (o)  ",
            f"Gray BG: {'ON' if getattr(self.config, 'enable_grayscale_bg', False) else 'OFF'} (g)  "
            f"Particles: {'ON' if getattr(self.config, 'enable_particles', True) else 'OFF'} (p)  "
            f"Trails: {'ON' if getattr(self.config, 'enable_trails', True) else 'OFF'} (t)  "
            f"Shake: {'ON' if getattr(self.config, 'enable_screen_shake', True) else 'OFF'} (k)  ",
            f"Laser: {'ON' if getattr(self.config, 'enable_finger_laser', True) else 'OFF'} (l)  "
            f"Glitch: {'ON' if getattr(self.config, 'enable_glitch', True) else 'OFF'} (c)"
        ]
        y = 28
        for text in lines:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 24

    def handle_key(self, key: int):
        """Handle keyboard input"""
        if key == ord('a'):
            self.config.enable_aura = not self.config.enable_aura
            print(f"[VFX] Aura: {'ON' if self.config.enable_aura else 'OFF'}")
        elif key == ord('b'):
            self.config.enable_beam = not self.config.enable_beam
            print(f"[VFX] Beam: {'ON' if self.config.enable_beam else 'OFF'}")
        elif key == ord('s'):
            self.config.enable_shield = not self.config.enable_shield
            print(f"[VFX] Shield: {'ON' if self.config.enable_shield else 'OFF'}")
        elif key == ord('o'):
            self.config.enable_power_orb = not getattr(self.config, 'enable_power_orb', True)
            print(f"[VFX] Orb: {'ON' if self.config.enable_power_orb else 'OFF'}")
        elif key == ord('g'):
            self.config.enable_grayscale_bg = not getattr(self.config, 'enable_grayscale_bg', False)
            print(f"[VFX] Gray BG: {'ON' if self.config.enable_grayscale_bg else 'OFF'}")
        elif key == ord('p'):
            self.config.enable_particles = not getattr(self.config, 'enable_particles', True)
            print(f"[VFX] Particles: {'ON' if self.config.enable_particles else 'OFF'}")
        elif key == ord('k'):
            self.config.enable_screen_shake = not getattr(self.config, 'enable_screen_shake', True)
            print(f"[VFX] Shake: {'ON' if self.config.enable_screen_shake else 'OFF'}")
        elif key == ord('t'):
            self.config.enable_trails = not getattr(self.config, 'enable_trails', True)
            print(f"[VFX] Trails: {'ON' if self.config.enable_trails else 'OFF'}")
            if not getattr(self.config, 'enable_trails', True):
                self.trail_canvas = None # Clear trails
        elif key == ord('l'):
            self.config.enable_finger_laser = not getattr(self.config, 'enable_finger_laser', True)
            print(f"[VFX] Laser: {'ON' if self.config.enable_finger_laser else 'OFF'}")
        elif key == ord('c'):
            self.config.enable_glitch = not getattr(self.config, 'enable_glitch', True)
            print(f"[VFX] Glitch: {'ON' if self.config.enable_glitch else 'OFF'}")

    def run(self):
        """Main application loop"""
        if self.cap is None:
            print("[VFX] Cannot run — no camera available.")
            return

        print("[VFX] Starting. Controls: q=quit  a=aura  b=beam  s=shield")

        # Validate first frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[VFX] ERROR: Could not read first frame.")
            self.cap.release()
            return
        print("[VFX] First frame received. Entering main loop.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[VFX] WARNING: Frame read failed, retrying...")
                    continue

                frame = cv2.flip(frame, 1)
                processed = self.process_frame(frame)

                cv2.imshow("Hand VFX System", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[VFX] Quit key pressed.")
                    break
                self.handle_key(key)

                self.frame_count += 1

        except KeyboardInterrupt:
            print("[VFX] Interrupted by user.")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[VFX] Application closed.")


def main():
    app = HandVFXApp()
    app.run()


if __name__ == "__main__":
    main()
