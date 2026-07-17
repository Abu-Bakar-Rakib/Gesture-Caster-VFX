# ✨ Neon Hand VFX System

<p align="center">
  <img src="assets/banner.gif" alt="Neon Hand VFX Banner" width="100%">
</p>

<p align="center">
  <strong>Real-Time Cyberpunk Hand Tracking Visual Effects</strong><br>
  Built with <b>OpenCV</b>, <b>MediaPipe</b>, and <b>NumPy</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg">
  <img src="https://img.shields.io/badge/Status-Active-success.svg">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg">
</p>

---

## 🚀 Overview

**Neon Hand VFX System** is a real-time gesture-driven visual effects engine that transforms your webcam into a futuristic cyberpunk experience.

Using Google's **MediaPipe Hand Tracking** and custom OpenCV rendering pipelines, the system generates glowing neon skeletons, magical shields, energy beams, power orbs, laser attacks, particle effects, shockwaves, and cinematic post-processing effects — all in real time.

Perfect for:

* 🎮 Interactive VFX Experiments
* 🎥 Live Streaming Effects
* 🧙‍♂️ AR Magic Simulations
* 🤖 Computer Vision Demonstrations
* 🧪 Creative Coding Projects
* 🎨 Cyberpunk Visual Art
<img width="1179" height="746" alt="image" src="https://github.com/user-attachments/assets/488139b5-3d64-4259-bc5e-80bae1ff78f2" />
<img width="1122" height="739" alt="image" src="https://github.com/user-attachments/assets/dbda7bd7-8e0b-47b8-a885-2781614b9f5a" />

# 🎬 Features

## 🌌 Neon Skeleton System

* Smooth hand landmark tracking
* Multi-layer neon glow rendering
* Dynamic pulse animation
* Individual finger color mapping
* Soft palm aura effect

---

## ✨ Fingertip Particle Engine

Every fingertip emits glowing energy particles.

### Features

* Dynamic particle spawning
* Gravity simulation
* Bloom effects
* Neon fading particles

---

## 👻 Ghost Motion Trails

Generate cinematic afterimages behind moving hands.

### Features

* Motion persistence
* Trail fading
* Neon blending
* Performance optimized rendering

---

## ⚡ Connection Beam System

Bring both hands closer to create energy links.

### Features

* Neon fingertip-to-fingertip beams
* Multi-layer glow rendering
* Distance-aware activation
* Animated energy connections

### Special Effect

When matching fingertips touch:

💥 Energy Burst
✨ Star Flare
🌟 Bright Contact Spark

---

## 🔮 Power Orb

Create a giant energy sphere between two hands.

### Features

* Dynamic orb growth
* Pulsating energy core
* Rotating energy rings
* Electrical lightning arcs
* Distance-based intensity scaling

### Bonus Effects

* Screen shake
* Orb charging animation
* High-energy visual bloom

---

## 🛡️ Magic Shield

Open your palm to summon a mystical shield inspired by cinematic magic effects.

### Features

* Rotating geometric circles
* Energy runes
* Inner rotating squares
* Octagonal structures
* Neon glow layers
* Animated magical core

---

## 🔫 Finger Laser

Perform a Finger Gun gesture.

### Effects

* Massive red laser beam
* Dynamic beam bloom
* Finger muzzle flash
* High-intensity glow rendering

---

## 💥 Snap Shockwave

Perform a rapid thumb-to-middle-finger snap gesture.

### Effects

* Expanding energy wave
* Reality distortion
* Screen impact
* Cinematic visual explosion

---

## 🧬 Reality Glitch

Triggered after a snap event.

### Features

* RGB channel separation
* Chromatic aberration
* Cyberpunk distortion
* Dynamic glitch intensity

---

## 🌃 Cyberpunk Grayscale Mode

Transforms the webcam feed into a dark monochrome scene while preserving all neon effects at full color intensity.

### Visual Style

* Sin City aesthetic
* Cyberpunk atmosphere
* High contrast rendering
* Vibrant neon highlights

---

# 🧠 Technology Stack

| Component            | Technology               |
| -------------------- | ------------------------ |
| Hand Tracking        | MediaPipe                |
| Computer Vision      | OpenCV                   |
| Numerical Processing | NumPy                    |
| Language             | Python                   |
| Rendering            | OpenCV Additive Blending |
| Effects Engine       | Custom VFX Pipeline      |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/neon-hand-vfx.git

cd neon-hand-vfx
```

## Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

---

# ▶️ Run

```bash
python main.py
```

---

# 🎮 Controls

| Key | Action                     |
| --- | -------------------------- |
| Q   | Quit Application           |
| A   | Toggle Palm Aura           |
| B   | Toggle Energy Beams        |
| S   | Toggle Magic Shield        |
| O   | Toggle Power Orb           |
| P   | Toggle Particles           |
| T   | Toggle Ghost Trails        |
| L   | Toggle Finger Laser        |
| K   | Toggle Screen Shake        |
| C   | Toggle Reality Glitch      |
| G   | Toggle Cyberpunk Grayscale |

---

# 🎭 Gesture Guide

| Gesture                    | Effect             |
| -------------------------- | ------------------ |
| Open Palm                  | Magic Shield       |
| Finger Gun                 | Laser Beam         |
| Two Hands Close            | Power Orb          |
| Two Hands Connected        | Energy Beams       |
| Thumb + Middle Finger Snap | Shockwave + Glitch |

---

# ⚙️ Architecture

```text
Webcam
   │
   ▼
MediaPipe Hand Tracking
   │
   ▼
Landmark Extraction
   │
   ├── Neon Skeleton
   ├── Particles
   ├── Ghost Trails
   ├── Magic Shield
   ├── Finger Laser
   ├── Connection Beams
   ├── Power Orb
   └── Shockwave System
            │
            ▼
Post Processing
(Bloom + Glitch + Screen Shake)
            │
            ▼
Final Render
```

---

# 📈 Performance

Optimized for real-time execution:

* Multi-thread friendly architecture
* Lightweight MediaPipe tracking
* Additive glow rendering
* Smooth 30–60 FPS performance
* Modular effect pipeline

---

# 🖼️ Screenshots

Add your screenshots here:

```markdown
assets/
├── shield.png
├── laser.png
├── orb.png
├── beam.png
└── cyberpunk.png
```

Example:

```markdown
![Magic Shield](assets/shield.png)
![Power Orb](assets/orb.png)
```

---

# 🔧 Future Improvements

* Hand gesture customization
* Audio reactive VFX
* Full-body pose effects
* GPU acceleration
* Multi-user interaction
* VR/AR integration
* Particle physics enhancements

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project interesting, consider giving it a ⭐ on GitHub.

It helps the project reach more developers and creators.

---

<p align="center">
  <b>Built with Python • MediaPipe • OpenCV • Neon Energy ⚡</b>
</p>
