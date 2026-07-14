# Neon Hand VFX System

A real-time, highly dynamic, cyberpunk-style hand tracking VFX system built with OpenCV and MediaPipe. 

It maps glowing neon skeletons to your hands and features multiple highly interactive visual effects triggered by different gestures!

## Features

- **Neon Skeleton & Aura:** A smooth, glowing skeletal overlay with a pulsing palm aura.
- **Fingertip Particles:** Glowing embers falling gracefully from your fingertips.
- **Ghost Trails:** Fluid, fading motion blur that trails behind your hand movements.
- **Connection Beams:** Bring two hands close together to draw brilliant neon energy lines connecting your fingertips. When fingertips touch, they erupt in a magical star-flare spark!
- **Power Orb:** Hold two hands apart to form a massive, crackling electrical energy orb between them. Bring your hands closer to intensify the orb, shaking the screen!
- **Magic Shield:** Hold one hand flat (open palm gesture) to conjure a Dr. Strange-style rotating geometric magic shield.
- **Finger Laser Vision:** Make a "Finger Gun" gesture to shoot a massive red laser beam across the screen!
- **Snap Shockwave & Glitch:** Perform a "Snap" gesture (touch thumb and middle finger rapidly) to trigger a violent reality-shattering shockwave and RGB chromatic aberration glitch!
- **Grayscale Cyberpunk Filter:** Desaturates the webcam feed while keeping the neon effects at 100% vibrance for an incredible "Sin City" / Cyberpunk aesthetic.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the script using Python:

```bash
python main.py
```

### Controls (Keyboard Toggles)

- `q`: Quit Application
- `a`: Toggle Palm Aura
- `b`: Toggle Fingertip Connection Beams
- `s`: Toggle Magic Shield
- `o`: Toggle Power Orb
- `p`: Toggle Fingertip Particles
- `t`: Toggle Ghost Trails
- `l`: Toggle Finger Laser
- `k`: Toggle Screen Shake
- `c`: Toggle Reality Glitch (on Snap)
- `g`: Toggle Grayscale Background Filter

## How it works

This app utilizes Google's `mediapipe` library for robust, real-time hand landmark detection. OpenCV handles the rendering using highly optimized additive blending techniques (`cv2.add` and `cv2.addWeighted`) to create true "glow" effects without muddying the background video feed. Each visual effect is encapsulated into its own modular class for easy extension.
