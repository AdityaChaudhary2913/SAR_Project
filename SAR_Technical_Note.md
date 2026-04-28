# SAR Technical Note
### A Beginner's Working Understanding of Synthetic Aperture Radar for ML Applications

*Aditya Chaudhary — SAR + ML Take-Home Assignment*

---

## What Even Is SAR?

Before starting this assignment, my mental model of satellite imagery was basically "camera in space." Optical satellites work exactly like that they are passive sensors that wait for sunlight to bounce off the Earth and hit their lens, just like a phone camera. The obvious problem: clouds block sunlight, and half the planet is dark at any given time.

SAR (Synthetic Aperture Radar) is a completely different idea. Instead of waiting for sunlight, it fires its own microwave pulses at the ground and listens for the echoes that bounce back. Think of it like a bat using echolocation, it's an *active* sensor that carries its own illumination. Because it uses microwaves (not visible light), it can image through clouds, through rain, and at night.

The "Synthetic Aperture" part is the clever engineering trick that makes it practical. To get a sharp radar image, physics requires a very large antenna but you can't strap a 1-km antenna onto a satellite. The solution is to fake it. As the satellite moves along its orbit, it fires pulses continuously and collects echoes from many positions. These echoes are then combined in software as if they came from one giant antenna. That software-constructed aperture is the "synthetic" part, and it's how Sentinel-1 achieves 10-meter resolution from 693 km above the Earth.

The key difference from optical imagery in one table:

| Property | Optical Satellite | SAR (Sentinel-1) |
|---|---|---|
| Energy source | Passive — uses sunlight | Active — fires own radar pulse |
| Works through clouds? | No | Yes |
| Works at night? | No | Yes |
| What pixels represent | Reflected visible light | Microwave backscatter intensity |
| Typical resolution | 10 cm – 10 m | 10 m (GRD-H) |
| Best at | Color, texture, visual ID | Structure, moisture, change detection |

---

## Key Concepts That Matter for ML

### Backscatter — What Your Pixel Values Actually Mean

Every pixel in a SAR image represents how strongly that patch of ground reflected the radar pulse back toward the satellite. This value is called **backscatter**, formally written as sigma-nought (σ⁰). It's expressed in decibels (dB), which means the pixel values are negative numbers like −22, −10, or −5 — not the 0–255 we expect from a regular image.

The dB scale is used because the raw backscatter range spans several orders of magnitude (calm water reflects almost nothing; a building corner can reflect thousands of times more). The conversion is:

\[\sigma^0_{dB} = 10 \times \log_{10}(\sigma^0_{linear})\]

Different surfaces scatter radar energy in physically distinct ways, which is what makes SAR useful for classification:

- **Specular scattering** — Smooth, flat surfaces (calm water, roads) act like mirrors and deflect radar *away* from the satellite. Result: very dark pixels, around −22 to −28 dB.
- **Diffuse scattering** — Rough surfaces (bare soil, gravel) scatter energy in all directions, some of which returns to the satellite. Result: medium-dark pixels, around −15 to −18 dB.
- **Volume scattering** — Vegetation (forests, crops) bounces the signal around internally off leaves and branches. The signal gets scrambled in polarization before returning. Result: medium grey pixels, around −8 to −12 dB.
- **Double-bounce** — Vertical walls next to flat ground (buildings, tree trunks in floodwater) create a retroreflector geometry — radar hits the ground, bounces to the wall, bounces straight back. Result: very bright pixels, around −2 to +5 dB.

This backscatter signature is what a model learns from. Water is dark, cities are bright, forests are in between — and the contrast is large enough to make many classification tasks tractable.

---

### Polarization — The Two Input Channels

When Sentinel-1 fires its radar pulse, it transmits a vertically polarized wave (V) and receives echoes in both V and H (horizontal). This gives two simultaneous channels: **VV** (transmit vertical, receive vertical) and **VH** (transmit vertical, receive horizontal).

These two channels capture different physical phenomena:

- **VV** is dominated by surface/specular scattering — it's most sensitive to smooth surfaces, water, and urban structures.
- **VH** is dominated by volume scattering — it's most sensitive to vegetation, because forests scramble the polarization of the returning wave.

In practice, a flooded area looks very dark in VV (smooth water surface) while a forest looks much brighter in VH relative to VV. This means the **VV/VH ratio** (computed as `VV − VH` in dB) acts like a vegetation index, similar conceptually to NDVI in optical remote sensing. I use this as a third engineered feature channel for my ML model, giving an input tensor of shape `[3, H, W]` per tile.

---

### Incidence Angle

The incidence angle is the angle between the radar beam and the vertical. Sentinel-1 IW mode operates between roughly 29° and 46° across its swath. A shallow incidence angle means the radar grazes the surface more, while a steep angle hits more directly.

For practical ML work, incidence angle matters mainly because: (1) the same surface looks different at different angles, and (2) it varies across a single scene from left edge to right edge. However, when using preprocessed GRD data with terrain correction applied (which is what I'm doing), most of this variation is normalized. So incidence angle is worth knowing about, but it doesn't require manual correction in my pipeline.

---

### Speckle — The Graininess Problem

Raw SAR images look grainy and speckled, even over perfectly uniform surfaces. This isn't regular sensor noise, it's a fundamental consequence of coherent radar imaging. Within a single 10×10m pixel, thousands of tiny scatterers (leaves, pebbles, blades of grass) all return echoes simultaneously. Their echoes interfere constructively or destructively based on phase differences, making the pixel randomly brighter or darker than its "true" backscatter value.

For my project, I'm starting without explicit filtering and relying on the CNN to handle speckle implicitly through its learned spatial smoothing.

---

### GRD vs. SLC

Two product types come up constantly in Sentinel-1 literature:

- **GRD (Ground Range Detected)**: Amplitude-only, multi-looked, already projected to ground coordinates. This is what I'm using — it's the right choice for backscatter-based ML tasks.
- **SLC (Single Look Complex)**: Contains both amplitude and phase, in slant range geometry. Used for interferometric applications (InSAR), deformation mapping, DEM generation. Not needed for classification or detection tasks.

---

## Application Areas I Find Interesting

### Flood and Water Detection

This is the application I'm building for this assignment. The physics make it unusually tractable: water has one of the lowest backscatter values in C-band SAR (~−22 to −28 dB in VV) due to specular reflection, creating a very high contrast against nearly every other land surface. When a flood happens, previously rough land (fields, roads, urban fringes) suddenly becomes water and drops 10–15 dB in brightness. That's a massive signal.

What makes it compelling beyond just the physics: floods happen during storms, and storms mean clouds, which means optical satellites are blind exactly when you need them most. SAR is the only operational tool that can reliably observe floods in near real-time.

The `Sen1Floods11` dataset covers 11 major flood events globally with Sentinel-1 imagery and hand-labeled water masks. It's the standard benchmark and what I'll train and evaluate on.

### Land Cover / Land Use Classification

The second application area I find interesting is mapping land cover types — urban, water, bare soil, vegetation, agriculture — directly from SAR. Each class has a distinct backscatter fingerprint from the scattering physics above, and the VV/VH combination adds meaningful discriminative power.

The challenge compared to flood detection is that the class boundaries are softer (forests and crops can overlap in backscatter range) and seasonal variation adds noise. Multi-temporal features — using the mean and standard deviation of backscatter over several months — significantly help. The `BigEarthNet-S1` dataset (590k patches, multi-label) and `SEN12MS` (paired Sentinel-1 + Sentinel-2 + labels) are the main open resources here.

---

*This note covers what I needed to understand to design and build the ML pipeline. The most practically important concepts turned out to be backscatter interpretation (what do my pixels mean?), polarization channels (what are my input features?), and the preprocessing pipeline (how do I go from raw data to a clean tensor?).*
