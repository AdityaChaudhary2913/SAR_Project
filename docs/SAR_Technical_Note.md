# SAR Technical Note

## What Even Is SAR?

Prior to the research, I perceived satellite imagery as a simple photo taken by a camera installed in space. Optical satellites operate in much the same way as any other photographic instrument, the images taken by such satellites are obtained passively, relying on sunlight hitting the Earth surface and then being captured by the sensor of the satellite. This poses an evident limitation, as clouds may obstruct sunlight and about half of the Earth surface is constantly in darkness.

However, SAR satellites operate according to entirely different principles. Unlike optical satellites that capture passive images, SAR uses active approach, meaning that SAR satellites emit their own microwaves toward the ground and collect data on received echoes. To draw an analogy, it operates in much the same way as echolocation does in bats. In addition, since SAR emits its own microwaves, which means that images are not dependent on daylight conditions and can be obtained at nighttime and through precipitation.

What makes the use of such imaging system feasible is the application of synthetic aperture, which helps achieve high resolution despite size limitations of equipment. Indeed, due to specific physical requirements, achieving 10-meter resolution would require fitting a very large antenna on a satellite; however, as the satellite orbits around the Earth, it receives multiple echoes and software simulates those from a large antenna.

The key difference from optical imagery in one table:

| Property | Optical Satellite | SAR (Sentinel-1) |
|---|---|---|
| Energy source | Passive — uses sunlight | Active — fires own radar pulse |
| Works through clouds? | No | Yes |
| Works at night? | No | Yes |
| What pixels represent | Reflected visible light | Microwave backscatter intensity |
| Typical resolution | 10 cm – 10 m | 10 m (GRD-H) |
| Best at | Color, texture, visual ID | Structure, moisture, change detection |

## Key Concepts That Matter for ML

### Backscatter — What Your Pixel Values Actually Mean

Each pixel within a SAR image contains information about the amount by which the corresponding surface scattered its reflection back into the space shuttle. Backscatter is formally described as the σ⁰ parameter in units of decibels (dB). Pixel values will be negative numbers, rather than positive integers ranging from 0 to 255. They will be between −22 dB and +20 dB.

This scale was chosen because the raw intensity of the backscatter may span several orders of magnitude (from zero for calm water to thousands for a building corner).

A wide variety of materials have specific scattering mechanisms, hence providing opportunities for classification:

* **Specular scattering** – smooth surfaces behave like mirrors in the sense that they deflect energy away from the sensor. As a result, pixel intensity is extremely low, usually below −22 dB.
* **Diffuse scattering** – rough surfaces scatter radar beams across a wide angle, and some of that is reflected back to the space shuttle. This corresponds to fairly dark pixels, from −15 to −18 dB.
* **Volume scattering** – vegetation acts as a random mirror with multiple surfaces within each pixel. Polarization gets confused in such an environment. Pixel intensity is in the gray zone at −8 to −12 dB.
* **Double-bounce** – when the vertical wall meets horizontal ground, the geometry turns into that of a retroreflector. Radar reflects from ground to wall and back. This yields very intense signals above −2 dB and up to +5 dB.

As you see, these properties provide a robust ground truth data for the learning model.

### Polarization — The Two Input Channels

Sentinel-1 emits a vertically polarized wave (V) and receives responses for both V and H (horizontal). Hence, there are two input channels: **VV** (vertical transmitted and received) and **VH** (vertical transmitted, horizontal received).

They correspond to two different types of physical interactions:

* **VV**: Surface / specular scattering - it is highly sensitive to water and smooth surfaces in particular.
* **VH**: Volume scattering - it is particularly sensitive to forests and other forms of vegetation, since forests disturb polarization of reflected waves.

A flooded zone will appear dark in VV and a forest will be bright in VH compared to the VV channel. Consequently, **VV/VH ratio** (equal to `VV - VH` in dB) can be used as an analog of vegetation indices known from optical sensing (like NDVI) and thus serves as a third engineered feature channel. Thus, the total number of inputs equals three and I have an input tensor of size `[3, H, W]` per tile.

### Incidence Angle

The angle between the direction of the emitted radar pulse and the vertical axis is called an incidence angle. The range in which Sentinel-1 IW works spans approximately from 29° to 46°.

From an engineering point of view, incidence angle becomes relevant in machine learning problems due to the following reasons: (1) The same surface will appear differently in SAR images depending on the angle at which it is illuminated. (2) For a single image, the incidence angle varies from the left to the right edge. Nevertheless, for my case, it doesn't matter too much since preprocessed GRD products with topographic corrections were used.

### Speckle — The Graininess Problem

A speckle is a natural characteristic of radar images: even perfectly uniform surfaces will appear grainy in SAR imagery. Note that this is not a simple sensor noise, it is rather caused by coherent nature of radar imaging: for example, within a 10×10 m pixel, thousands of small scatterers (leaves, rocks, blades of grass) all emit their signals at once. As their signals combine coherently (depending on the phase difference), the overall signal is sometimes stronger or weaker than the average value.

For my problem, I am starting with unfiltered images and leave the responsibility for managing the speckle to the CNN (which I believe is capable of it).

### GRD vs. SLC

There are two popular types of SAR products discussed in the literature about Sentinel-1:

* **GRD (Ground Range Detected)**: Single-channel amplitude with terrain correction, multi-looks, and projection into geographic coordinates. It is exactly the one that should be used for classification tasks like mine (since they rely on backscatter).
* **SLC (Single Look Complex)**: Amplitude and phase information is provided; not multi-looked; slant-range geometry. Used for InSAR tasks.

## Application Areas

### Flood and Water Detection

This is the application that I will develop within this project. As opposed to other applications, the underlying physics make it particularly easy to develop: water has one of the lowest backscatter coefficients in C-band SAR (~−22 to −28 dB for the vertical polarization). This creates extremely high contrast with pretty much every other land use scenario. In a case of flooding, the rough land surfaces (agriculture fields, road surfaces, suburbs) become water, causing the corresponding backscatter drop by 10–15 dB. Huge signal!

Aside from the physical simplicity, another aspect that makes this problem so exciting is that flood occurs during rain, and thus in presence of cloud cover. Optical imagery simply won't work at such moments. SAR is thus the only sensor able to detect flood in close to real time manner.

The official dataset `Sen1Floods11` provides images for 11 major flood events worldwide using Sentinel-1 imagery and manually labeled water masks. It is the standard benchmark used in this problem domain, and what I will use in my experiments.

### Land Cover / Land Use Classification

The second interesting application is mapping the classes of land usage: urban areas, water, bare soil, vegetation, agriculture – again, all using SAR data. All these categories correspond to a unique combination of scatterers in each particular type of land cover. Adding to this the difference between VH and VV backscattering further increases discriminative capabilities.

As opposed to the previous application, the boundaries are less clear (vegetation and agriculture often have similar backscattering ranges), and there are seasonal effects that increase the noise. Using mean/std features based on multi-month SAR acquisitions significantly helps in such cases. Main datasets: BigEarthNet-S1 (590k patches, multi-label); SEN12MS (paired Sentinel-1 + Sentinel-2 + labels).

***

# ML Survey & Use Case Scoping

## Classical Approaches

Prior to the advent of deep learning, the standard approach was a SAR + ML pipeline involving handcrafted features followed by classification using a standard machine learning algorithm.

For **texture features + SVM/Random Forest**, the canonical classical pipeline relies heavily on texture information due to the fact that SAR pixels have no meaning analogous to RGB pixels. This includes using GLCM (Grey Level Co-occurrence Matrix) texture features, contrast, homogeneity, correlation, entropy within a fixed-sized sliding window (typically 9x9 or 11x11 pixels). This set of features together with raw SAR backscatter (VV, VH) and calculated indices (VV-VH ratio) is used as input for a Random Forest or Support Vector Machine classifier for land cover mapping. While yielding surprisingly high accuracy at approximately 78%, the technique is sensitive to changes between regions, as well as requiring tedious manual tuning of the handcrafted features.

The most basic approach for flood detection would be some form of **thresholding** on the SAR pixels. Given the low value for water backscatter in the VV band (roughly -22dB to -28dB), a global or adaptive (Otsu) threshold on the same can be used to generate the flood mask. Not only simple and interpretable, the approach remains very competitive, especially in permanent water detection as shown in the Sen1Floods11 paper.

## Deep Learning Approaches

| Model Type | Task | Notes |
|---|---|---|
| **FCN / UNet** | Flood segmentation on SAR | Standard pixel-wise segmentation; UNet backbone is most common due to skip connections recovering spatial detail lost during downsampling |
| **DeepLabv3+** | SAR land-cover / PolSAR scene segmentation | Pre-trained DeepLabv3+ outperforms RF and SVM on LULC with transfer learning, reaching ~87.78% pixel accuracy vs ~77.91% for RF |
| **Bi-temporal UNet** | Change-based flood detection | Uses pre-flood vs during-flood SAR pairs; attention blocks fuse the temporal difference and improve IoU by ~6% over uni-temporal methods on Sen1Floods11 |
| **CNN feature extractor + classifier head** | SAR multi-label land cover | Used as a backbone on BigEarthNet-S1; straightforward fine-tuning target for the "advanced model" in this assignment |

The key insight from surveying these: **UNet is the default starting point for any SAR segmentation task**, and the flood detection task on Sen1Floods11 is well-studied enough that I can compare my numbers against known baselines without needing a separate held-out benchmark.

## Datasets

| Dataset | Task | Data Type | Size | Link |
|---|---|---|---|---|
| **Sen1Floods11** | Flood / water segmentation | Sentinel-1 GRD (VV, VH), 10 m | 4,831 chips × 512×512, 11 flood events | [github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) |
| **UrbanSARFloods** | Urban flood segmentation | Sentinel-1 SLC intensity + InSAR coherence | 8,879 chips, 18 events, 20 land cover classes | [github.com/jie666-6/UrbanSARFloods](https://github.com/jie666-6/UrbanSARFloods) |
| **BigEarthNet-S1** | Multi-label land cover classification | Sentinel-1 GRD, ~549k patches | ~549k patches, 43 CORINE classes | [bigearth.net](https://bigearth.net) |
| **SEN12MS** | Land cover segmentation / scene classification | Sentinel-1 (VV/VH) + Sentinel-2 + MODIS labels | 180,662 triplets, global, all seasons | [zenodo.org/record/3653150](https://zenodo.org/record/3653150) |

## Foundation Models Worth Knowing

**Prithvi (IBM + NASA, 2023)**: A ViT-based geospatial foundation model pre-trained with masked autoencoding on 1 TB+ of Harmonized Landsat-Sentinel-2 (HLS) data. It's multispectral-optical, *not* SAR-native, but it has been fine-tuned for flood mapping on Sen1Floods11 and outperforms SatMAE on that task. Available on HuggingFace at [huggingface.co/ibm-nasa-geospatial](https://huggingface.co/ibm-nasa-geospatial). The limitation for this project: since it was pre-trained on optical bands (not SAR backscatter), using it would require treating SAR channels as pseudo-optical, which isn't ideal but could still work as a warm-start encoder.

**RingMo (BAAI, 2022)**: A remote sensing foundation model pre-trained via masked image modeling on 2M optical satellite images. A SAR-adapted variant (RingMo-SAM) extends it to PolSAR and SAR segmentation using LoRA fine-tuning with modality-specific decoders. More directly SAR-aware than Prithvi but also heavier to set up.

**SAR-JEPA / SARATR-X (2024)**: Purpose-built SAR foundation models using self-supervised learning (joint-embedding predictive architecture) on 180k+ unlabeled SAR target patches. They're designed for SAR ATR (automatic target recognition: vehicles, ships, aircraft) rather than flood segmentation, so they're less directly applicable to my use case, but they represent the frontier of SAR-specific pre-training. Code at [github.com/waterdisappear/SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA).

## Chosen Use Case: Flood / Water Detection on Sen1Floods11

### Why This Task

Three reasons this is the right call:

1. **Physics-backed signal**: Water = very dark VV backscatter. The signal-to-noise ratio for classification is large, which means a simple model can already work, and improvements are clearly attributable to the model rather than lucky data.
2. **Ready-made benchmark**: Sen1Floods11 is public, download-ready, and pre-chipped into 512×512 tiles with hand labels. No custom labeling needed.
3. **Known SOTA**: Published baselines exist, so I can honestly report where my model lands in the field rather than inventing evaluation criteria.

### My Specific Angle

To avoid being a straight tutorial re-implementation, I'm training on a **geographic subset** (Asia flood events only) and explicitly evaluating generalization to a held-out continent. This tests cross-regional transfer, a real operational concern for flood response systems and gives something to analyze rather than just reporting numbers.

### Planned Model Setup

- **Baseline:** Pixel-wise logistic regression on `[VV, VH, VV−VH]` per pixel equivalent to a classical thresholding approach with learned weights. Fast, interpretable, easy to beat.
- **Advanced model:** Lightweight UNet with a ResNet-18 encoder and standard decoder, using 3-channel SAR input `[VV, VH, VV−VH]`. ImageNet-pretrained encoder weights, fine-tuned end-to-end.
- **Evaluation:** IoU (intersection over union) for the flood class and F1 score, the standard metrics for Sen1Floods11, enabling direct comparison to published results.

## Summary: Where the Field Stands

The field has moved from texture+SVM (solid baseline, ~78% pixel accuracy) → lightweight UNets (reliable ~85%+ IoU on open water) → foundation model fine-tuning (marginal but meaningful gains, especially in data-scarce settings). My pipeline targets the middle of this arc — a UNet that can be trained and evaluated in 1–2 days — with a thoughtful framing around geographic generalization that adds something beyond the standard tutorial.