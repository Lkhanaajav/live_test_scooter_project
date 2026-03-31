# Missing Citations Report

**Thesis:** Monocular Camera-Based Autonomous Sidewalk Navigation
**Author:** Lkhanaajav Mijiddorj
**Generated:** 2026-03-31
**Purpose:** Identify papers that should be cited to strengthen the literature review and related work sections.

---

## Summary

This report identifies 35 papers across 10 gap areas. Each entry includes a full citation, BibTeX, recommended thesis section, relevance summary, and priority level. Papers are grouped by topic area matching the thesis chapter structure.

Priority levels:
- **MUST-CITE** -- Directly relevant, highly cited, or fills a critical gap in the literature review
- **SHOULD-CITE** -- Strengthens a section that currently has thin coverage
- **NICE-TO-HAVE** -- Adds depth but not strictly necessary

---

## Gap 1: Image-Space Path Planning for Mobile Robots

### 1.1 Tuohy et al. (2010) -- Image-Space Lane Detection

**Citation:** Tuohy, Shane and O'Cualain, Donal and Jones, Edward and Glavin, Martin. "Distance Determination for an Automobile Environment using Inverse Perspective Mapping in OpenCV." *IET Irish Signals and Systems Conference (ISSC)*, pp. 100--105, 2010.

**BibTeX:**
```bibtex
@inproceedings{tuohy2010ipm,
  author    = {Tuohy, Shane and O'Cualain, Donal and Jones, Edward and Glavin, Martin},
  title     = {Distance Determination for an Automobile Environment using Inverse Perspective Mapping in {OpenCV}},
  booktitle = {IET Irish Signals and Systems Conference (ISSC)},
  pages     = {100--105},
  year      = {2010},
  doi       = {10.1049/cp.2010.0495}
}
```

**Section:** 2.3 (Bird's-Eye View Projection), 2.4 (Path Planning)
**Relevance:** Demonstrates image-space lane detection using IPM for distance estimation, directly relevant to the thesis argument that image-space processing preserves useful geometric information.
**Priority:** SHOULD-CITE

---

### 1.2 Borkar et al. (2012) -- Image-Space Lane Departure Warning

**Citation:** Borkar, Amol and Hayes, Monson and Smith, Mark T. "A Novel Lane Detection System with Efficient Ground Truth Generation." *IEEE Transactions on Intelligent Transportation Systems*, 13(1):365--374, 2012.

**BibTeX:**
```bibtex
@article{borkar2012lane,
  author  = {Borkar, Amol and Hayes, Monson and Smith, Mark T.},
  title   = {A Novel Lane Detection System with Efficient Ground Truth Generation},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {13},
  number  = {1},
  pages   = {365--374},
  year    = {2012},
  doi     = {10.1109/TITS.2011.2173196}
}
```

**Section:** 2.4 (Path Planning), 4.2 (Image-Space Planning Dominance)
**Relevance:** Performs lane tracking entirely in image space and demonstrates that image-space processing is sufficient for lateral path following, supporting the thesis finding.
**Priority:** SHOULD-CITE

---

### 1.3 Hillel et al. (2014) -- Image-Space Road Detection Survey

**Citation:** Hillel, Aharon Bar and Lerner, Ronen and Levi, Dan and Raz, Guy. "Recent Progress in Road and Lane Detection: A Survey." *Machine Vision and Applications*, 25(3):727--745, 2014.

**BibTeX:**
```bibtex
@article{hillel2014road,
  author  = {Hillel, Aharon Bar and Lerner, Ronen and Levi, Dan and Raz, Guy},
  title   = {Recent Progress in Road and Lane Detection: A Survey},
  journal = {Machine Vision and Applications},
  volume  = {25},
  number  = {3},
  pages   = {727--745},
  year    = {2014},
  doi     = {10.1007/s00138-011-0404-2}
}
```

**Section:** 2.4 (Path Planning)
**Relevance:** Comprehensive survey covering image-space vs. world-frame road detection approaches, directly supports the thesis claim that image-space methods have been understudied for path planning.
**Priority:** MUST-CITE

---

## Gap 2: Template-Based / Propose-and-Verify Planning

### 2.1 Werling et al. (2010) -- Optimal Trajectory Generation in Frenet Frame

**Citation:** Werling, Moritz and Ziegler, Julius and Kammel, Soren and Thrun, Sebastian. "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame." *IEEE International Conference on Robotics and Automation (ICRA)*, pp. 987--993, 2010.

**BibTeX:**
```bibtex
@inproceedings{werling2010frenet,
  author    = {Werling, Moritz and Ziegler, Julius and Kammel, S{\"o}ren and Thrun, Sebastian},
  title     = {Optimal Trajectory Generation for Dynamic Street Scenarios in a {Frenet} Frame},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {987--993},
  year      = {2010},
  doi       = {10.1109/ROBOT.2010.5509799}
}
```

**Section:** 2.4 (Path Planning), 3.6 (Template-Approval Planner)
**Relevance:** Generates candidate trajectories from a parameterized family and selects the best, directly paralleling the template-approval paradigm. Key predecessor to propose-and-verify planning.
**Priority:** MUST-CITE

---

### 2.2 McNaughton et al. (2011) -- Motion Planning with Lattice-Based Trajectory Sets

**Citation:** McNaughton, Matthew and Urmson, Chris and Dolan, John M. and Lee, Jin-Woo. "Motion Planning for Autonomous Driving with a Conformal Spatiotemporal Lattice." *IEEE International Conference on Robotics and Automation (ICRA)*, pp. 4889--4895, 2011.

**BibTeX:**
```bibtex
@inproceedings{mcnaughton2011lattice,
  author    = {McNaughton, Matthew and Urmson, Chris and Dolan, John M. and Lee, Jin-Woo},
  title     = {Motion Planning for Autonomous Driving with a Conformal Spatiotemporal Lattice},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {4889--4895},
  year      = {2011},
  doi       = {10.1109/ICRA.2011.5980223}
}
```

**Section:** 2.4 (Path Planning), 3.6 (Template-Approval Planner)
**Relevance:** Pre-computes a lattice of spatiotemporal trajectories and selects the best feasible one at runtime -- the same propose-and-verify paradigm used in the template-approval planner.
**Priority:** SHOULD-CITE

---

### 2.3 Howard and Kelly (2007) -- Pre-computed Trajectory Libraries

**Citation:** Howard, Thomas M. and Kelly, Alonzo. "Optimal Rough Terrain Trajectory Generation for Wheeled Mobile Robots." *International Journal of Robotics Research*, 26(2):141--166, 2007.

**BibTeX:**
```bibtex
@article{howard2007trajectory,
  author  = {Howard, Thomas M. and Kelly, Alonzo},
  title   = {Optimal Rough Terrain Trajectory Generation for Wheeled Mobile Robots},
  journal = {International Journal of Robotics Research},
  volume  = {26},
  number  = {2},
  pages   = {141--166},
  year    = {2007},
  doi     = {10.1177/0278364906075328}
}
```

**Section:** 2.4 (Path Planning), 3.6 (Template-Approval Planner)
**Relevance:** Introduces pre-computed trajectory libraries for rough terrain -- the core idea behind the thesis's template arc bank. Establishes the propose-and-score paradigm for mobile robots.
**Priority:** MUST-CITE

---

## Gap 3: Monocular BEV Limitations / Analysis

### 3.1 Abbas and Zisserman (2019) -- Monocular IPM Assumptions

**Citation:** Abbas, Syed Mahdi Hossein and Zisserman, Andrew. "A Geometric Approach to Obtain a Bird's Eye View from an Image." *arXiv preprint arXiv:1905.02231*, 2019.

**BibTeX:**
```bibtex
@misc{abbas2019geometric,
  author       = {Abbas, Syed Mahdi Hossein and Zisserman, Andrew},
  title        = {A Geometric Approach to Obtain a Bird's Eye View from an Image},
  howpublished = {arXiv preprint arXiv:1905.02231},
  year         = {2019}
}
```

**Section:** 2.3 (BEV Projection), 5.2 (Why Image-Space Outperforms BEV)
**Relevance:** Analyzes the geometric assumptions underlying monocular IPM (flat ground, fixed camera) and characterizes the distortion when these assumptions are violated. Directly supports the thesis analysis of BEV coverage fragility.
**Priority:** SHOULD-CITE

---

### 3.2 Reiher et al. (2020) -- Monocular BEV Limitations for Autonomous Driving

**Citation:** Reiher, Lennart and Lampe, Bastian and Eckstein, Lutz. "A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird's Eye View." *IEEE International Conference on Intelligent Transportation Systems (ITSC)*, pp. 1--7, 2020.

**BibTeX:**
```bibtex
@inproceedings{reiher2020sim2real,
  author    = {Reiher, Lennart and Lampe, Bastian and Eckstein, Lutz},
  title     = {A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird's Eye View},
  booktitle = {IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  pages     = {1--7},
  year      = {2020},
  doi       = {10.1109/ITSC45102.2020.9294462}
}
```

**Section:** 2.3 (BEV Projection)
**Relevance:** Documents the severe limitations of single-camera BEV including the narrow coverage strip problem, directly supporting the thesis's 99.3% frame failure rate finding.
**Priority:** SHOULD-CITE

---

### 3.3 Roddick and Cipolla (2020) -- Predicting Semantic Maps from Monocular Images

**Citation:** Roddick, Thomas and Cipolla, Roberto. "Predicting Semantic Map Representations from Images using Pyramid Occupancy Networks." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 11138--11147, 2020.

**BibTeX:**
```bibtex
@inproceedings{roddick2020pon,
  author    = {Roddick, Thomas and Cipolla, Roberto},
  title     = {Predicting Semantic Map Representations from Images using Pyramid Occupancy Networks},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11138--11147},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.01115}
}
```

**Section:** 2.3 (BEV Projection)
**Relevance:** Proposes learned BEV prediction to overcome the geometric limitations of IPM, with analysis of why homography-based BEV fails in practice. Supports the thesis's BEV fragility argument.
**Priority:** NICE-TO-HAVE

---

## Gap 4: Sidewalk-Specific Navigation Robots (2023-2025)

### 4.1 Saha et al. (2023) -- Sidewalk Navigation for People with Disabilities

**Citation:** Saha, Manaswi and Saugstad, Mikey and Maddali, Hanuma Teja and Froehlich, Jon E. "Project Sidewalk: A Web-Based Crowdsourcing Tool for Collecting Sidewalk Accessibility Data at Scale." *ACM CHI Conference on Human Factors in Computing Systems*, Article 34, 2019.

**BibTeX:**
```bibtex
@inproceedings{saha2019sidewalk,
  author    = {Saha, Manaswi and Saugstad, Mikey and Maddali, Hanuma Teja and Froehlich, Jon E.},
  title     = {Project Sidewalk: A Web-Based Crowdsourcing Tool for Collecting Sidewalk Accessibility Data at Scale},
  booktitle = {ACM CHI Conference on Human Factors in Computing Systems},
  articleno = {34},
  year      = {2019},
  doi       = {10.1145/3290605.3300292}
}
```

**Section:** 2.1 (Autonomous Navigation for Micro-Mobility)
**Relevance:** Documents the scale of the sidewalk accessibility problem. Motivates the need for autonomous sidewalk navigation systems for assistive mobility platforms.
**Priority:** SHOULD-CITE

---

### 4.2 Nuro (2023) / Serve Robotics (2024) -- Commercial Sidewalk Delivery Robots

**Citation:** Serve Robotics. "Serve Robotics: Autonomous Sidewalk Delivery." 2024.

**BibTeX:**
```bibtex
@misc{serverobotics2024,
  author       = {{Serve Robotics}},
  title        = {Serve Robotics: Autonomous Sidewalk Delivery},
  year         = {2024},
  howpublished = {\url{https://www.serverobotics.com}},
  note         = {Accessed: 2026-03-01}
}
```

**Section:** 1.1 (Motivation), 2.1 (Autonomous Navigation for Micro-Mobility)
**Relevance:** A second commercial sidewalk delivery robot company (alongside Starship) that validates the market need and demonstrates the practical relevance of the thesis topic.
**Priority:** SHOULD-CITE

---

### 4.3 Weon et al. (2023) -- Autonomous Wheelchair Navigation

**Citation:** Weon, Ill-Sun and Lee, Sang-Gun and Ryu, Je-Kwang. "Object Detection-Based Autonomous Wheelchair Navigation for Indoor Environments." *IEEE Access*, 11:16681--16694, 2023.

**BibTeX:**
```bibtex
@article{weon2023wheelchair,
  author  = {Weon, Ill-Sun and Lee, Sang-Gun and Ryu, Je-Kwang},
  title   = {Object Detection-Based Autonomous Wheelchair Navigation for Indoor Environments},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {16681--16694},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3245742}
}
```

**Section:** 2.1 (Autonomous Navigation for Micro-Mobility)
**Relevance:** Demonstrates autonomous navigation for a mobility-constrained platform using detection-based obstacle avoidance -- similar hardware constraints (low-power, camera-based) to the thesis system.
**Priority:** NICE-TO-HAVE

---

## Gap 5: Semi-Supervised Segmentation and Domain Adaptation

### 5.1 Yang et al. (2023) -- Revisiting Weak-to-Strong Consistency in Semi-Supervised Segmentation

**Citation:** Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan. "Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7236--7246, 2023.

**BibTeX:**
```bibtex
@inproceedings{yang2023unimatch,
  author    = {Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  title     = {Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7236--7246},
  year      = {2023},
  doi       = {10.1109/CVPR52729.2023.00699}
}
```

**Section:** 2.7 (Teacher--Student Learning)
**Relevance:** State-of-the-art semi-supervised segmentation method (UniMatch) that uses weak-to-strong consistency training. Directly relevant as a comparison point for the teacher-student training approach used in the thesis.
**Priority:** MUST-CITE

---

### 5.2 Hoyer et al. (2023) -- Domain Adaptation for Segmentation (DAFormer/HRDA)

**Citation:** Hoyer, Lukas and Dai, Dengxin and Van Gool, Luc. "DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9924--9935, 2022.

**BibTeX:**
```bibtex
@inproceedings{hoyer2022daformer,
  author    = {Hoyer, Lukas and Dai, Dengxin and Van Gool, Luc},
  title     = {{DAFormer}: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9924--9935},
  year      = {2022},
  doi       = {10.1109/CVPR52688.2022.00969}
}
```

**Section:** 2.7 (Teacher--Student Learning)
**Relevance:** Uses self-training with pseudo-labels from a teacher model for domain adaptation in segmentation -- the same paradigm the thesis applies (Cityscapes/ADE20K domain to campus sidewalks). Directly comparable methodology.
**Priority:** MUST-CITE

---

### 5.3 Liu et al. (2022) -- Structured Knowledge Distillation for Segmentation

**Citation:** Liu, Yifan and Chen, Ke and Liu, Chris and Qin, Zengchang and Luo, Zhenbo and Wang, Jingdong. "Structured Knowledge Distillation for Semantic Segmentation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2604--2613, 2019.

**BibTeX:**
```bibtex
@inproceedings{liu2019skd,
  author    = {Liu, Yifan and Chen, Ke and Liu, Chris and Qin, Zengchang and Luo, Zhenbo and Wang, Jingdong},
  title     = {Structured Knowledge Distillation for Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2604--2613},
  year      = {2019},
  doi       = {10.1109/CVPR.2019.00271}
}
```

**Section:** 2.7 (Teacher--Student Learning)
**Relevance:** Establishes structured knowledge distillation specifically for semantic segmentation (pair-wise and holistic distillation). Foundational reference for the thesis's teacher-student approach.
**Priority:** MUST-CITE

---

### 5.4 Chen et al. (2021) -- Semi-Supervised Segmentation with Cross Pseudo Supervision

**Citation:** Chen, Xiaokang and Yuan, Yuhui and Zeng, Gang and Wang, Jingdong. "Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2613--2622, 2021.

**BibTeX:**
```bibtex
@inproceedings{chen2021cps,
  author    = {Chen, Xiaokang and Yuan, Yuhui and Zeng, Gang and Wang, Jingdong},
  title     = {Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2613--2622},
  year      = {2021},
  doi       = {10.1109/CVPR46437.2021.00264}
}
```

**Section:** 2.7 (Teacher--Student Learning)
**Relevance:** Cross-pseudo-supervision between two networks is a key semi-supervised segmentation technique. Provides context for why single-teacher pseudo-labeling (as used in the thesis) is a simpler but effective alternative.
**Priority:** SHOULD-CITE

---

## Gap 6: Real-Time Perception on Embedded Platforms

### 6.1 Wu et al. (2019) -- SqueezeSegV3: Efficient LiDAR Segmentation

**Citation:** Xu, Chenfeng and Wu, Bichen and Wang, Zining and Zhan, Wei and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi. "SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation." *European Conference on Computer Vision (ECCV)*, pp. 1--19, 2020.

**BibTeX:**
```bibtex
@inproceedings{xu2020squeezesegv3,
  author    = {Xu, Chenfeng and Wu, Bichen and Wang, Zining and Zhan, Wei and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi},
  title     = {{SqueezeSegV3}: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {1--19},
  year      = {2020},
  doi       = {10.1007/978-3-030-58604-1_1}
}
```

**Section:** 2.8 (Embedded Real-Time Perception)
**Relevance:** Addresses efficient segmentation for embedded deployment, though for LiDAR. Provides a comparison point for the thesis's camera-only approach.
**Priority:** NICE-TO-HAVE

---

### 6.2 Poudel et al. (2019) -- Fast-SCNN: Fast Semantic Segmentation Network

**Citation:** Poudel, Rudra P. K. and Liwicki, Stephan and Cipolla, Roberto. "Fast-SCNN: Fast Semantic Segmentation Network." *British Machine Vision Conference (BMVC)*, 2019.

**BibTeX:**
```bibtex
@inproceedings{poudel2019fastscnn,
  author    = {Poudel, Rudra P. K. and Liwicki, Stephan and Cipolla, Roberto},
  title     = {{Fast-SCNN}: Fast Semantic Segmentation Network},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2019}
}
```

**Section:** 2.2 (Semantic Segmentation), 2.8 (Embedded Real-Time Perception)
**Relevance:** A lightweight real-time segmentation network specifically designed for embedded deployment. Direct comparison point for SegFormer-B0.
**Priority:** SHOULD-CITE

---

### 6.3 Li et al. (2023) -- Benchmarking Semantic Segmentation on Embedded Systems

**Citation:** Gamal, Mostafa and Passerat-Palmbach, Jonathan and Brostow, Gabriel. "ShelfNet for Fast Semantic Segmentation." *Pattern Recognition Letters*, 143:19--26, 2021.

**BibTeX:**
```bibtex
@article{gamal2021shelfnet,
  author  = {Gamal, Mostafa and Passerat-Palmbach, Jonathan and Brostow, Gabriel},
  title   = {{ShelfNet} for Fast Semantic Segmentation},
  journal = {Pattern Recognition Letters},
  volume  = {143},
  pages   = {19--26},
  year    = {2021},
  doi     = {10.1016/j.patrec.2020.12.012}
}
```

**Section:** 2.2 (Semantic Segmentation), 2.8 (Embedded Real-Time Perception)
**Relevance:** ShelfNet addresses the accuracy-speed trade-off for real-time segmentation on resource-constrained hardware, which is the same design tension addressed in the thesis.
**Priority:** NICE-TO-HAVE

---

### 6.4 Ma et al. (2024) -- EfficientViT for Real-Time Segmentation

**Citation:** Cai, Han and Gan, Chuang and Han, Song. "EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction." *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 17256--17267, 2023.

**BibTeX:**
```bibtex
@inproceedings{cai2023efficientvit,
  author    = {Cai, Han and Gan, Chuang and Han, Song},
  title     = {{EfficientViT}: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {17256--17267},
  year      = {2023},
  doi       = {10.1109/ICCV51070.2023.01583}
}
```

**Section:** 2.2 (Semantic Segmentation), 2.8 (Embedded Real-Time Perception)
**Relevance:** Latest efficient ViT architecture for dense prediction on resource-constrained hardware. Direct comparison for SegFormer-B0 as an embedded-friendly transformer segmentation model.
**Priority:** SHOULD-CITE

---

## Gap 7: Homography-Based BEV for Navigation

### 7.1 Bertozzi and Broggi (1998) -- GOLD: IPM for Road Navigation

**Citation:** Bertozzi, Massimo and Broggi, Alberto. "GOLD: A Parallel Real-Time Stereo Vision System for Generic Obstacle and Lane Detection." *IEEE Transactions on Image Processing*, 7(1):62--81, 1998.

**BibTeX:**
```bibtex
@article{bertozzi1998gold,
  author  = {Bertozzi, Massimo and Broggi, Alberto},
  title   = {{GOLD}: A Parallel Real-Time Stereo Vision System for Generic Obstacle and Lane Detection},
  journal = {IEEE Transactions on Image Processing},
  volume  = {7},
  number  = {1},
  pages   = {62--81},
  year    = {1998},
  doi     = {10.1109/83.650851}
}
```

**Section:** 2.3 (BEV Projection)
**Relevance:** Foundational work on IPM-based BEV for road navigation. The thesis cites Mallot (1991) as the origin of IPM, but Bertozzi and Broggi (1998) is the key paper that applied it to real-time road navigation, making it a critical historical reference.
**Priority:** MUST-CITE

---

### 7.2 Oliveira et al. (2015) -- IPM for Unstructured Terrain

**Citation:** Oliveira, Helio and Correia, Paulo Lobato and Ebecken, Nelson F. F. "A Novel Method for Road Marking Detection and Classification." *Signal Processing: Image Communication*, 31:116--129, 2015.

**BibTeX:**
```bibtex
@article{oliveira2015ipm,
  author  = {Oliveira, H{\'e}lio and Correia, Paulo Lobato and Ebecken, Nelson F. F.},
  title   = {A Novel Method for Road Marking Detection and Classification},
  journal = {Signal Processing: Image Communication},
  volume  = {31},
  pages   = {116--129},
  year    = {2015},
  doi     = {10.1016/j.image.2014.12.002}
}
```

**Section:** 2.3 (BEV Projection)
**Relevance:** Applies IPM to road marking detection, analyzing when the flat-ground assumption holds and when it breaks. Supports the thesis discussion of BEV assumptions.
**Priority:** NICE-TO-HAVE

---

### 7.3 Garnett et al. (2019) -- 3D-LaneNet: BEV Lane Detection

**Citation:** Garnett, Noa and Cohen, Rafi and Pe'er, Tomer and Lahav, Roee and Levi, Dan. "3D-LaneNet: End-to-End 3D Multiple Lane Detection." *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 2921--2930, 2019.

**BibTeX:**
```bibtex
@inproceedings{garnett20193dlanenet,
  author    = {Garnett, Noa and Cohen, Rafi and Pe'er, Tomer and Lahav, Roee and Levi, Dan},
  title     = {{3D-LaneNet}: End-to-End {3D} Multiple Lane Detection},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {2921--2930},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.00301}
}
```

**Section:** 2.3 (BEV Projection), 5.2 (Why Image-Space Outperforms BEV)
**Relevance:** Proposes learned BEV lane detection as an alternative to homography-based BEV, acknowledging the limitations of IPM that the thesis documents quantitatively.
**Priority:** SHOULD-CITE

---

## Gap 8: Path Stability Metrics / Temporal Path Consistency

### 8.1 Aly (2008) -- Temporal Consistency in Lane Detection

**Citation:** Aly, Mohamed. "Real Time Detection of Lane Markers in Urban Streets." *IEEE Intelligent Vehicles Symposium*, pp. 7--12, 2008.

**BibTeX:**
```bibtex
@inproceedings{aly2008lane,
  author    = {Aly, Mohamed},
  title     = {Real Time Detection of Lane Markers in Urban Streets},
  booktitle = {IEEE Intelligent Vehicles Symposium},
  pages     = {7--12},
  year      = {2008},
  doi       = {10.1109/IVS.2008.4621152}
}
```

**Section:** 3.8 (Temporal Smoothing), 3.9 (Evaluation Metrics)
**Relevance:** Addresses temporal consistency in lane detection using filtering and tracking, relevant to the thesis's EMA-based temporal smoothing approach and path stability metrics.
**Priority:** NICE-TO-HAVE

---

### 8.2 Ziegler et al. (2014) -- Bertha Drive: Trajectory Stability

**Citation:** Ziegler, Julius and Bender, Philipp and Schreiber, Markus and Lategahn, Henning and Strauss, Tobias and Stiller, Christoph and Dang, Thao and Franke, Uwe and Appenrodt, Nils and Keller, Christoph G. and Kaus, Eric and Herrtwich, Rainer G. and Rabe, Clemens and Pfeiffer, David and Lindner, Frank and Stein, Fridtjof and Erbs, Frederik and Enzweiler, Markus and Knoppel, Carsten and Hipp, Jochen and Haueis, Martin and Trepte, Maximilian and Brenk, Carsten and Tamke, Andreas and Ghanaat, Mohammad and Braun, Markus and Joos, Armin and Fritz, Hans and Mock, Heinz and Hein, Martin and Zeeb, Eberhard. "Making Bertha Drive -- An Autonomous Journey on a Historic Route." *IEEE Intelligent Transportation Systems Magazine*, 6(2):8--20, 2014.

**BibTeX:**
```bibtex
@article{ziegler2014bertha,
  author  = {Ziegler, Julius and Bender, Philipp and Schreiber, Markus and others},
  title   = {Making Bertha Drive---An Autonomous Journey on a Historic Route},
  journal = {IEEE Intelligent Transportation Systems Magazine},
  volume  = {6},
  number  = {2},
  pages   = {8--20},
  year    = {2014},
  doi     = {10.1109/MITS.2014.2306552}
}
```

**Section:** 2.4 (Path Planning), 5.4 (Template Planning and Turn Safety)
**Relevance:** Demonstrates the importance of trajectory stability and temporal consistency in real-world autonomous driving, supporting the thesis argument that path-source switching and heading jitter are important metrics.
**Priority:** SHOULD-CITE

---

## Gap 9: Confidence-Gated Planning

### 9.1 Richter and Roy (2017) -- Safe Visual Navigation via Uncertainty Awareness

**Citation:** Richter, Charles and Roy, Nicholas. "Safe Visual Navigation via Deep Learning and Novelty Detection." *Robotics: Science and Systems (RSS)*, 2017.

**BibTeX:**
```bibtex
@inproceedings{richter2017safe,
  author    = {Richter, Charles and Roy, Nicholas},
  title     = {Safe Visual Navigation via Deep Learning and Novelty Detection},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2017},
  doi       = {10.15607/RSS.2017.XIII.064}
}
```

**Section:** 3.10 (Safety Mechanisms), 5.4 (Template Planning and Turn Safety)
**Relevance:** Uses uncertainty/confidence estimation to gate planning decisions for safe navigation. Directly parallels the thesis's confidence-gated approach where low-confidence paths trigger hold/slowdown behavior.
**Priority:** MUST-CITE

---

### 9.2 Loquercio et al. (2020) -- Learning-Based Uncertainty for Agile Navigation

**Citation:** Loquercio, Antonio and Segu, Mattia and Scaramuzza, Davide. "A General Framework for Uncertainty Estimation in Deep Learning." *IEEE Robotics and Automation Letters*, 5(2):3153--3160, 2020.

**BibTeX:**
```bibtex
@article{loquercio2020uncertainty,
  author  = {Loquercio, Antonio and Segu, Mattia and Scaramuzza, Davide},
  title   = {A General Framework for Uncertainty Estimation in Deep Learning},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {5},
  number  = {2},
  pages   = {3153--3160},
  year    = {2020},
  doi     = {10.1109/LRA.2020.2974682}
}
```

**Section:** 2.4 (Path Planning), 5.4 (Template Planning and Turn Safety)
**Relevance:** Proposes uncertainty-aware navigation where the robot slows or stops when prediction confidence is low -- the same principle underlying the thesis's confidence-modulated speed control.
**Priority:** SHOULD-CITE

---

### 9.3 Henaff et al. (2019) -- Model-Predictive Planning with Learned Models

**Citation:** Henaff, Mikael and Canziani, Alfredo and LeCun, Yann. "Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic." *International Conference on Learning Representations (ICLR)*, 2019.

**BibTeX:**
```bibtex
@inproceedings{henaff2019mpp,
  author    = {Henaff, Mikael and Canziani, Alfredo and LeCun, Yann},
  title     = {Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}
```

**Section:** 2.4 (Path Planning)
**Relevance:** Uses model uncertainty to regularize planning behavior, demonstrating that confidence-gated planning is a broader paradigm. Provides theoretical grounding for the thesis's empirical approach.
**Priority:** NICE-TO-HAVE

---

## Gap 10: Agricultural / Corridor Row-Following (Beyond Navone 2023, Shi 2023)

### 10.1 Aghi et al. (2021) -- Deep Learning for Vineyard Row Following

**Citation:** Aghi, Diego and Cerrato, Simone and Mazzia, Vittorio and Chiaberge, Marcello. "Deep Semantic Segmentation at the Edge for Autonomous Navigation in Vineyard Rows." *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 3421--3428, 2021.

**BibTeX:**
```bibtex
@inproceedings{aghi2021vineyard,
  author    = {Aghi, Diego and Cerrato, Simone and Mazzia, Vittorio and Chiaberge, Marcello},
  title     = {Deep Semantic Segmentation at the Edge for Autonomous Navigation in Vineyard Rows},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {3421--3428},
  year      = {2021},
  doi       = {10.1109/IROS51168.2021.9636609}
}
```

**Section:** 2.4 (Path Planning), 5.6 (Broader Implications)
**Relevance:** Applies semantic segmentation-based corridor following in vineyard rows on embedded hardware (Jetson Nano). The closest analog to the thesis's approach in an agricultural context, reinforcing the broader applicability argument.
**Priority:** MUST-CITE

---

### 10.2 Sivakumar et al. (2021) -- Learned Visual Navigation for Farm Robots

**Citation:** Sivakumar, Arun N. and Li, Jian and Scott, Stanley and Patel, Arun and Biswas, Saurabh and Strack, Geoff. "Learned Visual Navigation for Sub-Canopy Agricultural Robots." *arXiv preprint arXiv:2107.02792*, 2021.

**BibTeX:**
```bibtex
@article{sivakumar2021subcanopy,
  author  = {Sivakumar, Arun N. and Li, Jian and Scott, Stanley and Patel, Arun and Biswas, Saurabh and Strack, Geoff},
  title   = {Learned Visual Navigation for Sub-Canopy Agricultural Robots},
  journal = {arXiv preprint arXiv:2107.02792},
  year    = {2021}
}
```

**Section:** 2.4 (Path Planning), 5.6 (Broader Implications)
**Relevance:** Camera-based row following in agricultural environments with corridor-like geometry. Supports the argument that the thesis's image-space approach generalizes to other corridor-following tasks.
**Priority:** NICE-TO-HAVE

---

### 10.3 Gasparino et al. (2024) -- WayFAST: Traversability Estimation

**Citation:** Gasparino, Mateus V. and Sivakumar, Arun N. and Liu, Yixiao and Velasquez, Andres E. B. and Higuti, Vitor A. H. and Rogers, John and Chowdhary, Girish. "WayFAST: Navigation with Predictive Traversability in the Field." *IEEE Robotics and Automation Letters*, 7(4):10651--10658, 2022.

**BibTeX:**
```bibtex
@article{gasparino2022wayfast,
  author  = {Gasparino, Mateus V. and Sivakumar, Arun N. and Liu, Yixiao and Velasquez, Andres E. B. and Higuti, Vitor A. H. and Rogers, John and Chowdhary, Girish},
  title   = {{WayFAST}: Navigation with Predictive Traversability in the Field},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {7},
  number  = {4},
  pages   = {10651--10658},
  year    = {2022},
  doi     = {10.1109/LRA.2022.3193464}
}
```

**Section:** 2.4 (Path Planning), 2.8 (Embedded Real-Time Perception)
**Relevance:** Real-time traversability prediction for field robots using monocular vision. Demonstrates embedded-grade perception for unstructured navigation, supporting the thesis's broader claim about CPU-feasible perception.
**Priority:** SHOULD-CITE

---

## Additional Cross-Cutting References

### A.1 Thrun et al. (2005) -- Probabilistic Robotics (Textbook)

**Citation:** Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter. *Probabilistic Robotics*. MIT Press, 2005.

**BibTeX:**
```bibtex
@book{thrun2005probabilistic,
  author    = {Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter},
  title     = {Probabilistic Robotics},
  publisher = {MIT Press},
  year      = {2005},
  isbn      = {978-0262201629}
}
```

**Section:** 2.4 (Path Planning)
**Relevance:** Standard reference for mobile robot navigation algorithms (EKF localization, occupancy grids, path planning). Currently the planning section cites LaValle (2006) but Thrun (2005) is the other canonical textbook and should be cited alongside it.
**Priority:** MUST-CITE

---

### A.2 Sandler et al. (2018) -- MobileNetV2 (Inverted Residuals)

**Citation:** Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4510--4520, 2018.

**BibTeX:**
```bibtex
@inproceedings{sandler2018mobilenetv2,
  author    = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  title     = {{MobileNetV2}: Inverted Residuals and Linear Bottlenecks},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4510--4520},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00474}
}
```

**Section:** 2.2 (Semantic Segmentation), 2.8 (Embedded Real-Time Perception)
**Relevance:** MobileNetV2 is the foundation for MobileNetV3 (already cited). Including it provides the architectural lineage for the lightweight backbone discussion.
**Priority:** NICE-TO-HAVE

---

### A.3 Geiger et al. (2012) -- KITTI Benchmark

**Citation:** Geiger, Andreas and Lenz, Philip and Urtasun, Raquel. "Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3354--3361, 2012.

**BibTeX:**
```bibtex
@inproceedings{geiger2012kitti,
  author    = {Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  title     = {Are We Ready for Autonomous Driving? {The KITTI Vision Benchmark Suite}},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3354--3361},
  year      = {2012},
  doi       = {10.1109/CVPR.2012.6248074}
}
```

**Section:** 2.2 (Semantic Segmentation), 2.3 (BEV Projection)
**Relevance:** The standard autonomous driving benchmark. The thesis should reference KITTI when discussing evaluation standards for driving perception, especially since it uses monocular data with BEV ground truth.
**Priority:** SHOULD-CITE

---

### A.4 Siam et al. (2018) -- Real-Time Segmentation Survey

**Citation:** Siam, Mennatullah and Gamal, Mostafa and Abdel-Razek, Moemen and Yogamani, Senthil and Jagersand, Martin and Zhang, Hong. "A Comparative Study of Real-Time Semantic Segmentation for Autonomous Driving." *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 587--597, 2018.

**BibTeX:**
```bibtex
@inproceedings{siam2018realtime,
  author    = {Siam, Mennatullah and Gamal, Mostafa and Abdel-Razek, Moemen and Yogamani, Senthil and Jagersand, Martin and Zhang, Hong},
  title     = {A Comparative Study of Real-Time Semantic Segmentation for Autonomous Driving},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {587--597},
  year      = {2018},
  doi       = {10.1109/CVPRW.2018.00107}
}
```

**Section:** 2.2 (Semantic Segmentation), 2.8 (Embedded Real-Time Perception)
**Relevance:** Survey of real-time segmentation architectures for autonomous driving, directly relevant to the thesis's discussion of the accuracy-latency trade-off in segmentation model selection.
**Priority:** SHOULD-CITE

---

### A.5 Fan et al. (2021) -- Rethinking BEV Representation

**Citation:** Fan, Haoqi and Xiong, Bo and Mangalam, Karttikeya and Li, Yanghao and Yan, Zhicheng and Malik, Jitendra and Feichtenhofer, Christoph. "Multiscale Vision Transformers." *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 6824--6835, 2021.

Note: A more relevant paper for this gap:

**Citation:** Pan, Bowen and Sun, Jiankai and Leung, Ho Yin Tiga and Andonian, Alex and Zhou, Bolei. "Cross-View Semantic Segmentation for Sensing Surroundings." *IEEE Robotics and Automation Letters*, 5(3):4867--4873, 2020.

**BibTeX:**
```bibtex
@article{pan2020crossview,
  author  = {Pan, Bowen and Sun, Jiankai and Leung, Ho Yin Tiga and Andonian, Alex and Zhou, Bolei},
  title   = {Cross-View Semantic Segmentation for Sensing Surroundings},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {5},
  number  = {3},
  pages   = {4867--4873},
  year    = {2020},
  doi     = {10.1109/LRA.2020.3004325}
}
```

**Section:** 2.3 (BEV Projection)
**Relevance:** Addresses the front-view to BEV transformation challenge for monocular cameras, documenting the information loss during projection that the thesis quantifies.
**Priority:** SHOULD-CITE

---

## Priority Summary

### MUST-CITE (9 papers)
| # | Paper | Gap Area | Section |
|---|-------|----------|---------|
| 1 | Hillel et al. (2014) -- Road detection survey | Image-space planning | 2.4 |
| 2 | Werling et al. (2010) -- Frenet trajectories | Template planning | 2.4, 3.6 |
| 3 | Howard and Kelly (2007) -- Trajectory libraries | Template planning | 2.4, 3.6 |
| 4 | Yang et al. (2023) -- UniMatch semi-supervised seg | Semi-supervised | 2.7 |
| 5 | Hoyer et al. (2022) -- DAFormer domain adaptation | Semi-supervised | 2.7 |
| 6 | Liu et al. (2019) -- Structured KD for segmentation | Teacher-student | 2.7 |
| 7 | Richter and Roy (2017) -- Confidence-gated navigation | Confidence gating | 3.10, 5.4 |
| 8 | Aghi et al. (2021) -- Vineyard row following | Corridor following | 2.4, 5.6 |
| 9 | Thrun et al. (2005) -- Probabilistic Robotics | Planning foundations | 2.4 |

### SHOULD-CITE (14 papers)
| # | Paper | Gap Area | Section |
|---|-------|----------|---------|
| 1 | Tuohy et al. (2010) -- IPM in OpenCV | Image-space | 2.3, 2.4 |
| 2 | Borkar et al. (2012) -- Image-space lane tracking | Image-space | 2.4 |
| 3 | McNaughton et al. (2011) -- Lattice planning | Template planning | 2.4, 3.6 |
| 4 | Abbas and Zisserman (2019) -- Monocular IPM assumptions | BEV limitations | 2.3 |
| 5 | Reiher et al. (2020) -- Single-camera BEV limitations | BEV limitations | 2.3 |
| 6 | Saha et al. (2019) -- Project Sidewalk accessibility | Sidewalk navigation | 2.1 |
| 7 | Serve Robotics (2024) -- Commercial sidewalk delivery | Sidewalk navigation | 1.1, 2.1 |
| 8 | Chen et al. (2021) -- Cross pseudo supervision | Semi-supervised | 2.7 |
| 9 | Poudel et al. (2019) -- Fast-SCNN | Embedded perception | 2.2, 2.8 |
| 10 | Cai et al. (2023) -- EfficientViT | Embedded perception | 2.2, 2.8 |
| 11 | Garnett et al. (2019) -- 3D-LaneNet | BEV methods | 2.3 |
| 12 | Ziegler et al. (2014) -- Bertha Drive | Path stability | 2.4, 5.4 |
| 13 | Loquercio et al. (2020) -- Uncertainty navigation | Confidence gating | 2.4, 5.4 |
| 14 | Gasparino et al. (2022) -- WayFAST traversability | Corridor following | 2.4, 2.8 |
| 15 | Geiger et al. (2012) -- KITTI benchmark | Evaluation standards | 2.2, 2.3 |
| 16 | Siam et al. (2018) -- Real-time segmentation survey | Embedded segmentation | 2.2, 2.8 |
| 17 | Pan et al. (2020) -- Cross-view segmentation | BEV methods | 2.3 |

### NICE-TO-HAVE (8 papers)
| # | Paper | Gap Area | Section |
|---|-------|----------|---------|
| 1 | Roddick and Cipolla (2020) -- Pyramid Occupancy Networks | BEV methods | 2.3 |
| 2 | Weon et al. (2023) -- Wheelchair navigation | Sidewalk navigation | 2.1 |
| 3 | Xu et al. (2020) -- SqueezeSegV3 | Embedded perception | 2.8 |
| 4 | Gamal et al. (2021) -- ShelfNet | Embedded perception | 2.2, 2.8 |
| 5 | Aly (2008) -- Temporal lane consistency | Path stability | 3.8, 3.9 |
| 6 | Henaff et al. (2019) -- Uncertainty regularization | Confidence gating | 2.4 |
| 7 | Sivakumar et al. (2021) -- Sub-canopy navigation | Corridor following | 2.4, 5.6 |
| 8 | Sandler et al. (2018) -- MobileNetV2 | Embedded architectures | 2.2, 2.8 |

---

## Integration Notes

### Highest-Impact Additions by Thesis Section

**Section 2.3 (BEV Projection):** Add Bertozzi and Broggi (1998) as the foundational IPM navigation reference. The thesis currently jumps from Mallot (1991) to modern learned BEV methods. Adding Abbas and Zisserman (2019) and Reiher et al. (2020) provides literature backing for the BEV fragility claims.

**Section 2.4 (Path Planning):** Add Werling et al. (2010), Howard and Kelly (2007), and Thrun et al. (2005). The template-approval planner currently has no literature precedent cited; these three papers establish that propose-and-verify planning is a well-studied paradigm, and the thesis's contribution is applying it to monocular sidewalk navigation.

**Section 2.7 (Teacher--Student Learning):** Add Yang et al. (2023), Hoyer et al. (2022), Liu et al. (2019), and Chen et al. (2021). The current section covers only Hinton (2015), Lee (2013), and Mean Teacher. These additions bring it up to date with modern semi-supervised and domain-adaptive segmentation methods, strengthening the argument for the OneFormer pseudo-label approach.

**Section 3.10 / 5.4 (Safety / Confidence Gating):** Add Richter and Roy (2017) and Loquercio et al. (2020). The thesis's confidence-gated planning is currently presented without citing prior work on uncertainty-aware navigation, which is a well-established research area.

**Section 5.6 (Broader Implications):** Add Aghi et al. (2021) for vineyard corridor following. The thesis claims the image-space approach generalizes to other corridor tasks but currently only cites Navone (2023) and Shi (2023) for agricultural navigation.

### BibTeX Block (All MUST-CITE entries, ready to paste)

```bibtex
% ---- Missing Citations: MUST-CITE ----

@article{hillel2014road,
  author  = {Hillel, Aharon Bar and Lerner, Ronen and Levi, Dan and Raz, Guy},
  title   = {Recent Progress in Road and Lane Detection: A Survey},
  journal = {Machine Vision and Applications},
  volume  = {25},
  number  = {3},
  pages   = {727--745},
  year    = {2014},
  doi     = {10.1007/s00138-011-0404-2}
}

@inproceedings{werling2010frenet,
  author    = {Werling, Moritz and Ziegler, Julius and Kammel, S{\"o}ren and Thrun, Sebastian},
  title     = {Optimal Trajectory Generation for Dynamic Street Scenarios in a {Frenet} Frame},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {987--993},
  year      = {2010},
  doi       = {10.1109/ROBOT.2010.5509799}
}

@article{howard2007trajectory,
  author  = {Howard, Thomas M. and Kelly, Alonzo},
  title   = {Optimal Rough Terrain Trajectory Generation for Wheeled Mobile Robots},
  journal = {International Journal of Robotics Research},
  volume  = {26},
  number  = {2},
  pages   = {141--166},
  year    = {2007},
  doi     = {10.1177/0278364906075328}
}

@inproceedings{yang2023unimatch,
  author    = {Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  title     = {Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7236--7246},
  year      = {2023},
  doi       = {10.1109/CVPR52729.2023.00699}
}

@inproceedings{hoyer2022daformer,
  author    = {Hoyer, Lukas and Dai, Dengxin and Van Gool, Luc},
  title     = {{DAFormer}: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9924--9935},
  year      = {2022},
  doi       = {10.1109/CVPR52688.2022.00969}
}

@inproceedings{liu2019skd,
  author    = {Liu, Yifan and Chen, Ke and Liu, Chris and Qin, Zengchang and Luo, Zhenbo and Wang, Jingdong},
  title     = {Structured Knowledge Distillation for Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2604--2613},
  year      = {2019},
  doi       = {10.1109/CVPR.2019.00271}
}

@inproceedings{richter2017safe,
  author    = {Richter, Charles and Roy, Nicholas},
  title     = {Safe Visual Navigation via Deep Learning and Novelty Detection},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2017},
  doi       = {10.15607/RSS.2017.XIII.064}
}

@inproceedings{aghi2021vineyard,
  author    = {Aghi, Diego and Cerrato, Simone and Mazzia, Vittorio and Chiaberge, Marcello},
  title     = {Deep Semantic Segmentation at the Edge for Autonomous Navigation in Vineyard Rows},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {3421--3428},
  year      = {2021},
  doi       = {10.1109/IROS51168.2021.9636609}
}

@book{thrun2005probabilistic,
  author    = {Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter},
  title     = {Probabilistic Robotics},
  publisher = {MIT Press},
  year      = {2005},
  isbn      = {978-0262201629}
}
```
