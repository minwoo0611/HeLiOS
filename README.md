# HeLiOS: Heterogeneous LiDAR Place Recognition via Overlap-based Learning and Local Spherical Transformer


**(Under Review) [IEEE ICRA 25]** This repository is the official repository for for **HeLiOS**.

  <a href="https://scholar.google.co.kr/citations?user=aKPTi7gAAAAJ&hl=ko" target="_blank">Minwoo Jung</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=I2pNZDkAAAAJ&hl=ko" target="_blank">Sangwoo Jung</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=n15gehEAAAAJ&hl=ko" target="_blank">Hyeonjae Gil</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=7yveufgAAAAJ&hl=ko" target="_blank">Ayoung Kim</a><sup>†</sup>

**[Robust Perception and Mobile Robotics Lab (RPM)](https://rpm.snu.ac.kr/)**

https://github.com/user-attachments/assets/7b741a6c-cb48-49d7-ae78-44b44c78e9ee

You can also check the video via [Youtube link](https://www.youtube.com/watch?v=tEEvWA3LyXY).

### Abstract
LiDAR place recognition is a crucial module in localization that matches the current location with previously observed environments. Most existing approaches in LiDAR place recognition dominantly focus on the spinning type LiDAR to exploit its large FOV for matching. However, with the recent emergence of various LiDAR types, the importance of matching data across different LiDAR types has grown significantly—a challenge that has been largely overlooked for many years. To address these challenges, we introduce HeLiOS, a deep network tailored for heterogeneous LiDAR place recognition, which utilizes small local windows with spherical transformers and optimal transport-based cluster assignment for robust global descriptors. Our overlap-based data mining and guided-triplet loss overcome the limitations of traditional distance-based mining and discrete class constraints. HeLiOS is validated on public datasets, demonstrating performance in heterogeneous LiDAR place recognition while including an evaluation for long-term recognition, showcasing its ability to handle unseen LiDAR types. We release the HeLiOS code as an open source for the robotics community.

### Comparison Methods
You can check the comparison methods and some results in [here](https://github.com/minwoo0611/HeLiPR-Place-Recognition)

Code for HeLiOS will be availble after review process.
