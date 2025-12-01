# HeLiOS🌞: Heterogeneous LiDAR Place Recognition via Overlap-based Learning and Local Spherical Transformer

**(Accepted) [IEEE ICRA 25]** This repository is the official repository for **HeLiOS🌞** **[[Paper]](https://minwoo0611.github.io/publications/icra2025-helios.pdf)** **[[Video]](https://www.youtube.com/watch?v=tEEvWA3LyXY)**.

  <a href="https://scholar.google.co.kr/citations?user=aKPTi7gAAAAJ&hl=ko" target="_blank">Minwoo Jung</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=I2pNZDkAAAAJ&hl=ko" target="_blank">Sangwoo Jung</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=n15gehEAAAAJ&hl=ko" target="_blank">Hyeonjae Gil</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=7yveufgAAAAJ&hl=ko" target="_blank">Ayoung Kim</a><sup>†</sup>

**[Robust Perception and Mobile Robotics Lab (RPM)](https://rpm.snu.ac.kr/)**

### Recent Updates
- [2025/05/10] First release of HeLiOS code.

### Contributions
Our work makes the following contributions:
1. **Tackles heterogeneous LiDAR place recognition**: Matches point clouds across diverse LiDAR types with varying fields of view, scanning patterns, and resolutions, addressing a gap in traditional spinning LiDAR-focused methods.
2. **Innovative deep learning approach**: Employs small spherical transformer windows and optimal transport-based clustering to generate robust global descriptors, enhanced by overlap-based data mining and guided-triplet loss.
3. **Superior performance and open-source**: Outperforms existing methods in heterogeneous and long-term place recognition on public datasets, with code available post-review for robotics community integration.
![distribution_hetero](https://github.com/user-attachments/assets/caa27998-9174-454f-be95-65c9827376e9)

### Comparison Methods
You can check the comparison methods and some results in [here](https://github.com/minwoo0611/HeLiPR-Place-Recognition)

Descriptions for HeLiOS will be availble soon.

### Usage

This section outlines how to set up and run HeLiOS using a Docker container. The code has been tested on NVIDIA RTX 3090 and 3080 GPUs.

#### 1. Build the Docker Image

To begin, build the Docker image using the provided Dockerfile and start a container with GPU support.

Build the image:

```
sudo docker build -t helios:github .
```

Run the container:

```
docker run --gpus all -dit --env="DISPLAY" --net=host --ipc=host \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v /:/mydata --volume /dev/:/dev/ \
    helios:github /bin/bash
```

**Note**: Adjust the docker run command to suit your environment.

#### 2. Clone the Repository

Clone the HeLiOS repository and enter to the project directory:

```
git clone https://github.com/minwoo0611/HeLiOS
cd HeLiOS
```

#### 3. Pre-process Data for Training and Evaluation

Prepare the data by following these steps. Alternatively, you can download pre-processed outputs for steps 3.1 to 3.5 from [Google Drive](https://drive.google.com/drive/folders/1b2HFFEKnPkcqjnwOQbTxoGJHfxrb3KwS?usp=sharing). The overlap file for the training dataset includes full sequences of DCC04-06, KAIST04-06, and Riverside04-06 for Aeva, Avia, Ouster, and Velodyne. If you wish to test custom settings, please follow the below pipelines.

#### 3-1 Download the HeLiPR Dataset

Visit the [HeLiPR dataset](https://sites.google.com/view/heliprdataset) and save the point cloud and ground truth files to `/data/RAW`. These files will be used for training and testing.

#### 3-2 Process Raw Point Clouds

Use the [HeLiPR-Pointcloud-Toolbox](https://github.com/minwoo0611/HeLiPR-Pointcloud-Toolbox) to process raw point clouds. The configuration file is located at `/dataset/preprocess/helipr_toolbox_config.yaml`. Save processed files to the training folder.

#### 3-3 Extract Timestamp and Location

Extract timestamps and locations from ground truth files in `/RAW/(sequence-sensor)` for the processed `.bin` files:

The key variables are:
- `local_base_dir`: Directory for training folder (Default: `../../data/training/`)
- `global_base_dir`: Directory for RAW folder (Default: `../../data/Raw/`)
- `datasets`: Sequences for training or evaluation (Default: `['DCC04', 'DCC05']`)
- `sensors`: Sensors for training or evaluation (Default: `['Aeva', 'Ouster', 'Avia', 'Velodyne']`)

The outputs are:
- `trajectory.csv`: The global trajectory of processed bin file
- `trajectory_transform.csv`: The local trajectory of processed bin file for 3-4

```
python3 timestamp_saver.py
```

#### 3-4 Transform Processed LiDAR Scans

Transform the processed scans to global coordinates for overlap calculation:

The key variables are:
- `local_base_dir`: Directory for training folder (Default: `../../data/training/`)
- `datasets`: Sequences for training or evaluation (Default: `['DCC04', 'DCC05']`)
- `sensors`: Sensors for training or evaluation (Default: `['Aeva', 'Ouster', 'Avia', 'Velodyne']`)

The output is:
- `LiDAR_transformed/`: The saving folder of transformed bin file

```
python3 LiDAR_transformer.py
```

#### 3-5 Calculate Overlap

Compute overlaps between scans. This step is time-consuming and best run overnight:

The key variables are (can be checked in config file):
- `base_path`: Directory for data root (Default: `../../../../data/`)
- `runs_folder`: Directory for validation folder (Default: `validation/`)
- `filename`: The global trajectory name saved in 3-3 (Default: `trajectory.csv`)
- `dir_txt`: The output file name (Default: `overlap_matrix_Roundabout.txt`)
- `folder_list`: The sequence lists utilized in overlap calculation (Default: `Roundabout01-Aeva, Roundabout01-Ouster, Roundabout01-Avia, ROundabout01-Velodyne`)

The output is:
- `overlap_matrix`: The overlap values between each scans

```
cd datasets/preprocess/calculate_overlap
mkdir build
cd build
cmake ..
make
./overlap
```

#### 3-6 Generate Pickle Files

Generate pickle files with positive, semi-positive, and negative point clouds for training or evaluation:

```
python3 generate_training_sets.py
python3 generate_test_sets.py
```

The resulting file structure should be:

```
data
├── overlap_matrix_training.txt
├── overlap_matrix_evaluation.txt
├── training.pickle
├── validation.pickle
├── evaluation_db.pickle
├── evaluation_query.pickle
├── RAW
│   └── Sequence-Sensor
│       ├── LiDAR
│       │   └── (Sensor)/time.bin
│       └── LiDAR_GT
│           ├── global_(Sensor)_gt.txt
│           └── (Sensor)_gt.txt
├── training
│   └── Sequence-Sensor
│       ├── LiDAR/(timestamp.bin)
│       ├── LiDAR_transformed/(timestamp.bin)
│       ├── trajectory.csv
│       └── trajectory-transform.csv
├── validation
│   └── (same as training)
```

#### 4. Run HeLiOS

#### 4-1 Training

HeLiOS provides two configurations: `helios` and `helios-s`. Choose the appropriate model configuration (`config/model_helios.txt` or `config/model_helios_s.txt`).

Train with the standard model:

```
python3 training/train.py --config config/config_baseline_helios.txt --model_config config/model_helios.txt
```

Train with the smaller model:

```
python3 training/train.py --config config/config_baseline_helios.txt --model_config config/model_helios_s.txt
```

#### 4-2 Pre-trained Models

Pre-trained models are available in the `weights` directory:
- `HeLiOS-S.pth`: Outputs a small-dimension descriptor.
- `HeLiOS.pth`: Outputs a large-dimension descriptor.

#### 4-3 Evaluation

Evaluate using pre-trained weights.

For the small model:

```
python3 eval/evaluate.py --config config/config_baseline_helios.txt --model_config config/model_helios.txt --weights weights/HeLiOS-S.pth
```

For the large model:

```
python3 eval/evaluate.py --config config/config_baseline_helios.txt --model_config config/model_helios.txt --weights weights/HeLiOS.pth
```
**Note**: The result values slightly differ from those in the manuscript, but the difference is only in the third decimal place.

### Acknowledgments

Our code is based on [MinkLoc3dv2](https://github.com/jac99/MinkLoc3Dv2) and [SphereFormer](https://github.com/dvlab-research/SphereFormer). If you use our code or dataset, please cite:

```
@INPROCEEDINGS { mwjung-2025-icra,
    AUTHOR = { Minwoo Jung and Sangwoo Jung and Hyeonjae Gil and Ayoung Kim },
    TITLE = { HeLiOS: Heterogeneous LiDAR Place Recognition via Overlap-based Learning and Local Spherical Transformer },
    BOOKTITLE = { Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) },
    YEAR = { 2025 },
    MONTH = { May. },
    ADDRESS = { Atlanta },
}

@article{jung2024helipr,
  title={HeLiPR: Heterogeneous LiDAR dataset for inter-LiDAR place recognition under spatiotemporal variations},
  author={Jung, Minwoo and Yang, Wooseong and Lee, Dongjae and Gil, Hyeonjae and Kim, Giseop and Kim, Ayoung},
  journal={The International Journal of Robotics Research},
  volume={43},
  number={12},
  pages={1867--1883},
  year={2024},
  publisher={SAGE Publications Sage UK: London, England}
}
```

### Contact

For questions, email [moonshot@snu.ac.kr](mailto:moonshot@snu.ac.kr) or create an issue in this repository.
