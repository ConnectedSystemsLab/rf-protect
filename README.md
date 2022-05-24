# RF-Protect: Privacy against Device-Free Human Tracking

RF-Protect is a a new framework that enables privacy by injecting fake humans in the sensed data. RF-Protect consists of a novel hardware reflector design that modifies radio waves to create reflections at arbitrary locations in the environment and a new generative mechanism to create realistic human trajectories. 

RF-Protect was accepted by SIGCOMM'22. If you find RF-Protect useful, we kindly ask you to cite the paper.

***

### Radar Dataset ([Dowload link](https://drive.google.com/file/d/1GF-jGDjKdpDDQ_BMC43pyN3AZ75rqkRC/view?usp=sharing))

**Explain:**

- This dataset collected from our FMCW radar equiped with USRP X310 and is comprised of raw signals. While Radar is sensing the environment, we turn on our RF-Protect platform to generate additional signals which will be shown as human phantoms on the radar heatmaps.

**Composition**:

- Radar dataset consists of three folders: `trajs_gt`,  `./trajs_from_usrp_Home` and `./trajs_from_usrp_Office`. The first one is the ground truth generated from our Trajectory GAN. The latter two represent two datasets collected from Home and Office environment with radar respectively.
- Each folder contains the .dat files, among which the _0.dat is the radar signal and _1.dat is used for reference signal

**Usage:**

- Download and unzip, put it in the main directory

***

### Trajectory Extraction

Giving the radar dataset, we can then generate FMCW radar heatmaps with peaks representing the location of the phantom human. By conducting peak detection and signal processing across all heatmaps, we can form the trajectories of this phantom. To do so, run the following matlab script:

```
./Trace/Trace_processing.m
```

To reproduce the trajectory accuracy figure shown as Fig.11 in the paper, run the following matlab script:

```
./Trace/get_CDF.m
```

