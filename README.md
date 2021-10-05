# Quantiative Gaze Metric

The purpose of this code is to develop a quantiative method of evaluating gaze models in 2d and, eventually, 3d.

Choosing a dataset:

From the [Papers with Code Gaze Estimation Page](https://paperswithcode.com/task/gaze-estimation):

MPI-Gaze: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild
- Easily downloaded from homepage
- Problem: Images are all eye-crops

EYEDIAP: https://www.idiap.ch/en/dataset/eyediap
- Requested Dataset Access: https://zenodo.org/record/4467455

RT-GENE: https://zenodo.org/record/2529036
- Access is here: https://zenodo.org/record/2529036
- "The dataset consists of two parts: 1) One where the eyetracking glasses were worn (and thus ground truth labels for head-pose and eye gaze are available; suffix _glasses), and 2) One with natural appearances (no eyetracking glasses are worn; suffix _noglasses). The _noglasses images were used to train subject-specific GANs, and these GANs were used to inpaint the region covered by the eyetracking glasses in the _glasses images."

UT Multi-view: https://www.ut-vision.org/datasets/
- Access is here: https://www.ut-vision.org/datasets/
- Problem: Images are all eye-crops

From [Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://arxiv.org/pdf/2104.12668.pdf):

MPIIFaceGaze: https://perceptualui.org/research/datasets/MPIIFaceGaze/
- Easily downloadable from homepage
- 