# Change Log

> In this work, an approach to detect human body joints with only little computation complexity will be researched with a special focus on figure ice skating. The resulting application should be able to run on mobile phones and detect body joints from just the camera image. 
>
> Therefore, the different implementations from OpenPose in C++, TensorFlow and PyTorch will be explored and inspected. These already include some experiments with different neuronal network architectures, however their keypoint detection amount varies as does their current work status or rather commit frequency. Furthermore, a new paper suggests ‘spatio temporal affinity fields’, which will be analyzed in this project.
>
> Different experiments and evaluations of the already existing research projects on keypoint detection will add to the final application. The training will be conducted on the provided DeepLearn server from the Stuttgart Media University. Therefore, the data from PoseTrack and the Coco dataset will be used.
>
> Since this research focuses on the detection of keypoints in figure ice skating, further considerations must be taken in mind: people in the background should not be detected, but only the person in focus for later analysis. The question there is, how to get the corespondent training data.

---

## 2019-12-03 [Status Report]: Analyze Figure Ice Skating Dataset

- find figure ice skating videos for labeling:
  - 332 MB of Video Data
  - filmed from different mobile devices and a GOPRO 5 Hero Black
  - videos of different length, perspectives, moving camera, different ice rinks and lightning conditions
- set-up Docker container on hal9k MI server: create Docker image
- create bash script to traverse all video files and label by OpenPose
- inspect labeled video quality
  - write script to detect on how many frames no person was detected and analyze these frames
    - create video data without these frames
      - analysis: on some videos on every frame a person was detected. No pattern for device type identifiable
      - videos still play fluently
    - detect the person in focus by taking the person with the highest score
  - check quality of videos by iterating over all videos, find subsequent frames with a minimum score
    - &rarr; This tests two conditions
      - whether from the subsequent frames, the keypoints with the maximum scores applied to these frames, would enhance the labels
      - if there can't be found subsequent frames with the specified score, the videos are marked as 'not usable labels'
    - manually analyze the images of the found subsequent frames
      - **Result**: sometimes very bad labeled frames, however they receive a relatively high score

---

## 2019-11-22 [Status Report]: Get Ready For Machine Learning

- Work over the book: [Deep Learning for Computer Vision By Rajalingappa Shanmugamani](https://learning.oreilly.com/library/view/deep-learning-for/9781788295628/)
  - book repeats several basics in a rough overview
  - many practical examples from different ML problems
- Practical pre-experiments
  - implement simple UNet with a pre-trained VGG-16 encoder
  - include tensorboard statistics for later evaluations

---

## 2019-11-17 [Status Report]: Planning
- create [initial project plan](https://docs.google.com/spreadsheets/d/1UiY_DN8u3v3q78M2Kt0J7INSgaR_6g2AVlFXYTfgT5g/edit#gid=0)
- write [project proposal](https://docs.google.com/document/d/1Yug-XjIxy3NjOoxbsPSLWGuDJFEE9NyPOWErCEk7S0I/edit)
- Difficulties in planning:
  - problems to setup OpenPose (mainly CMAKE)
    - only works with a reduced network on a usual laptop
    - not mobile ready
- Change focus of investigation *(included in project plan)*
  - investigations of the tensorflow and Pytorch github repositories
  resulted in the conclusion, to put the main focus on
  keypoint recognition with focus on performance,
  than the analysis of the curves
  - the Tensorflow and Pytorch repositories rely on the old *[initial](https://arxiv.org/pdf/1611.08050.pdf)* architecture of PAF *(no active commits to the repositories)*
    - make the training run, needed some tweaking in the code
    - the tensorflow implementation experiments with different pre-trained nets *(e.g. mobile-net versions)*
      - labeling works well on a usual laptop
  
---

## 2019-11-06 [Kickoff-Meeting]: Initial Research

- compare multiple paper - two main paper: 
  - [OpenPose: Whole-Body Pose Estimation](https://www.ri.cmu.edu/wp-content/uploads/2019/05/MS_Thesis___Gines_Hidalgo___latest_compressed.pdf)
    - network is trained in a single stage using multi-task learning
    - find keypoints with the help of Part Affinity Fields and Confidence Maps
  - [Efficient Online Multi-Person 2D Pose Tracking with
Recurrent Spatio-Temporal Affinity Fields](https://arxiv.org/pdf/1811.11975.pdf)
    - additionally to OpenPose, the temporal component is regarded

---
---
