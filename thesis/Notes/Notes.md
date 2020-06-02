Dibra et al
**Fields**
* Medicine
    * Precise human pose estimation based on two-dimensional images for kinematic analysis
    https://spie.org/Publications/Proceedings/Paper/10.1117/12.2542539?SSO=1
    * 
    
41,30,35

# Human Action Recognition and Prediction:A Survey

**Real World Applications:**
- state of the art algorithms 
    - 26 “Temoral segment networks: Toward good practices for deep action recognition,” in ECCV, 2016.
    - 27 “Spatiotemporal multiplier networks for video action recognition,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017, pp. 7445–7454 
    - 28 “Action prediction from videos via memorizing hard-to-predict samples,” in AAAI, 2018.
    - 29 “Learning activity progression in lstms for activity detection and early detection,” in CVPR, 2016.
- surveillance, entertainment, robotics, autonomous driving

In some real-world scenarios (e.g. vehicle accidents and criminal activi-
ties), intelligent machines do not have the luxury of waiting for
the entire action execution before having to react to the action
contained in it. For example, being able to predict a dangerous
driving situation before it occurs; opposed to recognizing it
thereafte


    
    
# Synergetic Reconstruction from 2D Pose and 3D Motion for Wide-Space Multi-Person Video Motion Capture in the Wild
**various fields**
    - 41 Estimation of physically and physiologically valid somatosensory information. In IEEE International Conference on Robotics and Automation (ICRA), 2005.
    - 30 Musculoskeletal-see-through mirror: Computational modeling and algorithm for whole-body muscle activity visualization in real time. 
    Progress in Biophysics and Molecular Biology, 103(2):310–317, 2010. Special Issue on BiomechanicalModelling of Soft Tissue Motion.
    - 35 Synthesis of Whole Body Motion with Pose-Constraints from Stochastic Model. In IEEE International Conference on Robotics and Automation (ICRA), 2014
    
    
 - 36 Scanning 3D Full Human Bodies Using Kinects. IEEE Transactions on Visualization and Computer Graphics, 18(4):643–650, April 2012.
 
 
 
 # https://losspreventionmedia.com/using-artificial-intelligence-to-catch-shoplifters-in-the-act/
 - Vaak: Japaneese startup: catch shoplifters in the act - alert staff - cant leave store
    - how people walk, hand movements, facial expressions, clothing choices
- Xloong company: implement aufmented reality classes in chine --> cross-reference faces against national database - spot criminals
- Japanese company, the communications giant NTT East: 
    - AI Guardsman, a camera that uses similar technology to analyze shoppers’ body language for signs of possible theft
    
    
# See your mental state from your walk: Recognizing anxiety and depression through Kinect-recorded gait data
-  monitoring individual’s anxiety and depression based on the Kinect-recorded gait pattern
    - article https://theconversation.com/how-you-walk-could-be-used-to-identify-some-types-of-dementia-124023
    - https://www.bbc.com/future/article/20190116-the-invisible-warning-signs-that-predict-your-future-health
    
# china ai
- https://www.youtube.com/watch?v=Onm6Sb3Pb2Y
- https://www.handelszeitung.ch/digital-switzerland/software-durchschaut-ladendiebe-schon-vor-der-tat

# Real-world Anomaly Detection in Surveillance Videos
- https://arxiv.org/pdf/1801.04264.pdf

# Motion-Aware Feature for Improved Video Anomaly Detection
- outperform previous approaches by a large margin on both anomaly detection and anomalous action recognition tasks in the UCF Crime dataset


# 3DPeople Dataset: Modeling the Geometry of Dressed Humans (2019 Apr 9)
**dataset info**
- synthetic dataset 
- 2.5 mio frames - 80 subjects (40/40 m/w) - 70 different actions
- distinct body shapes, skin tones & clothing outfits 
- 640x480 rgb images 4 viewports
- textrue of clothes, lightning direction, backround image - randomly changed
- 3d geometry of body clothing
- 3d skeletons
- depth maps, optical flow, semantic info (body parts, cloth labels)
**paper**
- paper: use dataset to model geometry of dressed humans 
-> predict geometry from single images - end2end deep generative network for predicting shape (GimNet)
- area-preserving parameterization algorithm based on the optimal mass transportation method
- dataset
--> good results on synthetic images and on the wild
**dataset**
- body models: adobe fuse + makehuman (skin tones, body shapes, hair geometry)
- clothing models: variety of gaments, tight/ loose clothes, sunglasses, hats, caps (final rigged ~ 20K vertices)
- Mocap sequences: 70 realistic from Mixamo https://www.mixamo.com/#/?page=1&query=ballett&type=Motion%2CMotionPack mean length: 110 frames
    - short sequence --> large expressivity, drinking (small movements) - breakdance/ backflip (complex movements)
- Textures/ camera/lights/background: 
    - blender: apply mocap animation
    - 22,400 clips
    - projective camera with 800 nm focal length
    - viewpoints: othogonal directions aligned with ground
    - different distances to person --> ensure full view of body in all frames
    - illumination: ambient lightning + light source at infinite
**geometry image for reference mesh**
- repair the mesh:
    - voxelization, selecttion of largest connected region of alpha shape & subsequent hole filling using medial axis approach 
        ref. --> J. B. A. Sinha and K. Ramani. Deep learning 3D Shape Surfaces using Geometry Images. In ECCV, 2016.
**GimNet**
- extract 2d joint locations p - represent as heatmaps

# predict the 3D position of the body joints
- J. Martinez, R. Hossain, J. Romero, and J. Little. **A simple yet effective baseline for 3d human pose estimation**. In ICCV, 2017
- D. Mehta, S. Sridhar, O. Sotnychenko, H. Rhodin, M. Shafiei, H.-P. Seidel, W. Xu, D. Casas, and C. Theobalt.**VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera. ACM Transactions on Graphics**, 36(4), 2017
- F. Moreno-Noguer. **3D Human Pose Estimation from a Single Image via Distance Matrix Regression**. In CVPR, 2017.
- G. Pavlakos, X. Z. and. K. G. Derpanis, and K. Daniilidis.Coarse-to-fine **volumetric prediction for single-image 3D human pose**. In CVPR, 2017
- G. Rogez and C. Schmid. **MoCap-guided Data Augmentation for 3D Pose Estimation in the Wild**. In NIPS, 2016
- X. Sun, B. Xiao, S. Liang, and Y. Wei. **Integral Human Pose Regression.** In ECCV, 2018.
- G. Varol, J. Romero, X. Martin, N. Mahmood, M. J. Black,I. Laptev, and C. Schmid. **Learning from synthetic humans**.In CVPR, 2017
- W. Yang, W. Ouyang, X. Wang, J. Rena, H. Li, , and X. Wang. **3d human pose estimation in the wild by adversarial learning**. In CVPR, 2018

# Human3.6M Dataset
C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu.
Human3.6M: Large Scale Datasets and Predictive Methods
for 3D Human Sensing in Natural Environments. PAMI,
36(7):1325–1339, 2014

# HumanEva Dataset
L. Sigal, A. O. Balan, and M. J. Black. HumanEva: Synchro-
nized Video and Motion Capture Dataset and Baseline Algo-
rithm for Evaluation of Articulated Human Motion. IJCV,
87(1-2), 2010.

# SURREAL datset 2017
largest & more complete dataset so far, with more than 6 million frames generated by projecting synthetic textures of clothes onto random SMPL body shapes. 
+ annotated with body masks, optical flow and depth
G. Varol, J. Romero, X. Martin, N. Mahmood, M. J. Black,,I. Laptev, and C. Schmid. Learning from synthetic humans.
In CVPR, 2017

# Synergetic Reconstruction from 2D Pose and 3D Motion for Wide-Space Multi-Person Video Motion Capture in the Wild (2020-01-16)
- motion-capture method with spatio-temporal accuracy & smoothness from multiple cams in wide & multi-person environments
- predict 3d pose of each person + determine bounding box of multi cam img
- 3d reconstruction --> predict bounding box of each cam img in next frame = feedback from 3d motion to 2d pose
**motion capture methods**
- optical: reflective markers attached to characteristic parts of the body - 3d pose are measured
    - http://www.motionanalysis.com, VICON Corporation. http://www.vicon.com/
- inertial: use IMU sensors attached to body parts --> positions are calculated using sensor speed
    - Xsens Technologies. http://www.xsens.com/
    - Noitom Ltd. http://neuronmocap.com/.
- markerless: depth camera or single/multiple RGB video cams 32, 36, 3, 4
    - RADiCAL. http://getrad.co/
    - The Captury. http://www.thecaptury.com
    - J. Shotton, A. Fitzgibbon, M. Cook, T. Sharp, M. Finocchio, R. Moore, A. Kipman, and A. Blake. **Real-time Human Pose Recognition in Parts from Single Depth Images**.In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011
    - J. Tong, J. Zhou, L. Liu, Z. Pan, and H. Yan. **Scanning 3D Full Human Bodies Using Kinects**. IEEE Transactions on Visualization and Computer Graphics, 18(4):643–650, April 2012
    
- only in closed environments (not concerts, on sport fields, streets)
- capture in real world -difficulties-
    - multiple persons --> occlusion -> identification/ tracking
    - large measurement field --> greater calibration errors
    - real environment: no additional sensors, lightning/ desired sensor position
 **paper**
- 3D human motion reconstruction with spatiotemporal accuracy and smoothness even in a challenging multi-person environment, by extending the single-person video motion capture method
    - T. Ohashi, Y. Ikegami, K. Yamamoto, W. Takano, and Y.Nakamura. **Video Motion Capture from the Part Confidence Maps of Multi-Camera Images by Spatiotemporal Filtering Using the Human Skeletal Model**. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018.
- synchronized multiple calibrated cameras to record video images of human subjects from different directions
- human skeletal model for reconstructing 3D motion by spatiotemporal filtering of joint movements
- with bounding box - kp positions of each subject in imgs estimated via top-down pose estimation approach
    --> received as part confidence maps (PCM)
- PCM of multi-cam img + predicted 3d pose -> probable kp pos can be calculated
**single-view pose estimation** 15, 40, 34]
- top-down: detect pos of muliple persons in img as bounding box --> then estimate kp positionof persin in cropped img
    - K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask R-CNN. In IEEE/CVF International Conference on ComputerVision (ICCV), 2017.
    - Y. Chen, Z. Wang, Y. Peng, Z. Zhang, G. Yu, and J. Sun. Cascaded Pyramid Network for Multi-Person Pose Estimation.In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018
    - B. Xiao, H. Wu, and Y. Wei. Simple Baselines for Human Pose Estimation and Tracking. In European Conference on Computer Vision (ECCV), 2018
    - K. Sun, B. Xiao, D. Liu, and J. Wang. Deep High-Resolution Representation Learning for Human Pose Estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019
- bottom-up: first estimate kp position of all persons from entire image then associate to each person [38, 14, 24]
    - S.-E. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh. Convolutional pose machines. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016
    - Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017
    - S. Kreiss, L. Bertoni, and A. Alahi. PifPaf: Composite Fields for Human Pose Estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019
--> top down: more accurate; bottom-up = faster
--> top down relies on human detection results = problem with many occlusions
- estimate 3d poses from single img
    - detect 2d keypoint po to 3d space [29, 26, 8]
    - directly estimate 3d pose [27, 12, 23, 45]
    - estimate pose + body shapes [39, 19]
    --> ill-posed problem (many assumptions) --> better: use multiple cams
**Synergetic reconstruction**
- multiple cams around subjects with multiple fields of view at one location
- kps are estimated top-down with HPNet & data received as PCM
    - B. Xiao, H. Wu, and Y. Wei. Simple Baselines for Human Pose Estimation and Tracking. In European Conference on Computer Vision (ECCV), 2018
    - K. Sun, B. Xiao, D. Liu, and J. Wang. Deep High-Resolution Representation Learning for Human Pose Esti-mation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 201


# 3D human pose estimation in video with temporal convolutions and semi-supervised training (2019-03-29 Facebook)
- has applications in many fields like sports and health care
- perform exercise therapy on their own, without physiotherapist get feedback, no money
- avoid scoring scandals
    - Wikipedia. List of olympic games scandals and controversies
    — M.-J. Borzilleri. Olympic figure skating controversy: Judging system is most to blame for uproar, 2014. 

# Learning To Score Olympic Events 2017


# Learning to Score Figure Skating Sport Videos 2018
- Self-Attentive LSTM and Multi-scale Convolutional Skip LSTM --> learn the local and global sequential information in each video
- FisV dataset: 500 figure skating videos with the average length of 2 minutes and 50 seconds.
    - annotated by two scores of nine different referees, i.e., Total Element Score(TES) and Total Program Component Score (PCS)
    - validated on FisV and MIT-skate datasets
    - 500 videos of 149 professional figure skating players from more than 20 different countries.
    - each video - one skater + whole performance
- 2 comlementary subnetworks:
    - Self-Attentive LSTM (S-LSTM)
        - select important clip features which are directly used for regression tasks
        - learns to represent the local information
        - can efficiently learn to model the local sequential information by a self-attentive strategy.
    - Multi-scale Convolutional Skip LSTM (M-LSTM)
        - models local and global sequential information at multi-scale
        - save computational cost by skipping some video features
**Related Work**
- Video Representation
    - improving video representations fo-cuses on local motion features such as HOF [26][24] andMBH [5] in the Dense Trajectories feature [15] and the corre-sponding variants [52].
    - videos can be naturally considered as an ensemble of spatial and temporal component
        - Simonyan and Zisserman introduced a two-stream framework:
            - learn spatial & temporal feature representations concurrently with 2 convolutional networks
- Video Fusion
    - efficiently exploit the relationships of features
    - 2 types of feature fusion
        - early fusion
        - late fusion
    - multiple kernel learning: estimate fusion weights (needed in both types)
- Sports Video Analysis
    - sports video analysis has been tropical in the research communities (Zach Lowe. Lights, cameras, revolution. Grantland, March, 2013.)
    - action/ short sequence of actions
        - various works - how well people perform actions:
            - e.g. automated video assessment: analyzes videos of gymnasts performing the vault (A.S Gordon. Automated video assessment of human performance. In AI-ED, 1995.)
            - basketball team: based on trajectories of all players: (Marko Jug, Janez Perš, Branko Dežman, and Stanislav Kovačič. Trajectory based assessment of coordinated human activity. In International Conference on Computer Vision Systems, pages 534–543. Springer,2003.)
            - multi-player basketball activity - trajectory based evaluation using bayesian network (M. Perse, M. Kristan, J. Pers, and S Kovacic. Automatic evaluation of organized basketball activity using bayesian networks. In Citeseer, 2007)
            - machine learning classifier on top of a rule-based algorithm to recognize on-ball screens (A. McQueen, J. Wiens, and J. Guttag. Automatically recognizing on-ball screens. In MIT Sloan Sports Analytics Conference (SSAC), 2014.)
        - learn to score sports - only few studies: 
            - H. Pirsiavash, C. Vondrick, and Torralba. Assessing the quality of actions. In ECCV, 2014.
                - learning-based framework evaluating on two distinct types of actions (diving and figure skating)
                - regression model from spatiotemporal pose features to scores obtained from expert judges
            - Paritosh Parmar and Brendan Tran Morris. Learning to score olympic events. In Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on, pages 76–84. IEEE, 2017.
                - Support Vector Regression (SVR) and Long Short-Term Memory (LSTM) on C3D features of videos to obtain scores on the same dataset
            - (both) regression model is learned from the features of video clips/actions to the sport scores
            - their (paper) model is capable of modeling the nature of figure skating "model both the local and global sequential information which is essential in modeling the TES and PCS."
                - alleviate the problem that figure skating videos are too long for an ordinary LSTM to get processed
**Dataset**:
- NHK Trophy (NHK), Trophee Eric Bompard (TEB), Cup of China (COC), Four Continents Figure Skating Championships (4CC)
    - previous datasets (e.g., UCF 101[47], HMDB51 [25], Sports-1M [21] and ActivityNet[13]),
    - constructed by searching and downloaded from various search engines (e.g., Google, Flickr and Bing, etc) or the social media sharing platforms (e.g. Youtube, DailyMotion, etc.).
- better and more consistent visual quality of our TV videos from the high standard international competitions than those consumer videos downloaded from the Internet.
- maintain standard & authorized scoring, we select the videos only from the highest level of international competitions with fair and reasonable judgement
- ISU Championships, ISU Grand Prix of Figure Skating and Winter Olympic Games.
- only the competition videos about ladies’ singles short program happened over the past ten years are utilized (problem rating system has changed over the past 10 years)
- Data Analysis:
    - compute the Spearman correlation and Kendall tau correlation between TES and PCS over different matches: independent, weak correlations
    
- FIS-V vs MIT-skate:
    - larger data scale
    - higher annotation scores (TES & PCS not just total)
    - videos come from 12 competitions from 2012 to 2017 (fis more recent videos, MIT before 2012)

**Pre-processing**
- 100h videos
- manually processed: cut redundant info/ clips (2.50) 4300 frames frame rate 25


**Problem setup**:
- weakly labeled regression - no scores for every movement PCS
- Video Features:
    - deep spatial-temporal convolution networks
    - extract deep clip-level features off-the-shelf from 3D Convolutional Networks
    - pre-trained on Sports-1M [22], which is a large-scale dataset containing 1,133,158 videos which have been annotated automatically with 487 sports labels
    - sliding window of size 16 frames over the video temporal to cut the video clips with the stride as 8
    
**Model**:
- S-LSTM
    1) The features of clips that are important to difficulty technical movements should be heavy weighted.
    2) The produced compact feature representations should be the fixed length for all the videos.
- Multi-scale Convolutional Skip LSTM (M-LSTM)
    - essential to model the sequential frames/clips containing the local (technical movements) and global (performance of players)
        - dense clip-based C3D video features give the good representations of local sequential information
        - skip redundant info
        - combine local/ global info
**Experiments/ Resulsts**:
- Results of the spearman correlation
    - decent results - outperform some baselines
- Results of the mean square error


# Figure Skating Simulation from Video 2019
- demonstrate figure skating motions with a physically simulated human-like character
- difficult to obtain reference motions from figure skaters because figure skating motions are very fast and dynamic
- use key poses extracted from videos on YouTube and complete reference motions using trajectory optimization
- generate a robust controller for figure skating skills

- Wang et al. [WFH10] did simulate walking motion on ice-like ground by employing a lower friction coefficient to demonstrate the robustness of their controller, research on real skating motion has not been conducted
- Skaterbot [GPD ∗ 18] esigning and optimizing tools for skating robots, but uses wheels not blades & no ice environment
- professional figure skaters can represent dynamic moves, such as jumps, artistically

- motion capture - not possible -> observed videos -> obtain key poses: 3d pose from an image
    - key pose from video: not exact global position - no time & space coherence
    - apply the trajectory optimization method suggested by Al Borno et al. [ABDLH13]
    - reinforcement learning: find robust controller for each skating skill
- severlal skating motions: crossover, three-turn, jump + create short program

-implement a skate blade-ice contact model using non-holonomic constraints
- succeeded in generating figure skating motions without motion capture data.

- pose estimation to achieve naturalness and used high-level objectives for each motion for trajectory optimization.
- from images: 3D joint positions for key poses are extracted using HMR
- did not work well for estimating acrobatic poses
- 3D pose estimation from a single 2D image is not accurate
- failed to generate spin because of the friction issues

- ice environment, we set the friction coefficient of the ground to 0.02 -> Dart (opensource engine for simulation)

- a successful simulated motion, we applied a trajectory optimization framework [ABDLH13] based on CMAES 2d ->3d

**Related Work**:
- Recently, Peng et al. [PKM ∗ 18] successfully extracted referencemotion from videos. They used pose estimation techniques from images [WRKS16, KBJM18]

use figure skating videos on YouTube to obtain key pose data. In
this study, we choose key pose images from video and use a frame-
work by Kanazawa et al. [KBJM18] to obtain a 3D character pose.
After that, we use a trajectory optimization method proposed by Al
Borno et al. [ABDLH13] to generate reference motion

















# FSD-10: A Dataset for Competitive Sports Content Analysis (ice skating) (2020 Feb 09)
- 1484 clips 30fps; 1080x720
- worldwide figure skating championships in 2017-2018 80h (3-30 sec)
- consist of 10 different actions in men/ladies programs
- clips annotated by type GOE, skater info
- KTSN = Keyframe based temporal segment network for classification
- moving cam - ensure skater in each frame

- 2A in two seconds (has image)


- sports content analysis (SCA) - many enterprises eg Bloomberg, SAP, Panasonic
    https://www.bloomberg.com/company/press/bloomberg-sports-launches-stats-insights-sports-analysis-blog/
    https://www.svgeurope.org/blog/headlines/video-based-sports-analytics-from-sap-and-panasonic-announced-at-ibc/
- problems with current datasets:
    - do not meet complexity of competetive sports (daily actions)
    - action depends on static human pose & background
    - video segmentation rarely discussed in current ds
    


# Continuous Video to Simple Signals for Swimming Stroke Detection with Convolutional Neural Networks
In many sports, it is useful to analyse video of an athlete in competition for training purposes. 
In swimming, stroke rate is a common metric used by coaches; requiring a laborious labelling of each individual stroke. 
We show that using a Convolutional Neural Network (CNN) we can automatically detect discrete events in continuous video (in this case, swimming strokes). 
We create a CNN that learns a mapping from a window of frames to a point on a smooth 1D target signal, with peaks denoting the location of a stroke, evaluated as a sliding window. 
To our knowledge this process of training and utilizing a CNN has not been investigated before; either in sports or fundamental computer vision research. 
Most research has been focused on action recognition and using it to classify many clips in continuous video for action localisation. 
In this paper we demonstrate our process works well on the task of detecting swimming strokes in the wild. 
However, without modifying the model architecture or training method, the process is also shown to work equally well on detecting tennis strokes, implying that this is a general process. 
The outputs of our system are surprisingly smooth signals that predict an arbitrary event at least as accurately as humans (manually evaluated from a sample of negative results). 
A number of different architectures are evaluated, pertaining to slightly different problem formulations and signal targets.


# Computer Vision in Sports (ice hockey, swimming)
- https://towardsdatascience.com/computer-vision-in-sports-61195342bcef
# Hockey Action Recognition via Integrated Stacked Hourglass Network



# A 2019 Guide to Human Pose Estimation
- https://www.kdnuggets.com/2019/08/2019-guide-human-pose-estimation.html



# 3D Human Pose Estimation in RGBD Images for Robotic Task Learning 2018
- https://lmb.informatik.uni-freiburg.de/Publications/2018/ZB18/paper-Rgb-Pose3D-Task-Learning.pdf


# 3D Human Pose Estimation: A Review of the Literature and Analysis of Covariates 2016
https://www.researchgate.net/publication/307905073_3D_Human_Pose_Estimation_A_Review_of_the_Literature_and_Analysis_of_Covariates

# Multi-person 3D Pose Estimation and Tracking in Sports


//# Let’s Dance: Learning From Online Dance Videos

# https://www.stuttgarter-zeitung.de/inhalt.expertin-bei-eislaufmeisterschaften-drei-entscheidende-minuten.3b370419-3df9-4ea7-81ba-97d64480e78f.html




# Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (2017)
- efficiently detect 2d pose of multiple people in image
- mutliple people in image --> interaction, occlusion challenge
- PAF: encode location & orientation of limbs
    - S: confidence maps for body part locations (keypoints)
    - L set of 2d vector fields
    - each ps 2d vector
    - multi stage 2 branches
    
- Machines, endowed with such perception in realtime, would be able to react to and even participate in the individual and social behavior of people.


# OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (may 2019)
- runtime improvements
- dataset with body & foot kp detector


# STAF Efficient Online Multi-Person 2D Pose Tracking with Recurrent Spatio-Temporal Affinity Fields (12.06.2019)
https://cmu-perceptual-computing-lab.github.io/spatio-temporal-affinity-fields/
**general**
- detect & track people at one time
- predict spatio temporal fields accros a video sequence
- ingect heatmaps from previos frames & estimate for currenct frame
- only onlline inference & tracking (currently fastest & most bottom up approach, runtime invariant to number of people in scene & accuracy invariant to inpute frame rate)
- highly competetive results on PoseTrack benchmarks

- --> truly online, real-time multi-person 2D pose estimator + tracker --> deployable & scalable while achieving high performance + requiring minimal post-processing
- handle large camera motion and motion blur across frames
- Temporal Affinity Fields (TAFs) which encode connections between keypoints across frames:
    - unique cross-linked limb topology
    - information preservation even, if person stops moving, new person comes/ leaves scene
        - zero motion --> TAF becomes PAF
    - learns to propagate PAF
    
- previosly: PAF - multiple stages to refine heatmaps [6, 25]
    - they (here) exploit redundant info across frames
    - use previous pose heatmaps to predict keypoints & staf
    - lower computation costs, faster etc than others
- 
**attention**
- COCO [21] and MPII [3] datasets. PoseTrack dataset [17] - large scale corpus of video data with multiple people in the scenes
- use cases: self driving cars, augmented reality -> deploy on low-power embedded systems

**results**
64.6% mAP, 58.4% MOTA] on single scale at ∼30 FPS, and [71.5% mAP, 61.3% MOTA] on multiple scales at ∼7 FPS on the PoseTrack 2017 validation set using one GTX 1080 Ti

**topology**
- 21 body parts/ key points = sum of COCO (ears, nose, eyes) & MPII (head, neck)
- tafs & pafs = 21 -> 48 connections directly across keypoints similar to [8]

**training**
- pre-trained in image mode
    - strategies for unlabeled frames
    - heatmaps computed with l2 loss
    - 400k iterations
- next stage: training in video mode
    - augment: synthesize motion: scale, rotation, translation, skip frame (up to 3 frames)
    - logg weights of VGG
    
**inference & training**
- pose inference & tracking across frames given predicted heatmaps
- compute dot product between the mean vector of the sampled points & directional vector from first to second keypoin
- PAFs and TAFs sorted by their scores before inferring complete poses, associating them with unique ids (across frames)
- sorted list --> initialize new pose if both kp in PAF are unassigned
    --> add to existing pose if one of kp is assigned
    --> update score of PAF in pose if both assigned to same pose
    --> merge 2 poses if kp belong to different poses with opposing kp
    - assin id to each pose in current frame with most frequent id of kp from previous frame
    
**Experiments**
- filter sizes: replace 7x7 with 3 3x3, --> faster, less memory, better accuracy
- Video Mode/ Depth of 1st stage

**comparison**
FlowTrack - top-down --> slower than bottom-up STAF

**Inference & training**
- CNNs at every module + pass required data computed from previous frame


# markerless motion capture
- smartphones with depth cam:
    - https://developer.apple.com/documentation/avfoundation/cameras_and_media_capture/capturing_photos_with_depth
    - https://www.cashify.in/top-10-smartphones-with-a-dedicated-depth-sensor-camera-to-capture-perfect-bokeh-shots
- https://www.cs.ubc.ca/~shafaei/homepage/projects/papers/crv_16.pdf
    - Real-Time Human Motion Capture with Multiple Depth Cameras
- https://cinemontage.org/motion-capture-for-the-masses/
- https://graphics.tu-bs.de/upload/publications/multikinectsMocap.pdf Markerless Motion Capture using multiple Color-Depth Sensors

# Integrability and Chaos in Figure Skating
- physics, motion (2019)
https://arxiv.org/pdf/1809.11146.pdf

# 3D Sensing technology for real-time quantification of Athletes' Movements

# A Simple 3D Scanning System of Human Foot Using a Smartphone with a Depth Camera
    - https://www.researchgate.net/publication/329425920_A_Simple_3D_Scanning_System_of_Human_Foot_Using_a_Smartphone_with_a_Depth_Camera
    
# Template Deformation-Based 3-D Reconstruction of Full Human Body Scans From Low-Cost Depth Cameras


# Neuron: what is motion capture
- https://neuronmocap.com/content/mocap-101-what-motion-capture

https://www.xsens.com/inertial-sensors


# A data augmentation methodology for training machine/deep learning gait recognition algorithms
- create gait synthetic dataset for gait regognition systemes - characteristics in gait/ idetity of subjects
- multi camera Vicon motion capture system + 10 infrared cams, 38 reflective markers
- makehuman: attach avatar of human subject to a rig (skeleton, known as armature) = skinning procedure
    - each vertex of 3d mesh model is associated with bone in rig + weighed according physics engine - determines how body moves relative to skeleton
    - retargetting: connect motion capture data to the rig
- synthetic dataset allows arbitrary degrees of variation


# 3D Human Pose Estimation: A Review of the Literature and Analysis of Covariates
- (SynPose300)
- current datasets controlled environments, wear tight clothes 
    - Human3.6M: wear shorts and t-shirts
    - HumanEva one out of four subjects is female
- use blender & makehuman
    - export from blender in biovision hiearchy format
    - process in matlab
    - parse 3d coordinates for all joints


# HUMANEVA: Synchronized Video and Motion Capture Dataset and Baseline Algorithm for Evaluation of Articulated Human Motion
- optical trackers
- 6/2 subjects performing some actions
- trackers applied onto clothes

# High-Resolution Representations for Labeling Pixels and Regions
- maintain high-resolution representations
    - connect high-to-low resolution convolutions in parallel
    - repeatedly conduct fusions across parallell convolutions
- semantic segmentations: top results on Cityscapes, Pascal, Context, facial landmark detection on AFLW, COFW, 300W, WFLW
- apply to Faster R-CNN object detection framework 
    - superior results on COCO object detection
    
**general**
- low-resolution representations:
    - image classification
- high-resolution representations:
    -  semantic segmentation, object detection, human pose estimation
    - 2 main lines of computing:
        - recover high-resolution from low resolution ResNet, optional medium-res Hourglass, Segnet, DeconvNet, U-Net, encoder-decoder
        - maintain high-res + strengthen representations with parallel low-res 
    - dilated convolutions: replace strided convolutions & associate regular convs in classifications networks to compute medium-res
- HRNet:
    - connect high-to-low + multi-scale fusions -> strong + spatially precise
    - HRNetV2: eplore representations from all high-to-low resolution parallell convolutions (not only high-res as original HRNet)

- HRNetv1
    - maintain high-to-low res in parallell + repeated multi-scale fusions
    -only representations (feature maps) from high-res conv are outputtes --> only subset of output channels from high-res conv = exploited (other subsets from low-res are lost)
    
- HRNetv2
    -fully explore multi-res conv --> bileaniar upscaling + concat subset of representations
    - downsample: average pooling
    - 1st stage: 4 residual units, each formed by bottleneck with width 64, followed by one 3x3 conv  --> reducing the width of feature maps to C
    - 2nd, 3rd,4th stage 1,4,3 multi resolution blocks
    - each branch 4 residual unitis + 3x3 convs
    - pass to linear classifier/regressor with softmax/MSE loss predict maps/facial landmar heatmaps
    - reduce to 256 (high-res)
    - 1x1 conv before forming feature pyramid


# original HRNet: Deep High-Resolution Representation Learning for Human Pose Estimation
-resolution increase by bilinear (nearest neighbour) upsampling

# ON EMPIRICAL COMPARISONS OF OPTIMIZERS FOR DEEP LEARNING
- optimizer: 
- training speed, final predictive performance
    - today, no theory on choice --> empirical studies & benchmarking
- never underperform optimizer which is approximated (e.g. Adam should not underperform Momentum)
    - adabtives should not underperform SGD
    
-  ADAM (Kingma and Ba, 2015) & RMSPROP (Tieleman and Hinton, 2012) --> approximately simulate MOMENTUM (Polyak, 1964) 
    - if eta term in param updates is allowed to grow very large
    
- more general optimizers never underperform special cases --> ADAM other adabtive never underperform Momentum/ SGD
- Hyperparameter tuning is a crucial step
of the deep learning pipeline (Bergstra and Bengio, 2012; Snoek et al., 2012; Sutskever et al., 2013; Smith, 2018)

# test videos: 
https://www.youtube.com/watch?v=TlXCk1LDlC0
https://www.youtube.com/watch?v=8gP_qYEgaog
https://www.youtube.com/watch?v=hAJf7zc6S_8
https://www.youtube.com/watch?v=lNxG8XPP41M
- 


aims at
Ineed,
 Indeed, it is the de facto standard 
In fact
to date
Yet, 
Instead, 
Although
Accordingly,
popular first-order optimizers form a natural inclusion hierarchy
Despite conventional wisdom
carefully tuned, 
The remainder of this paper is structured as follows.1910.05446


https://ruder.io/optimizing-gradient-descent/index.html#nadam

gradient descent -> minimize objective function J(\theta) model's parameters \theata element \mathbb{R}^d
objective function \nabla_\theta J(\theta) -> update parameters in oppisite direction of gradient
follow direction of the slope of the surface downhill - reach valley

gradient descent variants: accuracy of parameter update vs time to update
1) batch gradient descent (vanilla) \theta = \theta - \eta \cdot \nabla_\theta J( \theta)
    - gradients of cost fct for entire training dataset w.r.t. parameters \theta
    - redundant computations -> recompute gradients for similar examples before each param update
2) stochastic gradient descent (SGD)
    - update for each training example x_i and label y_i \theta = \theta - \eta \cdot \nabla_\theta J( \theta; x_i}; y_i)
    - much faster, frequent updates with high variance
    - jump to another minimum -> complicate convergence to exact minimum (overshooting problem)
    - slowly decrease lr -> convergence behavior == BGD
3) Mini-batch gradient descent
    - update for every mini-batch of n training examples \theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})
    - reduce variance of parameter updates --> more stabel convergence
    - highly optimized matrix optimizations integrated into common libraries (50-256)
    - challenges: 
        - choose right lr: flunctuate around minimum/ land on plateau
            - lr scheduels - define in advance, do not adapt to ds charachteristics 
            - same lr to all parameter updates -- sparse data, differenct feature frequences: larger updates for rarely occuring features
        - trapped in local minima, or saddle point (one dimension slopes up/ one down)