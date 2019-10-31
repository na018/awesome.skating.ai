# Action Recognition in Figure Ice Skating

> Next level figure skating ai analysis.

<p style="text-align:center; background-color: gray;"><img src="axel_paf.gif" width="200px"></p>

[Action Recognition in Figure Ice Skating](#action-recognition-in-figure-ice-skating)

# TOC

- [Action Recognition in Figure Ice Skating](#action-recognition-in-figure-ice-skating)
- [TOC](#toc)
  - [Summary 🗒](#summary-%f0%9f%97%92)
  - [Goals ⛳️](#goals-%e2%9b%b3%ef%b8%8f)
    - [Minor](#minor)
    - [Major](#major)
  - [Team](#team)

---

## Summary 🗒

This investigative research project tries to build upon the very well created Part Affinity Field Python library from [OpenPose](https://arxiv.org/pdf/1812.08008.pdf) and map the resulting vectors to specific actions in figure ice skating.

Here we will follow a bottom-up approach as suggested by the OpenPose [Paper](https://arxiv.org/pdf/1812.08008.pdf).

> The research work of [PAF](https://github.com/CMU-Perceptual-Computing-Lab/openpose) was first published in 2017. Further development processes with additional facial and foot tracking were reported publicly in 2019. This project allows to track body parts in realtime thus will serve as the foundation for this project.  
> A great benefit there is the publicly available sourcecode on [Github](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and the Python library.
> In this project, we will first concentrate on a single jump `the Axel` and later in our "Major Goals" try to extend the found knowledge.

## Goals ⛳️

### Minor

- map vector movements of body parts to specific curves
- map combination of curves to specific actions _(first we will concentrate on the `Axel jump`)_

### Major

- App to support coaches
- Service to support competition's _(fairness, substitutive jury)_

## Team

- Nadin-Katrin Apel
- Johannes Horn
- Nandor Babina
- Jonas
