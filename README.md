# Motion Prediction on Waymo Open Motion Dataset

Course - CSC490H5F: Machine Learning for Vision

## Team

- Andrew Xing
- Dan Nguyen
- Prem Patel
- Arjun Jayakrishna

## Project Overview

This project focuses on motion prediction for autonomous driving. Given the recent trajectories of the agents around a self-driving vehicle, along with a map of the surrounding roads, the goal is to predict multiple possible future trajectories for each agent over the next eight seconds, each with a confidence score.

## Dataset

Waymo Open Motion Dataset (WOMD): https://waymo.com/open/

Note: the dataset is not included in this repository and must be downloaded separately as per Waymo's terms and licensing.

## Baseline

This project will build on DONUT-NLL (Knoche et al., ECCV 2026): https://github.com/MKnoche/DONUT-NLL

## Evaluation

In order to evaluate the model, official WOMD motion prediction metrics will be computed on the validation set. Among these metrics, we will put emphasis on minimum Final Displacement Error (minFDE) and soft mean Average Precision (soft mAP)(Soft mAP, mAP, miss rate, minFDE).
