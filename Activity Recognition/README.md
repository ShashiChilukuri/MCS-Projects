This folder contains following files realted to Assignment1 - Activity Recognition Project
- Project report: ASU CSE 572 Data Mining - Porfolio Report.pdf
- Code: Project1.py, project1.ipynb


For code in Project.py
- Need following libraries - os, glob, numpy, pandas, matplotlib, sklearn
- All three section are well commented with relevant details


In this project, we have develop a computing system that can understand human activities. I was provided with data (real world wristband data) for a given activity, specifically eating action mixed with other unknown activities. Aim of this project is to identify the eating activities amongst the noise.

In the first phase we will do data cleaning, noise reduction, data organization and feature extraction.
- Phase 1: Data Cleaning and Organization
The data provided to you is collected using two sources:
A) Wristband data: Where the subject is wearing the wristband and performing eating actions periodically interspersed with non-eating unknown actions. The wristband provides you with i) accelerometer, ii) gyroscope, iii) orientation, and iv) EMG sensors. The sampling rate is 50 Hz for all the sensors.
B) Video recording of the person performing eating actions only used for establishing ground truth. In the data, you will be provided with the ground truth in the form of frame numbers where an eating action starts and ends. The actual video data will not be provided for privacy issues. The video data is taken at 30 frames per second. Hence you have to convert the frame numbers into sample numbers for the wristband sensors through some logic that you have to develop.The assumption that you can take is that the start frame and sample #1 of the wristband are synchronized. The output of this step will be a set of data snippetsthat are labelled eating actions and a set of data snippets that are non-eating.The way to convert the frame numbers into sample numbers is as follows:Consider the ground truth file, where there are three columns. Ignore the last column. The first column is the start frame of an eating action, the second column is the end frame. Each row is an eating action. The first frame number can be
multiplied by 50 and divided by 30. This gives you the corresponding sample number that indicates the start of an eating action. The second frame number can also be multiplied by 50 divided by 30 to get the end sample of an eating action. Do this for every row to get the sample numbers for eating actions for a person.
