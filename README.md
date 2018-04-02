## CSIT-5800 Mini Project Proposal Report

**Group 3: GUO,Yuchen, LI, Can, LIN, Jiajun, LUO, Xiaoyuan**



**Background**

After centuries of intense whaling, recovering whalepopulations still have a hard time adapting to warming oceans and struggle tocompete every day with the industrial fishing industry for food. To aid whaleconservation efforts, scientists use photo surveillance systems to monitorocean activity. They use the shape of whales’ tails and unique markings foundin footage to identify what species of whale they’re analyzing and meticulouslylog whale pod dynamics and movements. For the past 40 years, most of this workhas been done manually by individual scientists, leaving a huge trove of datauntapped and underutilized.

 

### Task - Whale Identification

The task is to build an algorithm toidentify whale species in images. There are over 25,000 images, gathered fromresearch institutions and public contributors. The picture only contains thefluke of a whale.

 

### Data Description

This training datacontains thousands of images of humpback whale flukes. Individual whales havebeen identified by researchers and given an Id. The challenge is topredict the whale Id of images in the test set. train.zip - afolder containing the training images.

l  train.csv -maps the training Image to theappropriate whale Id. Whales thatare not predicted to have a label identified in the training data should belabeled as new_whale.

l  test.zip -a folder containing the test images to predict the whale Id.

l  sample_submission.csv -a sample submission file in the correct format.

 

### Challenge and Output

What makes this such achallenge is that there are only a few examples for each of 3,000+ whale Ids.

For each Image in the testset, we may predict up to 5 labels for the whale Id. Whales that are notpredicted to be one of the labels in the training data would be labeledas new_whale.