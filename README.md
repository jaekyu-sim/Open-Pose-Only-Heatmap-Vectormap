# Open-Pose-Only-Heatmap-Vectormap
### Open Pose(Detecting Human Body Skeleton) network(Only heatmap and vectormap)

### This project only could make vectormap and heatmap. So, I'll upload the project that could express the pose data applied the non-maxima suppression.

#### The Loss of this project is expressed here.

![Heatmap_output1](/github_image/heat_loss.png)
![Heatmap_output1](/github_image/vector_loss.png)

#### the left is heatmap loss and right is vectormap loss.

#### In my project, there are many python files.

#### 1. For_parsing_mat_to_json.ipynb
#### this ipynb file makes .mat file to json file. When you download 'mpii_human_pose_dataset', there are annotation data and its format is .mat. So you convert this file format to json.

#### 2. Image_resize.ipynb
#### this ipynb file makes image files to regular size(regular width x regular height). The reason why I make image files size regulary is for saving time. By making image file size regular, training time could be reduced. Because there are no time to resize image during training.

#### 3. network.py and network.ipynb
#### this files are containing VGG-19 network.

#### 4. Pose.ipynb
#### this ipynb file is for training. It contains class and util function to train.


##### Here is a output of this project. There are only vectormap and heatmap images.

##### input images
![input_image](/github_image/test_image.png)

##### 1. Heatmap images


![Heatmap_output1](/github_image/test_heatmap1.png)
![Heatmap_output2](/github_image/test_heatmap2.png)
![Heatmap_output3](/github_image/test_heatmap3.png)
![Heatmap_output4](/github_image/test_heatmap4.png)
![Heatmap_output5](/github_image/test_heatmap5.png)
![Heatmap_output6](/github_image/test_heatmap6.png)
![Heatmap_output7](/github_image/test_heatmap7.png)
![Heatmap_output8](/github_image/test_heatmap8.png)
![Heatmap_output9](/github_image/test_heatmap9.png)
![Heatmap_output10](/github_image/test_heatmap10.png)
![Heatmap_output11](/github_image/test_heatmap.png)


##### 2. Vectormap images
![Vectormap_output1](/github_image/test_vectormap1.png)
![Vectormap_output2](/github_image/test_vectormap2.png)
![Vectormap_output3](/github_image/test_vectormap3.png)
![Vectormap_output4](/github_image/test_vectormap4.png)
![Vectormap_output5](/github_image/test_vectormap5.png)
![Vectormap_output6](/github_image/test_vectormap6.png)
![Vectormap_output7](/github_image/test_vectormap7.png)
![Vectormap_output8](/github_image/test_vectormap8.png)
![Vectormap_output9](/github_image/test_vectormap9.png)
![Vectormap_output10](/github_image/test_vectormap10.PNG)
![Vectormap_output11](/github_image/test_vectormap.PNG)
