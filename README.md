Lib folder contains all libraries used in each classifier
datasets folder contains json files we want to test

Dependencies:
Install visual studio (2019 recomended).
Install CMake from this link https://github.com/Kitware/CMake/releases/download/v3.19.0-rc2/cmake-3.19.0-rc2-win64-x64.msi

Create a new environment called attention-analysis from anaconda prompt
	conda create --name attention-analysis python=3.7
	source activate attention-analysis
	
Install opencv, tensorflow, tqdm packages to the environment we just created (face-detection) from anaconda navigator	

Install some packages from anaconda prompt
	1- conda install scikit-learn
	2- conda install -c conda-forge scikit-image
	3- conda install -c menpo opencv3
	4- pip install cvlib  
	(#reference https://github.com/arunponnusamy/cvlib)
	5- conda install cmake
Either use step 6 or step 7, 
	6- conda install -c conda-forge dlib
	7- pip install dlib
After successfully installing dlib, install face recognition using the following command.
	8- pip install face_recognition 
	(#reference https://pypi.org/project/face-recognition/)
	9- pip install keras
	10-pip install pickle
  11-pip install fer
  12-pip install pyQt5

If you faced any issues while installing dlib or face_recognition please follow this link https://github.com/ageitgey/face_recognition/issues/175

download the dataset used in training attention model from https://drive.google.com/drive/folders/1lxwO-A-VBZCVuKKP1FgE_h0ZgfxUvlR4 
