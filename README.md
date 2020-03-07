# CS499_Deep_Learning_Project-3

## Project Description
In this project our goal will be implementing a stochastic gradient descent algorithm for a neural network with one hidden layer.

#### NNOneSplit function
See our implementation of NNetOneSplit function using R language [here](NNetOneSplit.R).

#### Gradient Descent Algorithm
You can also see our implementation of Gradient Descent Algorithm using R language [here](GradientDescent.R).

## How to run it
### 1 Environment Configuration
#### 1.1 Install R
Because the project is written in R language, so it is necessary to have R on your machine.

You can download R [here](https://www.r-project.org/) and install it.

#### 1.2 Install RStudio
This is the easiest way for everyone can run the project.

You can download RStudio [here](https://rstudio.com/products/rstudio/download/#download) and install it.

### 2 Having the project
#### 2.1 Download the project
You can use ```git clone``` to clone the project or just click the green button to download a ZIP file.

#### 2.2 Unzip the project
Use any tools you like to unzip the project into the folder you want.

### 3 Run the project
#### 3.1 Set the path
***This is an important step, ignoring this may cause some problems with reading data.***
Open RStudio, in the 'console' command line at the bottom left corner, type
```
setwd('PATH')
```
where PATH is where you unzip all the R files and data on your machine.

WARNING: In your path, use '/' instead of '\'.

#### 3.2 Open R files
Click 'File'->'Open File' at the top left corner, the choose
```
GradientDescent.R
NNetOneSplit.R
Experiment.R
Extra_Credit-2.R
Extra_Credit-3.R
Extra_Credit-5.R
```
to open it.

#### 3.3 Run R files
First go to GradientDescent.R, then click the 'run' button in the file section (not the whole window) once, this will run the current line, until there's nothing running in the console section.

Then go to NNetOneSplit.R, then click 'run' again until there's nothing running in the console section.

Then go to Experiment.R, then click 'run' until we see the package 'ggplot' is called, a graph will show in the bottom right section, showing the relationship between subtrain/validation loss and # of epochs with a point to emphasize the minimum of the validation loss curve. Click more, a percentage table will show. Maybe you need to wait for a while to train and load. When the console in the bottom left corner is running, please do not load new commands.

For extra credits, just go to Extra_Credit-2.R, Extra_Credit-3.R, Extra_Credit-5.R, and run from the first line again to the end. Maybe you need to wait for a while to train and load. When the console in the bottom left corner is running, please do not load new commands.

Maybe you need to wait for a while to train and load. When the console in the bottom left corner is running, please do not load new commands.

## About
This is our third group project of CS499 Deep Learning course in Spring 2020 at [NAU](https://nau.edu/)

### Project Requirements
You can find the requirements for this project [here](https://github.com/tdhock/cs499-spring2020/blob/master/projects/3.org)

### Instructor
Dr. T.D.Hocking - [tdhock](https://github.com/tdhock) at [SICCS](https://nau.edu/school-of-informatics-computing-and-cyber-systems/)

### Authors
* Zhenyu Lei - [lei37927](https://github.com/lei37927)
* Jianxuan Yao - [JianxuanA](https://github.com/JianxuanA)
* Shuyue Qiao - [SHUYUEQIAO](https://github.com/SHUYUEQIAO)

### Copyright ©
Any cloning or downloading before the project due date constitutes an infringement of our intellectual property rights, and after that it goes to open source. For any of the aforementioned infringements, Zhenyu Lei, Jianxuan Yao and Shuyue Qiao will report this to the NAU [Academic Integrity Hearing Board](https://in.nau.edu/academic-affairs/academic-integrity/).
