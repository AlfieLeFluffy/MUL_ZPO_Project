# ZPO/MUL Project


**Author**: Bc. Tomáš Vlach

## Overview

This project implements convolution through the spectral domain transformation and blind/non-blind deconvolution for image restoration.

## Project Structure

```
MUL_ZPO_Project
├── doc/
│   └── Project papers        
├── img/
│   └── Example images 
├── src/
│   ├── conv/
│   │   └── Convolution Implementation
│   ├── deconv/
│   │   └── Deconvolution Implementaion
│   ├── gui/
│   │   └── GUI Tools
│   ├── util/
│   │   └── Misc Tools
│   ├── app.py
│   ├── convolution.py
│   ├── deconvolution.py
│   ├── image.py
│   └── processing.py
├── Dockerfile
├── README.md
├── main.py
├── run_docker.sh
└── run_program.sh
```


## Setup

The project is implement for **Python 3.12.2** and all required libraries are found within the `requirements.txt` file and can be installed through `pip install -r requirements.txt`. 

Another way to use the project is through Docker and that is done by using the include `Dockerfile` and either using the also included `run_docker.sh` bash script or running the Docker commands yourself.

## Use

The application uses a GUI interface created through `customtkinter` and can be run through the command `python ./main.py` or the atteched bash script `run_program.sh`.

If you are using the attached `Dockerfile` then the program will run automatically up upon starting up the container.