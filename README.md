<p align="center">
    <a href="https://ibb.co/64Jj23T"><img src="https://i.ibb.co/kh6NpRw/SkyStock.png" alt="SkyStock" border="0"></a>
</p>

## Introduction
SkyStock is a Pyside6 application for extracting stock piles from point clouds. 

**The project is still in pre-release, so do not hesitate to send your recommendations or the bugs you encountered!**

<p align="center">
    <a href="https://ibb.co/WggSf4P"><img src="https://i.ibb.co/nzzhjNn/skystock-gui.jpg" alt="skystock-gui" border="0"></a>
    
    GUI for stock piles segmentation
</p>


## Principle
The application is based on "Segment Anything" image segmentation, and then projects the 2D segmented area onto the 3D data.
We decided to add a simple graphical user interface to facilitate the data extraction process!


### Step 1: Importing an image
Simply choose a point cloud from your HDD. The process will automatically create a top view of the interest zone.

### Step 2: Click on a stock pile
Click on a pixel belonging to a stock pile to start the 2D segmentation. When the computation is finished, the app will propose three segmentation outputs.

### Step 3: Choose best 2D segmentation output

### Step 4: Give a name to the stock pile

### Step 5: Wait for 3D segmentation result!

<p align="center">
    <a href="https://ibb.co/7y0p3Qb"><img src="https://i.ibb.co/ryXpB3v/skystock-output.jpg" alt="skystock-output" border="0"></a>
    
    Result of the segmentation process
</p>

## Upcoming key features:

- **2D SAM Segmentation**:
    - Adding positive and negative prompts
    - Adding box selection
- **General stock pile detection using object detection**
- **CloudCompy support for faster process**
- **Integrated WebODM support to start from drone images**

## Installation instructions

1. Clone the repository:
```
git clone https://github.com/s-du/SkyStock
```

2. Navigate to the app directory:
```
cd SkyStock
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
python main.py
```

## User manual
(coming soon)

## Contributing

Contributions to the IRMapper App are welcome! If you find any bugs, have suggestions for new features, or would like to contribute enhancements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request describing your changes.

## Acknowledgements
This project was made possible thanks to subsidies from the Brussels Capital Region, via Innoviris.
Feel free to use or modify the code, in which case you can cite Buildwise and the Pointify project!


