# Comparison-of-Target-Datasets-for-U-Net-Precipitation-Estimation
In this Github we are providing the code used in the paper: A Comparative Evaluation of Target Datasets for U-Net-Based Precipitation Estimation by Jimenez Arellano et al. (2025), currently in prep.

In the paper, we trained the same U-Net model for precipitation estimation three separate times using different target datasets. The target datasets were the National Severe Storms Laboratory (NSSL) Multi-Radar Multi-Sensor (MRMS), NASA’s satellite precipitation product, the Integrated Multi-satellitE Retrievals for Global Precipitation Measurement Final Run (IMERG Final), and Climate Prediction Center (CPC) Combined Passive Microwave Precipitation (MWCOMB). The only input to the model was infrared data, specifically the CPC 4km IR data. 

In this repository you can find the code for the model trained. The code was the same for all three target datasets, which explains why there is only one code. 
