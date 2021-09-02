# affine_model 
This project implements and tests affine jump-diffusion stochastic volatility models, and calibrate the model to 50 EFT options in Chinese market.

The folder includes the following directories and files:

Data folder : includes 50ETF options contract specifications, historical option price and underlying price.

Utils folder: includes date time utility functions, black-shcoles analytical solutions, and system of ODEs used in solving affine process

Output folder: includes all testing results described in the paper

The remaining scripts define affine model class and tests
