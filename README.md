# 2-D Flow modeling in the Allan Hills, Antarctica using Firedrake

This repo contains the work to model flow along a variety of flowline transects in the Allan Hills, Antarctica using finite elements through Firedrake.

## Contents

- Borehole_Locations  
  
    Contains a shapefile that are the locations of ice cores drilled in the Allan Hills Blue Ice Area (BIA) including those in which temperature measurements have been taken.

- Borehole_Temps  
  
    Contains pickle files of the processed distributed temperature sensing (DTS) measurements from the Allan Hills boreholes.  

- Figures  
  
    Contains png files of figures generated from the figure creating notebooks e.g. fig1_map.ipynb etc. This is to be updated regularly, hopefully.

- GPS_velocities  
  
    Processed GPS surface velocity data collected from the ablation stake deployment. Courtesy Margot Shaya. Not to be used or published without approval from Margot Shaya and John-Morgan Manos at the University of Washington.

- Meshes  
  
    2-D meshes produced from flowlines that are in the flowlines folder.

- Figures  
  
    Contains png files of figures generated from the figure creating notebooks e.g. fig1_map.ipynb etc. This is to be updated regularly, hopefully.

- Processing_noterbooks  
  
    Contains the notebooks that are to do all the analysis up until this point. Once we are satisfied with a figure or analyis, it will be moved to a new notebook in the main folder for more streamlined plotting. Also in this folder is a requirements text file with the package requirements for the container.

- flowlines  
  
    Shapefiles with the flowline transects used to extract the surface and bed elevations used for the meshing.


- scripts  
  
    Contains a py file that has uses functions to do some of the processing for firedrake modeling that the processing notebooks use.
