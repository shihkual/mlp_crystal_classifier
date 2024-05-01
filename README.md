## mlp_crystal_classifier
## Source code of the paper "Classification of complex local environments in systems of particle shapes through shape symmetry-encoded data augmentation."

Note: In the 2023 AIChE annual meeting, we present the same work with the old title "Order parameter-free classification of complex local environment of shape particles." Info: https://aiche.confex.com/aiche/2023/meetingapp.cgi/Paper/668719

Here, we demonstrate the source code for training the Multilayer Perceptron local geometric classifier for shape particle systems written in Python. We take the first testing case of the hard cubes system in the paper to demonstrate our works.


# Prerequisites

The source code requires the following Python packages:

* [Numpy](https://github.com/numpy/numpy)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [rowan](https://github.com/glotzerlab/rowan)
* [coxeter](https://github.com/glotzerlab/coxeter)
* [freud](https://github.com/glotzerlab/freud)
* [gsd](https://github.com/glotzerlab/gsd)
* [Pytorch](https://github.com/pytorch/pytorch)
* [torchvision](https://github.com/pytorch/vision)

Conda users can install these from [conda-forge](https://conda-forge.org/)


# Usage
The trajectory data and scripts were generated for our work "Classification of complex local environments in systems of particle shapes through shape-symmetry encoded data augmentation" (admits peer review process). The trajectory data and scripts are organized as follows: 1) The `utils` directory stores Python scripts in order to perform data preparation, model training, and model testing tasks. It contains three different scripts to separate functions aiming for different usage. 2) The `data` folder stores training and testing trajectory data in the **GSD** file format of the seven systems of particle shapes. 3) The `cube` folder stores the Jupyter notebooks for performing data preparation, model training, and model testing tasks. 

We explain the three folders in detail:

1. The `utils` folder:

   - `data_prep.py`:

      Store functions for performing data preparation by constructing local geometric fingerprints of particles of the selected snapshots in a given trajectory.
   
      - `prepare_data`: Prepare data without symmetry augmentation.
      
      - `prepare_data_augmented`: Prepare data with symmetry augmentation.
   
      - `prepare_triangles_data`: Prepare data without symmetry augmentation specifically for the patchy triangles system, which requires labeling particles based on particle types.
   
      - `prepare_triangles_data_augmented`: Prepare data with symmetry augmentation specifically for the patchy triangles system, which requires labeling particles based on particle types.
   
      - `normalize_angleij`: Normalize the angular part of the relative position in spherical coordinates based on the input reference particle orientation and shape symmetry factors.
   
      - `normalize_qij`: Normalize the relative quaternion based on the input reference particle orientation and shape symmetry factors.

      - `‎appendSpherical_np`: Transform the positions from cartesian coordinates to spherical coordinates.


   - `evaluation.py`:
   
      Store functions for testing.

      - `mlp_eval`: Evaluate the classification of the MLP classifier trained on non-augmented data.
      
      - `mlp_eval_augmented`: Evaluate the classification of the MLP classifier trained on augmented data.
      
      - `calc_rdf`: Calculate radial distribution function given trajectory file and frames using freud's functionality.

   - `mlp.py`:

     For creating MLP classifier and data sets loading. Besides, store the functions for training and testing.



2. The `data` folder
  
   Stores trajectory files in **GSD** file format of the hard cubes system.



3. The `cube` folder

   Stores Jupyter notebooks of `cookdata.ipynb`, `training.ipynb`, and `testing.ipynb` to demonstrate data preparation, training and testing of the MLP classifier.

   
# Cite this work

Lee, S. K. A., Tsai, S. T., & Glotzer, S. C. (2024). Classification of complex local environments in systems of particle shapes through shape symmetry-encoded data augmentation. The Journal of Chemical Physics, 160(15).
