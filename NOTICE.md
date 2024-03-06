                    FEDPYLOT

Copyright (C) 2023  Cyprien Quéméneur

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For questions or inquiries about this program, 
please contact [cyprien.quemeneur@polymtl.ca](mailto:cyprien.quemeneur@polymtl.ca).

                    THIRD-PARTY COMPONENTS

This program includes third-party code, redistributed under the GNU General Public License 3.0 (GPL-3.0).

Third-Party Component: Official YOLOv7
  * License: GNU General Public License 3.0
  * Source: https://github.com/WongKinYiu/yolov7
  * Latest commit: a207844 on 2023-11-03 (main branch)

- Modifications to the original repository
  - Modifications to yolov7/train.py and yolov7/train_aux.py
    - Add sys and pandas imports
    - Modify the training and validation paths collected from the dataset yaml file at runtime based on the client rank
    - Fixed an issue where "nesterov" was set to True in SGD even when the momentum was set to 0
    - The "notest" parameter now deactivates all testing-related code, even for the final epoch
    - The number of warm-up iterations is no longer set to a minimum of 1000 to allow training with FedAvg
    - Record the learning rates, moment, losses, and results in csv files
    - Interrupt the training subprocess at the end of each communication round in the federated experiments
    - Add "client-rank" and "round-length" parameters to the argparser
  - Modifications to yolov7/test.py
    - Add pandas and utils.loss.ComputeLoss imports
    - Modify the validation path collected from the data file at runtime
    - Record the validation loss when the model is not traced
    - Only retain the subclass name to improve readability with nuImages in the image plots and the confusion matrix
    - Modify the spacing when printing the results table to improve readability with nuImages
    - Add "saving-path" and "kround" parameters to the argparser
    - Save testing results in a csv file
  - Other modifications
    - The .gitignore file was moved to the program's root
    - Fixed a SyntaxError in YOLOv7-E6E re-parameterization in yolov7/tools/reparameterization.ipynb
    - Fixed a device issue in ComputeLossAuxOTA in yolov7/utils/loss.py which prevented training of P6 models
    - Deactivated TensorBoard recording in yolov7/train.py and yolov7/train_aux.py
   
- Additionally, the module node.py contains modified code taken from YOLOv7
  - The function reparameterize is based on yolov7/tools/reparameterization.ipynb
  - The function post_init_update is based on the "Model" and "Model parameters" sections of YOLOv7 training scripts
