# Learn To Paint 

Code for automated painting.

This a refactor of [ctmakro repository](https://github.com/ctmakro/opencv_playground).

# Run

Install requirements 

    pip install numpy opencv-python
    
Run main function via

    painter_fun.py  -i input_dir -o output_dir [--salience_path salience_path] -e nb_epochs
    
It will target all images present in the given input directory, and expect equally named grayscale salience images in the  given salience directory (this parameter is optional).

For a more details list of parameters, see

    painter_fun.py  --help
    
For salience detection I suggest [3D Photography using Context-aware Layered Depth Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) or [my code for face extraction](https://github.com/5agado/data-science-learning/tree/master/face_utils). 

# TODO
- explore other image distance functions
- edge detection and cut brush
- k-means to select dominant colors from the image