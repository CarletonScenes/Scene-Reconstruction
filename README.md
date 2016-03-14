# Carleton 3D Scene Reconstruction Comps

## Pulling the repo

We use git-lfs to track large files in the repo. Install that on OSX like this:

    brew install git-lfs
    
or take a look at [their docs](https://git-lfs.github.com/).

Then run the ol' clone command:

    git clone git@github.com:CarletonScenes/Scene-Reconstruction.git

## Installing OpenCV for feature detection

On Mac OSX:
```bash
brew tap homebrew/science
brew install opencv3 --c++11 --with-cuda --with-contrib
brew ln opencv3 --overwrite --force
```

## To get the environment set up

    cd python
    sudo easy_install pip
    sudo pip install virtualenv matplotlib
    virtualenv env
    source ./env/bin/activate
    pip install -r requirements.txt

    brew install opencv3 --c++11 --with-cuda --with-contrib
    brew ln opencv3 --overwrite --force
    cp /usr/local/lib/python2.7/site-packages/cv* env/lib/python2.7/site-packages

At this point, you have opencv installed on your computer, copied the shared object files from opencv to the virtualenv python sources folder, and installed matplotlib to your regular python installation.

To activate the environment, run (in bash):
    . ./env/bin/activate
    export PYTHONPATH=$PYTHONPATH:/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

Or in fish:
    source ./venv_activate.fish # This will both start the virtual env and set the PYTHONPATH variable.

# Moving forward

Once you have the virtualenv installed, and activated (activation is custom, check out the commands above, they set the $PYTHONPATH env variable for you), you can run the program from the project root:


    ✗ python python/do_comps.py
    Welcome to do_comps.py!
            To run this program, you'll need to select one of the
            modes below and perhaps provide more input.

            modes:
                python do_comps.py detect -i img1.jpg [-i img2.jpg ...] [-f input_folder] [-o output.jpg]
                python do_comps.py match -i img1.jpg -i img2.jpg [-i img3.jpg ...] [-f input_folder] [-o output.jpg]
                python do_comps.py triangulate -i img1.jpg -i img2.jpg [-i img3.jpg ...] [-f input_folder]
                                    --scene_output scene.ply [--projection_output projection.ply]

    python python/do_comps.py triangulate -i python/photos/c1.jpg -i python/photos/c2.jpg --scene_output tri_out.ply
    python python/do_comps.py triangulate -i python/photos/c1.jpg -i python/photos/c2.jpg --scene_output tri_out.ply --projection_output proj_out.ply

    python python/do_comps.py manual_pts python/points/pdppoints.txt -i python/photos/pdp1.jpeg -i python/photos/pdp2.jpeg --scene_output pdp_out.ply --projection_output pdp_proj.ply

    
## Pythonpath stuff (in fish)
    set -x PYTHONPATH /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/
