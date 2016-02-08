# Welcome to COMPS!!

This is gonna rock.

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

## Running feature_detect
In Python:
```bash
cd feature_detect
python detect_basic.py
```

In C++:
```bash
cd feature_detect
cmake .
make
./FDTest test.jpg
./FMTest test1.jpg test2.jpg
```

## TODO:
- Update this todo with proper build instructions
- Save intermediate results (images with features) in separate directory
- Demo that generates points from two images
- Generate stereolithography (or equivalent) file from point outputs that we can open with meshlab
- Write end-to-end script that runs our program, saves the output, and opens it in meshlab

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



    
    
## Pythonpath stuff (in fish)
    set -x PYTHONPATH /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/


## Triangulate points test
    
    >>> for i in ['point1']:
    ...     for j in ['view1','view2']:
    ...             for k in ['kindx0','kindx1','kindx2','kindx3']:
    ...                     print "matrA({}*2+0, {} ".format(j,k) + " = projPoints[{}][0][{}] * ".format(j,i) + " projMatrs[{}][2][{}] - projMatrs[{}][0][{}]".format(j,k,j,k)
    ...                     print "matrA({}*2+1, {} ".format(j,k) + " = projPoints[{}][0][{}] * ".format(j,i) + " projMatrs[{}][2][{}] - projMatrs[{}][1][{}]".format(j,k,j,k)
    ...
    matrA(view1*2+0, kindx0  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx0] - projMatrs[view1][0][kindx0]
    matrA(view1*2+1, kindx0  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx0] - projMatrs[view1][1][kindx0]
    matrA(view1*2+0, kindx1  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx1] - projMatrs[view1][0][kindx1]
    matrA(view1*2+1, kindx1  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx1] - projMatrs[view1][1][kindx1]
    matrA(view1*2+0, kindx2  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx2] - projMatrs[view1][0][kindx2]
    matrA(view1*2+1, kindx2  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx2] - projMatrs[view1][1][kindx2]
    matrA(view1*2+0, kindx3  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx3] - projMatrs[view1][0][kindx3]
    matrA(view1*2+1, kindx3  = projPoints[view1][0][point1] *  projMatrs[view1][2][kindx3] - projMatrs[view1][1][kindx3]
    matrA(view2*2+0, kindx0  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx0] - projMatrs[view2][0][kindx0]
    matrA(view2*2+1, kindx0  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx0] - projMatrs[view2][1][kindx0]
    matrA(view2*2+0, kindx1  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx1] - projMatrs[view2][0][kindx1]
    matrA(view2*2+1, kindx1  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx1] - projMatrs[view2][1][kindx1]
    matrA(view2*2+0, kindx2  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx2] - projMatrs[view2][0][kindx2]
    matrA(view2*2+1, kindx2  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx2] - projMatrs[view2][1][kindx2]
    matrA(view2*2+0, kindx3  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx3] - projMatrs[view2][0][kindx3]
    matrA(view2*2+1, kindx3  = projPoints[view2][0][point1] *  projMatrs[view2][2][kindx3] - projMatrs[view2][1][kindx3]
