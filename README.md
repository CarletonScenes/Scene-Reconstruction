# Welcome to COMPS!!

This is gonna rock.

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