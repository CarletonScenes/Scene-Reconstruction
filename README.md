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
