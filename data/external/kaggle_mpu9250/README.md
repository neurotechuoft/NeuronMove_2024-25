# Kaggle hand tremor (MPU9250)

Grabbed from here: https://www.kaggle.com/datasets/aaryapandya/hand-tremor-dataset-collected-using-mpu9250-sensor  

`Dataset.csv` is the only file — ~1.2 MB, ~28k rows. Git won’t track it because of `*.csv` in gitignore, so if you clone fresh you’ll need to download again and drop it in this folder.

**What’s in the columns:** accel (aX/aY/aZ), gyro (gX/gY/gZ), mag (mX/mY/mZ), then `Result`. Per the Kaggle blurb: **1 = shaking hand, 0 = stable hand** — so it’s not really “Parkinson’s vs healthy,” it’s shake vs not for that row.

**Heads up:** the magnetometer values are basically all `-1`, so we’re treating them as useless and only using accel + gyro unless someone figures out otherwise.

License on Kaggle says “Unknown” — worth double-checking before publishing anything with this data.
