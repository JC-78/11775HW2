<<<<<<< HEAD
To genereate sift.csv,
follow TA's github instructions and extract SIFT. Then, 
python code/run_bow.py data/labels/train_val.csv sift_15 data/sift
python code/run_bow.py data/labels/test_for_students.csv sift_15 data/sift
python code/run_mlp.py sift --feature_dir data/bow_sift_15 --num_features 15

To generate cnn.csv,
python code/run_cnn.py data/labels/train_val.csv
python code/run_cnn.py data/labels/test_for_students.csv
python train.py

To generate video.csv,
python code/run_cnn3d.py data/labels/train_val.csv
python code/run_cnn3d.py data/labels/test_for_students.csv
python train3d.py



=======
>>>>>>> master
