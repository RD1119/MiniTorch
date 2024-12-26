# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

<<<<<<< HEAD
This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running
=======
# Task 5: Training
>>>>>>> 2b323a7dd4c16a8ab008d1abbd8073a8436e0b36

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

<<<<<<< HEAD
# Task 5
## Log for sentiment.txt: [sentiment.txt](sentiment.txt)
## Log for mnist.txt: [mnist.txt](mnist.txt)
=======
## XOR
Time per epoch: 0.109s.
### Parameters
    PTS = 50
    HIDDEN = 6
    LEARNING RATE = 0.5
    EPOCHS = 800
### Visualization
![task5-result](./Visualizations/XOR.png)
### Records
        Epoch: 10/800, loss: 32.29949422517297, correct: 29
        Epoch: 20/800, loss: 31.215947918101516, correct: 29
        Epoch: 30/800, loss: 29.835743936276437, correct: 29
        Epoch: 40/800, loss: 28.02155207840872, correct: 40
        Epoch: 50/800, loss: 25.701040961698414, correct: 44
        Epoch: 60/800, loss: 22.99711278171713, correct: 43
        Epoch: 70/800, loss: 20.227976273552528, correct: 43
        Epoch: 80/800, loss: 22.71861324590036, correct: 35
        Epoch: 90/800, loss: 17.39094840356826, correct: 45
        Epoch: 100/800, loss: 14.906676295635279, correct: 46
        Epoch: 110/800, loss: 22.02889162301, correct: 36
        Epoch: 120/800, loss: 19.38225706187765, correct: 41
        Epoch: 130/800, loss: 16.539010512394675, correct: 43
        Epoch: 140/800, loss: 16.17661595320811, correct: 44
        Epoch: 150/800, loss: 14.519292754457325, correct: 44
        Epoch: 160/800, loss: 14.139724103132439, correct: 45
        Epoch: 170/800, loss: 13.306844016797037, correct: 45
        Epoch: 180/800, loss: 12.726757428453636, correct: 45
        Epoch: 190/800, loss: 12.201448435690464, correct: 45
        Epoch: 200/800, loss: 11.456182440039912, correct: 45
        Epoch: 210/800, loss: 10.968615952930076, correct: 45
        Epoch: 220/800, loss: 10.611578801260803, correct: 45
        Epoch: 230/800, loss: 10.1393204344284, correct: 45
        Epoch: 240/800, loss: 9.725921701781406, correct: 45
        Epoch: 250/800, loss: 9.32875998341023, correct: 45
        Epoch: 260/800, loss: 8.950257006833338, correct: 45
        Epoch: 270/800, loss: 8.700479562598364, correct: 45
        Epoch: 280/800, loss: 8.431386167073372, correct: 45
        Epoch: 290/800, loss: 7.818816893567001, correct: 46
        Epoch: 300/800, loss: 7.429088029349825, correct: 47
        Epoch: 310/800, loss: 7.138555413370131, correct: 47
        Epoch: 320/800, loss: 6.618159635151138, correct: 48
        Epoch: 330/800, loss: 6.014936526190467, correct: 48
        Epoch: 340/800, loss: 5.570890502513264, correct: 49
        Epoch: 350/800, loss: 5.1046572458522235, correct: 49
        Epoch: 360/800, loss: 4.657698740289088, correct: 49
        Epoch: 370/800, loss: 4.281753436346307, correct: 49
        Epoch: 380/800, loss: 3.864601874488897, correct: 49
        Epoch: 390/800, loss: 3.5983901850696514, correct: 49
        Epoch: 400/800, loss: 3.389990603821109, correct: 49
        Epoch: 410/800, loss: 3.149216581761838, correct: 49
        Epoch: 420/800, loss: 3.0091373883786887, correct: 50
        Epoch: 430/800, loss: 2.8682896386278007, correct: 50
        Epoch: 440/800, loss: 2.7254403499685904, correct: 50
        Epoch: 450/800, loss: 2.616599014455752, correct: 50
        Epoch: 460/800, loss: 2.5164973156441013, correct: 50
        Epoch: 470/800, loss: 2.4310675482476776, correct: 50
        Epoch: 480/800, loss: 2.346134269966821, correct: 50
        Epoch: 490/800, loss: 2.304862191266229, correct: 50
        Epoch: 500/800, loss: 2.223869164543311, correct: 50
        Epoch: 510/800, loss: 2.1624193227199964, correct: 50
        Epoch: 520/800, loss: 2.106863944684386, correct: 50
        Epoch: 530/800, loss: 2.0561778139291125, correct: 50
        Epoch: 540/800, loss: 2.0100058711178095, correct: 50
        Epoch: 550/800, loss: 1.9623787591507513, correct: 50
        Epoch: 560/800, loss: 1.930258545516557, correct: 50
        Epoch: 570/800, loss: 1.894595378546751, correct: 50
        Epoch: 580/800, loss: 1.8607093794554364, correct: 50
        Epoch: 590/800, loss: 1.830158660176231, correct: 50
        Epoch: 600/800, loss: 1.806028756924987, correct: 50
        Epoch: 610/800, loss: 1.7739522169458442, correct: 50
        Epoch: 620/800, loss: 1.7474964163561741, correct: 50
        Epoch: 630/800, loss: 1.7254334584063162, correct: 50
        Epoch: 640/800, loss: 1.7026990294899542, correct: 50
        Epoch: 650/800, loss: 1.5398514732781534, correct: 50
        Epoch: 660/800, loss: 1.6616872263380447, correct: 50
        Epoch: 670/800, loss: 1.6424100997221165, correct: 50
        Epoch: 680/800, loss: 1.6260285581294076, correct: 50
        Epoch: 690/800, loss: 1.600122750044167, correct: 50
        Epoch: 700/800, loss: 1.5970649951057452, correct: 50
        Epoch: 710/800, loss: 1.569140659406843, correct: 50
        Epoch: 720/800, loss: 1.5721194759766257, correct: 50
        Epoch: 730/800, loss: 1.5320055648462598, correct: 50
        Epoch: 740/800, loss: 1.3695578137408642, correct: 50
        Epoch: 750/800, loss: 1.5189795169824358, correct: 50
        Epoch: 760/800, loss: 1.5482116319484849, correct: 50
        Epoch: 770/800, loss: 1.4258436307236182, correct: 50
        Epoch: 780/800, loss: 1.553428501993174, correct: 50
        Epoch: 790/800, loss: 1.5013671161814295, correct: 50
        Epoch: 800/800, loss: 1.4365862980731352, correct: 50
>>>>>>> 2b323a7dd4c16a8ab008d1abbd8073a8436e0b36
