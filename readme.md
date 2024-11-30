# Commands

```
conda install --yes -c pytorch pytorch torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip3 install git+https://github.com/openai/CLIP.git
```

# Example

![cozy_room](cozy_room.jpg)

```
(base) yangjin@Ryzen-5600X:/mnt/c/Users/ChoYangJin/Desktop/workspace/CSC4018-prototype$ python3 main.py
Furniture Predictions:
a bed: 0.9937
a sofa: 0.0004
a table: 0.0036
a chair: 0.0005
a bookshelf: 0.0018

Style Predictions:
modern: 0.0099
cozy: 0.7617
luxurious: 0.1498
minimalist: 0.0024
rustic: 0.0762
```

![modern_room](modern_room.jpg)

```
(base) yangjin@Ryzen-5600X:/mnt/c/Users/ChoYangJin/Desktop/workspace/CSC4018-prototype$ python3 main.py
Furniture Predictions:
a bed: 0.0427
a sofa: 0.0670
a table: 0.6547
a chair: 0.1683
a bookshelf: 0.0674

Style Predictions:
modern: 0.8700
cozy: 0.0003
luxurious: 0.1263
minimalist: 0.0014
rustic: 0.0019
```
