# IntTower

<https://github.com/archersama/IntTower/> お試し。

とりあえずWindowsで。

Pythonは3.8である必要があった。

```console
$ pip install -r requirements.txt
```

requirements.txt は制約が厳しすぎて満たせなかった。
幾つかのパッケージのバージョン指定を消した。

```
deepctr
deepctr_torch
pytorch_lightning
torchvision
```

もっと良い消し方があるかもしれない。

```
$ python train_movielens_IntTower.py
```

1 Epoch 1分くらいで進行していたが
Epoch 3 あたりからえらく遅い。
64GBのメモリを食いつぶしてスラッシングが起きてた。
とりあえずキャンセル。

その際の実行状況

```console
$ time python train_movielens_IntTower.py
1
cpu
Train on 472896 samples, validate on 118224 samples, 231 steps per epoch
Epoch 1/10
68s - loss:  2.8416 - auc:  0.7588  - val_auc:  0.8327 - accuracy:  0.8251  - val_accuracy:  0.8189 - logloss:  1.3092  - val_logloss:  0.4904
Epoch 00001: val_auc improved from -inf to 0.83271, saving model to movie_Intower.ckpt
Epoch 2/10
64s - loss:  0.4563 - auc:  0.8705  - val_auc:  0.8634 - accuracy:  0.8556  - val_accuracy:  0.8398 - logloss:  0.3825  - val_logloss:  0.3681
Epoch 00002: val_auc improved from 0.83271 to 0.86344, saving model to movie_Intower.ckpt
Epoch 3/10
74s - loss:  0.3263 - auc:  0.8868  - val_auc:  0.8623 - accuracy:  0.8624  - val_accuracy:  0.8400 - logloss:  0.3263  - val_logloss:  0.3738
Epoch 00003: val_auc did not improve from 0.86344
Traceback (most recent call last):
  File "train_movielens_IntTower.py", line 184, in <module>
    model.fit(train_model_input, train[target].values, batch_size= batch_size, epochs=epoch, verbose=2, validation_split=0.2,
  File "D:\home\koron\work\github.com\archersama\IntTower\model\base_tower.py", line 167, in fit
    contras = contrast_loss(y_contras, user_embedding, item_embedding)
  File "D:\home\koron\work\github.com\archersama\IntTower\preprocessing\utils.py", line 67, in contrast_loss
    scores = torch.matmul(user_embedding, item_embedding.t()) / tau
KeyboardInterrupt


real    6m48.613s
user    0m0.109s
sys     0m0.047s
```

Epochを2に落として再実行。完走はした。

```console
$ time python train_movielens_IntTower.py
1
cpu
Train on 472896 samples, validate on 118224 samples, 231 steps per epoch
Epoch 1/2
66s - loss:  2.8416 - auc:  0.7588  - val_auc:  0.8327 - accuracy:  0.8251  - val_accuracy:  0.8189 - logloss:  1.3092  - val_logloss:  0.4904
Epoch 00001: val_auc improved from -inf to 0.83271, saving model to movie_Intower.ckpt
Epoch 2/2
63s - loss:  0.4563 - auc:  0.8705  - val_auc:  0.8634 - accuracy:  0.8556  - val_accuracy:  0.8398 - logloss:  0.3825  - val_logloss:  0.3681
Epoch 00002: val_auc improved from 0.83271 to 0.86344, saving model to movie_Intower.ckpt
user_hist_list:

{'auc': 0.8827242115720174, 'accuracy': 0.8576938692651238, 'logloss': 0.33331138625162604}
test LogLoss 0.3235
test AUC 0.8921

real    3m17.601s
user    0m0.062s
sys     0m0.109s
```


AUC, accuracy, logloss ってなんだ?

* AUC: 二値分類タスクに対する評価指標の1つ。 1.0 に近いほど良い。
* Accuracy: 正答率 大きいほど良い。最大1.0
* LogLoss: (Binary Logarithmic Loss) 二値化ベースの正答率は微妙なズレを表現できない。確率値まんまで正解と比較して差を見る。小さいほど良い。

<https://atmarkit.itmedia.co.jp/ait/articles/2103/04/news023.html>

---

importに意外と時間がかかる。

rating が3のモノは除外している。
rating が3以上を1へ、未満を0に変換している。
残り約739万件。
学習用に80%の591万件、テスト用に20%の147万件に振り分ける。

`get_user_feature`

`get_item_feature`
