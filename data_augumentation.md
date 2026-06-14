# data_augmentation.py 調査結果

## 推測される意図

`diffusion_planner/diffusion_planner/utils/data_augmentation.py` の `StatePerturbation` は、training時にego current stateをランダムに摂動し、その摂動後のcurrent stateから元のfuture trajectoryへ滑らかにつながるego futureを生成するためのaugmentationと考えられる。

実装の流れは以下。

1. `augment()` で、速度が十分大きいサンプルだけ `aug_flag=True` にする。
2. ego current state の横位置、heading、速度、加速度などをランダムに摂動する。
3. `interpolation_future_trajectory()` で、摂動後current stateから元futureの途中点へつながるquintic trajectoryを作る。
4. `aug_flag=True` のサンプルだけ current state と ego future を置き換える。
5. `aug_flag=True` のサンプルでは `ego_agent_past` を全0にする。
6. `centric_transform()` で、摂動後のego current poseを原点・heading 0の座標系に戻す。
7. neighbor、map、static object、futureなども同じ座標変換で新ego中心座標に変換する。

## 現在できていること

- augmentationが実際に適用されたサンプルでは、`inputs["ego_agent_past"][aug_flag] = 0.0` によりego historyは消える。
- 速度が低いサンプルは `aug_flag=False` になり、ego historyは保持される。
- `ego_current_state`、`ego_future`、neighbor past/future、lane、route、polygon、line string、static objectは、基本的には摂動後ego current pose基準の座標系へ変換される。
- 空要素は変換後に0へ戻す処理が入っている。

## 修正済みの問題

### 1. `goal_pose` が新しいego座標系へ変換されていなかった

`train_epoch.py` ではaugmentation前に `goal_pose` が `heading_to_cos_sin()` で `(x, y, cos, sin)` に変換される。しかし `centric_transform()` は `goal_pose` を変換していない。

そのため、augmentationでego current poseを変えたサンプルでは、他のscene tokenは新しいego frameへ移る一方、`goal_pose` だけ古いego frameのまま残る。Encoderは `goal_pose` を使用しているため、これは意図に反する可能性が高い。

再現確認でも、augmentation後に `goal_pose` は入力値のまま変化しなかった。

該当箇所:
- `diffusion_planner/diffusion_planner/utils/data_augmentation.py`: `centric_transform()` 内に `goal_pose` 変換がない
- `diffusion_planner/diffusion_planner/train_epoch.py`: augmentation前に `goal_pose` はcos/sin化済み

修正:

- `goal_pose[..., :2]` を座標変換する。
- `goal_pose[..., 2:4]` をcos/sinベクトルとして回転する。
- 全0のgoal poseは変換後も0へ戻す。

### 2. smoothing用の `ego_past4d` が現在の入力形式と合っていなかった

`train_epoch.py` では、augmentation前に `ego_agent_past` は `heading_to_cos_sin()` により `(x, y, cos, sin)` へ変換される。

しかし `centric_transform()` は以下のように `inputs["ego_agent_past"][..., 2:3]` をheading角として扱い、さらに `cos()` / `sin()` を取っている。

```python
ego_past4d = torch.cat(
    [
        inputs["ego_agent_past"][..., :2],
        torch.cos(inputs["ego_agent_past"][..., 2:3]),
        torch.sin(inputs["ego_agent_past"][..., 2:3]),
    ],
    dim=-1,
)
```

現在の入力では `[..., 2]` はheading角ではなく `cos(heading)` なので、`cos(cos(theta))` / `sin(cos(theta))` になってしまう。

`use_smoothing_future_trajectory=True` の場合、smoothingに誤ったego history headingが渡る。さらに、augmentation適用サンプルでは直前に `ego_agent_past` を0にしているため、smoothingは消去済みhistoryを参照する。これは「モデル入力としてhistoryを消す」意図とは別問題で、future smoothingの境界条件まで壊す可能性がある。

修正:

- `ego_agent_past` は既に `(x, y, cos, sin)` なので、再度 `cos()` / `sin()` しない。
- augmentation適用サンプルは入力historyを0に保ち、future smoothing対象から除外する。
- 非augmentationサンプルだけ、変換済みego historyを使ってsmoothingする。

### 3. 摂動乱数が9次元独立になっていなかった

`lo` / `hi` はタプルにリストを入れた形で定義されているため、`self._low.shape == (1, 9)` になる。

```python
lo = ([0.0, -0.75, -0.2, -1, -0.5, -0.2, -0.1, 0.0, 0.0],)
```

その後、`random_tensor = torch.rand(B, len(self._low))` としているため、`len(self._low) == 1` になり、乱数は `(B, 1)` しか生成されない。broadcastにより9要素へ展開されるため、横位置、heading、速度、加速度などの摂動が同じ1つの乱数に強く連動する。

これは、おそらく9次元それぞれを独立に摂動する意図と合っていない。

再現確認:

```text
low_shape (1, 9)
len_low 1
random_shape (4, 1)
scaled_shape (4, 9)
```

修正:

- `lo` / `hi` をshape `(9,)` のtensorにする。
- `torch.rand(B, self._low.shape[0])` を使い、9次元を独立にsampleする。

### 4. `ego_past_noise_std` は現在使われていなかった

以前はego pastをscaleする処理に使われていたが、現在はaugmentation時にego historyを消す設計になったため、`self._ego_past_noise_std` は保持されるだけで使用されない。

これは実害のあるバグではないが、config/APIとしては混乱を招く。削除するか、互換性のために残すなら未使用であることを明記した方がよい。

修正:

- `StatePerturbation.__init__` から `ego_past_noise_std` を削除。
- `train_predictor.py` のCLI引数と生成時引数からも削除。

### 5. `num_refine` とfuture lengthの関係にガードがなかった

`interpolation_future_trajectory()` は `ego_future[:, P]`、`ego_future[:, P - 1]`、`ego_future[:, P - 2]` を参照する。現在のdefaultでは `num_refine=20`、future lengthは80なので成立するが、設定を変えると範囲外参照や不自然な計算が起きる。

少なくとも `2 <= num_refine < ego_future.shape[1]` の前提がある。

修正:

- `interpolation_future_trajectory()` の先頭で `2 <= num_refine < future_len` を満たさない場合に `ValueError` を出す。

## 結論

調査で見つけた問題は修正済み。

修正後の挙動は以下。

1. augmentation適用サンプルではego history入力を全0にする。
2. augmentation適用サンプルでは、消去済みhistoryを使ったfuture smoothingは行わない。
3. 非augmentationサンプルでは、cos/sin形式のego historyを正しく変換してsmoothingへ渡す。
4. `goal_pose` を他のscene tokenと同じ新ego frameへ変換する。
5. 摂動乱数は9次元独立にsampleされる。
6. `num_refine` の不正設定は明示的にエラーになる。
