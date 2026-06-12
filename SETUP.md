# SETUP.md — コンテナ再構築・再起動ガイド

Dockerコンテナを作り直す・再起動するときに必要な作業のまとめ。

---

## 1. Dockerfile への追記（再ビルド前にやること）

現在のイメージ（4/19ビルド）以降に手動 `pip install` したパッケージは以下の3つ。
**Dockerfileに追記しないと再ビルドで消えます**：

```dockerfile
# SAM2マスク生成・評価指標・画像クリックUI
RUN pip install sam2==1.1.0 lpips streamlit-image-coordinates
```

| パッケージ | 用途 |
|---|---|
| `sam2` | SAM2マスク生成（`scripts/generate_masks.py`）。hydra-core等の依存も自動で入る |
| `lpips` | 学習時のLPIPS評価指標（`scripts/train_custom.py`） |
| `streamlit-image-coordinates` | SAM2マスクページの画像クリックUI（`pages/08_sam2_masks.py`） |

> SAM2のモデル本体（チェックポイント）は `/workspace/models/pretrained/sam2/` にあり
> NFS上なので再ビルドしても消えません。再ダウンロード不要。

---

## 2. Claude / GitHub に毎回ログインし直さないために

**原因**：`/workspace` はNFSマウントなので永続ですが、`/root`（ホームディレクトリ）は
コンテナ内部にあるため、**コンテナを作り直すたびに消えます**。ここに入っているのが：

- `/root/.claude/` と `/root/.claude.json` — Claude Codeのログイン情報・設定
- `/root/.gitconfig` と `/root/.git-credentials` — GitHubの認証トークン
- `/root/.cache/torch/` — HLocが使うSuperPoint/NetVLAD等の学習済み重み（毎回再DLしてた原因）

**対策**：`docker run` に `-v` を1つ足して `/root` を丸ごとホスト側に永続化する：

```bash
# ホスト側で一度だけ作成（chmod 700 で自分以外アクセス不可にする）
mkdir -p ~/3dgs_container_root
chmod 700 ~/3dgs_container_root
ls -ld ~/3dgs_container_root   # → drwx------ になっていればOK

# docker run に追加（--gpus all 等の既存オプションはそのまま）
docker run ... \
  -v ~/3dgs_container_root:/root \
  ...
```

これで **Claude・GitHubのログインは初回の1回だけ**になり、HLocのモデルキャッシュも
持ち越されます。docker-compose の場合は `volumes:` に `- ~/3dgs_container_root:/root` を追加。

> **共有マシンでの注意**：このディレクトリには `/root/.git-credentials`（GitHubの
> Personal Access Tokenが**平文**で入る）と Claude のログイントークンが保存される。
> `chmod 700` はディレクトリ自体への他ユーザーのアクセスを遮断するためのもの。
> 中のファイルはコンテナ（root権限）が作るため通常 root所有・600 になり二重に守られるが、
> ホストで sudo を持つ人には読める。気になる場合は GitHub 側で Fine-grained PAT を使い、
> 対象をこのリポジトリのみ・有効期限つきに絞っておくと漏れたときの被害を限定できる。

---

## 3. コンテナ再起動後のチェックリスト

```bash
# ── 1. GPU が見えるか確認 ──────────────────────────────
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# → False の場合: docker run に --gpus all が付いているか確認。
#   付いていても False ならホスト側で docker restart（NVMLエラーの典型対処）

# ── 2. Streamlit を起動（tmuxで永続化・ポート8501） ──────
tmux new-session -d -s streamlit "streamlit run /workspace/streamlit_app.py"

#   ログを見る:   tmux attach -t streamlit   （抜けるのは Ctrl+B → D）
#   再起動する:   tmux kill-session -t streamlit してから上のコマンド
#   ※ pipでライブラリを入れた/更新した後は必ず再起動すること
#     （稼働中プロセスのimportが壊れて謎のImportErrorが出ます）

# ── 3. バッチデーモン（任意） ──────────────────────────
# GUIからキューに追加すると自動起動されるので、手動起動は必須ではない
nohup python3 /workspace/scripts/batch_daemon.py > /dev/null 2>&1 &
# 状態確認: キューページ上部の「🟢稼働中 / 🔴停止中」表示

# ── 4. ログイン（/root を永続化していない場合のみ） ─────
claude          # 起動後に /login → ブラウザ認証
git push        # 初回にユーザー名 + Personal Access Token を入力
                # （credential.helper=store 設定済みなので2回目以降は不要）
```

### 動作確認（全部そろったかの最終チェック）

```bash
python3 -c "import torch, sam2, lpips, streamlit_image_coordinates, pycolmap, scipy; \
print('CUDA:', torch.cuda.is_available()); print('imports: OK')"
curl -s -o /dev/null -w "Streamlit: HTTP %{http_code}\n" http://localhost:8501
```

---

## 補足：よくあるトラブル

| 症状 | 対処 |
|---|---|
| `nvidia-smi` が `Failed to initialize NVML` | ホスト側で `docker restart <コンテナ名>`。直らなければホストのNVIDIAドライバとdocker runの `--gpus all` を確認 |
| Streamlitで突然 `ImportError`（PIL等） | pip更新後にサーバーを再起動していないのが原因。tmuxセッションを作り直す |
| キューが進まない | キューページでデーモンが🔴停止中になっていないか確認 → ▶起動 |
| `claude` コマンドがない | Dockerfileの `npm install -g @anthropic-ai/claude-code` が入っているか確認 |
