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

## 2. Claude / GitHub のログインについて

**仕組み**：`/workspace` はNFSマウントなので永続だが、`/root`（ホームディレクトリ）は
コンテナ内部にあるため、**コンテナを「作り直す」と消える**。ここに入っているのは：

- `/root/.claude/` と `/root/.claude.json` — Claude Codeのログイン情報・設定
- `/root/.gitconfig` と `/root/.git-credentials` — GitHubの認証トークン
- `/root/.cache/torch/` — HLocが使うSuperPoint/NetVLAD等の学習済み重み

**重要な区別**：

| 操作 | /root | ログインし直し |
|---|---|---|
| `docker restart`（再起動） | **消えない** | **不要** |
| `docker rm` → `docker run`、イメージ再ビルド（作り直し） | 消える | 必要（下記手順） |

頻繁に作り直すわけではないので、**作り直したときだけ以下を再実行する**運用とする
（共有マシンに認証トークンを永続保存しない、というセキュリティ上の利点もある）。

### 作り直した後のログイン手順

```bash
# ① Claude Code
claude            # 起動して /login → 表示されるURLをブラウザで開いて認証

# ② Git（ユーザー名・メールはリポジトリ側 .git/config に保存済みなので不要）
git config --global credential.helper store

# ③ GitHub認証（初回のpush時に聞かれる）
cd /workspace && git push
#   Username: bushdoge
#   Password: Personal Access Token（GitHubの Settings > Developer settings で発行）
#   → ②の設定により2回目以降は聞かれない
```

> PATは Fine-grained PAT で「このリポジトリのみ・Contents Read/Write・有効期限つき」に
> 絞って発行しておくと、漏れたときの被害を限定できる。

> HLocのモデル重み（`/root/.cache/torch/`）も消えるが、これは初回実行時に
> 自動で再ダウンロードされるので何もしなくてよい（初回の姿勢推定だけ数分遅くなる）。

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

# ── 4. ログイン（コンテナを「作り直した」場合のみ。docker restart なら不要） ─
# 詳細な手順は上の「2. Claude / GitHub のログインについて」を参照
claude                                        # 起動後に /login → ブラウザ認証
git config --global credential.helper store   # トークン保存を有効化
git push                                      # 初回にユーザー名 + PAT を入力（以降は不要）
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
