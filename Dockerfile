# Base Image: RTX A6000 (CUDA 12.2) に最適化
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. 基本パッケージ・GUI・COLMAP依存ライブラリのインストール
RUN apt-get update && apt-get install -y \
    git curl wget bzip2 ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential cmake ninja-build \
    libboost-all-dev libsuitesparse-dev libfreeimage-dev \
    libgoogle-glog-dev libgflags-dev \
    libeigen3-dev libflann-dev libmetis-dev \
    libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev \
    ffmpeg libgl1-mesa-glx \
    xvfb x11vnc fluxbox novnc websockify \
    python3-pip python3-dev \
    tmux htop tree ncdu vim rsync unzip imagemagick \
    && rm -rf /var/lib/apt/lists/*

# 2. Node.js (for Claude Code) のインストール
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code

# CUDAのパスを明示的にセット
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.6"

# 3. COLMAP をCUDA対応でソースビルド（apt版より大幅に高速化）
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && git checkout 3.9 && \
    mkdir build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 && \
    ninja -j$(nproc) && ninja install && \
    rm -rf /tmp/colmap

# 4. Python環境の構築
# torch/streamlit は現環境と同じバージョンに固定（再ビルドで勝手に新しくなって壊れるのを防止）
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install numpy opencv-python tqdm scipy plyfile streamlit==1.56.0 wandb py360convert

# 5. HLoc（高精度特徴マッチング）のインストール（コミット固定: c13273b 2025-12-10時点）
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization /opt/hloc && \
    cd /opt/hloc && git checkout c13273b && \
    git submodule update --init --recursive && \
    pip3 install -e .

# 5.5. pycolmap をバージョン固定
# （HLocの依存で勝手に入るが、5.x系が入ると run_hloc.py の 4.x API前提コードが動かない）
RUN pip3 install pycolmap==4.0.3

# 5.6. SAM2マスク生成・LPIPS評価・画像クリックUI
# sam2==1.1.0: scripts/generate_masks.py（SAM2.1チェックポイントは /workspace/models/ 配下なのでDL不要）
# lpips: scripts/train_custom.py のLPIPS評価
# streamlit-image-coordinates: pages/08_sam2_masks.py のクリックUI
RUN pip3 install sam2==1.1.0 lpips streamlit-image-coordinates

# 6. 3DGS公式コードと依存サブモジュールの準備（コミット固定: 54c035f 2024-10-30時点）
WORKDIR /opt
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting && \
    cd gaussian-splatting && git checkout 54c035f && \
    git submodule update --init --recursive

WORKDIR /opt/gaussian-splatting
RUN pip3 install submodules/diff-gaussian-rasterization --no-build-isolation
RUN pip3 install submodules/simple-knn --no-build-isolation

# 7. GUI (noVNC) 用の起動スクリプト作成
# Streamlit・バッチデーモンはtmuxセッションで自動起動する
# （再起動後の手動起動が不要・tmux attach -t streamlit でログ確認可能）
RUN { \
    echo '#!/bin/bash'; \
    echo 'Xvfb :1 -screen 0 1920x1080x24 &'; \
    echo 'export DISPLAY=:1'; \
    echo 'fluxbox &'; \
    echo 'x11vnc -display :1 -nopw -forever -quiet &'; \
    echo '/usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen 6080 &'; \
    echo 'if [ -f /workspace/streamlit_app.py ]; then'; \
    echo '  tmux new-session -d -s streamlit "streamlit run /workspace/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"'; \
    echo 'fi'; \
    echo 'if [ -f /workspace/scripts/batch_daemon.py ]; then'; \
    echo '  tmux new-session -d -s daemon "python3 /workspace/scripts/batch_daemon.py"'; \
    echo 'fi'; \
    echo 'exec "$@"'; \
    } > /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
