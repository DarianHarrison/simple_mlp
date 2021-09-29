# simple_mlp
a simple MLP network written in Rust and C++ torch bindigs


# Prereqs

1. Have Rust on your machine or container 

here is the way to install it in case you dont have it
```
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```
optionally persist the rust env to bash profile
```
cat << 'EOF' >> ~/.bashrc
. "$HOME/.cargo/env"
EOF
```
test
```
rustc --version
```
result
```
darian@darian-laptop:~$ rustc --version
rustc 1.54.0 (a178d0322 2021-07-26
```

2. Set up Torch C++ on your local machine or container

download libtorch (C++-only) torch bindings
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip
unzip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip
```
set up environment paths for session (this may be different on each OS, my case its an ubuntu 20.04 desktop)
```
export LIBTORCH=$HOME/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

for convencience, lets persist the above env variables across terminal restarts by appending the following to your bash profile (this may be different on each OS, my case its an ubuntu 20.04 desktop)
```
cat << 'EOF' >> ~/.bashrc
export LIBTORCH=/home/darian/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
EOF
```
refresh profile
```
soruce ~/.bashrc
```

3. Clone this repo
```
cd ~/
git clone https://github.com/DarianHarrison/simple_mlp
```

4. Data files are already pre-processed and saved in the 'dataset' directory

These files 4 should be extracted in the 'data' directory.
```
.
├── Cargo.lock
├── Cargo.toml
├── config
│   └── Default.toml
├── dataset
│   ├── x_concat.pt
│   ├── x_labels.pt
│   ├── y_concat.pt
│   ├── y_concat_unlabeled.pt
│   └── y_labels.pt
├── LICENSE
├── README.md
├── src
│   ├── gnn
│   │   ├── mlp.rs
│   │   ├── mod.rs
│   │   └── prediction.rs
│   ├── lib.rs
│   ├── main.rs
│   └── settings.rs
```

# Run

1. Ensure the code is compileable
```
cd simple_mlp/
cargo build
cargo check
```
2. Run linear classifier, then an mlp, then the convolutional netowrk
```
cargo run mlp
cargo run predict
```

# Results

Results over very few Epochs
```
darian@darian-laptop:~/Desktop/simple_mlp$ cargo run mlp
    Finished dev [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/rhagle mlp`
loading file from: "/home/darian/Desktop/simple_mlp/dataset/x_concat.pt"
loading file from: "/home/darian/Desktop/simple_mlp/dataset/x_labels.pt"
loading file from: "/home/darian/Desktop/simple_mlp/dataset/y_concat.pt"
loading file from: "/home/darian/Desktop/simple_mlp/dataset/y_labels.pt"
epoch:    1 train loss:  0.69522 test acc: 55.56%
epoch:    2 train loss:  0.68781 test acc: 77.78%
epoch:    3 train loss:  0.67957 test acc: 77.78%
epoch:    4 train loss:  0.67072 test acc: 77.78%
epoch:    5 train loss:  0.66157 test acc: 77.78%
epoch:    6 train loss:  0.65228 test acc: 77.78%
epoch:    7 train loss:  0.64289 test acc: 77.78%
epoch:    8 train loss:  0.63351 test acc: 83.33%
epoch:    9 train loss:  0.62410 test acc: 83.33%
epoch:   10 train loss:  0.61471 test acc: 83.33%
epoch:   11 train loss:  0.60529 test acc: 83.33%
epoch:   12 train loss:  0.59587 test acc: 83.33%
epoch:   13 train loss:  0.58653 test acc: 88.89%
epoch:   14 train loss:  0.57708 test acc: 88.89%
epoch:   15 train loss:  0.56756 test acc: 88.89%
epoch:   16 train loss:  0.55812 test acc: 88.89%
epoch:   17 train loss:  0.54888 test acc: 88.89%
epoch:   18 train loss:  0.53975 test acc: 88.89%
epoch:   19 train loss:  0.53065 test acc: 88.89%
epoch:   20 train loss:  0.52164 test acc: 88.89%
epoch:   21 train loss:  0.51274 test acc: 88.89%
epoch:   22 train loss:  0.50392 test acc: 88.89%
epoch:   23 train loss:  0.49526 test acc: 88.89%
epoch:   24 train loss:  0.48679 test acc: 88.89%
epoch:   25 train loss:  0.47851 test acc: 88.89%
epoch:   26 train loss:  0.47041 test acc: 88.89%
epoch:   27 train loss:  0.46259 test acc: 88.89%
epoch:   28 train loss:  0.45507 test acc: 94.44%
epoch:   29 train loss:  0.44782 test acc: 94.44%
```
```
darian@darian-laptop:~/Desktop/simple_mlp$ cargo run predict
    Finished dev [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/rhagle predict`
loaded model from: "/home/darian/Desktop/simple_mlp/weights.pt"
[0.38273391127586365, 0.6172661185264587]
```

# References