# README

这个项目基于 `mdbook` 发布材料，如果你想进行贡献并在本地预览该书，请按以下流程进行配置。

## 安装 Rust

`mdbook` 是用 rust 编写的，故需先安装 rust，在 https://www.rust-lang.org/tools/install 获取所需脚本。

## 安装 mdbook 并配置插件

运行以下命令

```shell
cargo install mdbook # 安装 mdbook

cargo install mdbook-katex # 数学公式插件

cargo install mdbook-admonish # Hint样式插件

cargo install mdbook-pdf # pdf导出插件（需要开启终端代理）

pip install mdbook-pdf-outline # 支持导出pdf目录
```

## 使用 mdbook

完成内容更改后，可使用以下命令进行预览。

```shell
mdbook build --open
```

该命令会在 `./book` 目录下生成相应文件，**该目录下的文件不应当被提交**。
