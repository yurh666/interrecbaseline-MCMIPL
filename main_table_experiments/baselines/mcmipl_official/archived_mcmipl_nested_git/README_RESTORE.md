# 嵌套仓库 `.git` 备份（分片）

GitHub **单文件不能超过 100MB**，而原路径 `MCMIPL/.git/objects/pack/*.pack` 约 **108MB**，无法直接提交。  
因此将 `MCMIPL/` 下的完整 `.git` 目录打成 `tar.gz` 后，用 `split -b 95m` 拆成多个分片；内容与原先嵌套克隆的历史一致。

## 还原（推荐）

在 **`MCMIPL` 目录内**解压，使 `.git` 落在 `.../MCMIPL/.git`：

```bash
cd /path/to/baselines/mcmipl_official/MCMIPL
cat ../archived_mcmipl_nested_git/mcmipl_dotgit.tar.gz.* | tar -xzf -
```

（若分片与此 README 同级，改用 `cat mcmipl_dotgit.tar.gz.* | tar -xzf -`。）

然后：

```bash
git status
git remote -v
```

## 说明

外层仓库 **`interrecbaseline-MCMIPL`** 已不再把 `MCMIPL/.git` 作为子 Git 仓库跟踪，避免「嵌套仓库」与 100MB 限制冲突；需要完整子模块历史时，用本分片还原即可。
