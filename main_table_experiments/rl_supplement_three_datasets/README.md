# RL 补跑包：BOOK / LAST_FM_STAR / MOVIE（seed 0、1）

本目录说明如何在**另一台服务器**上仅跑 Phase B RL，补齐三数据集 **seed 0 与 seed 1** 的 checkpoint（seed 2 已在 CPU 归档）。

**本机（当前 CPU）** 只跑 **YELP_STAR seed 0→1→2**，与此包分离，避免抢 CPU。

## 目录内容（随 git 推送）

| 路径 | 说明 |
|------|------|
| `run_supplement_s01_serial.sh` | 一键串行补跑（screen 可选） |
| `restore_phase_a_to_tmp.sh` | 从 `experiments/.../archives/phase_a/` 恢复到 `MCMIPL/tmp/` |
| `MANIFEST.md` | 工件清单与校验 |
| `../scripts/mcmipl_*.sh` | 归档 / 串行 / 状态检查脚本 |

## 已有归档（git 内）

- `experiments/mcmipl_interrec_protocol_eval/archives/phase_a/{book,last_fm_star,movie}/`
- `experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/*/seed_2/`（参考，勿覆盖）

## 新服务器

见仓库根目录 **`docs/PROMPT_NEW_SERVER_RL_SUPPLEMENT_S01.md`**（可复制给 AI）。
