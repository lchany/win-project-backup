跳板机机器：
IP地址：101.245.102.26
端口：22
账号：root
密码：4fBFVQr54Qrjfkkv7noR


npu训练机器：
10.199.148.42   root   密码：DYyZdKyZKAn8v4SHHpMo
端口：2288


注意：npu训练机器只能通过本机ssh到跳板机机器后，再ssh到npu训练机器
两个机器都能访问共同的地址：/mnt/sfs_turbo/workdir/wfc1_leicheng  
性能优化请严格在：asend_npu_optimize分支代码上修改

建议：修改代码，直接在跳板机机器上访问挂载盘修改即可
训练机器，需要到npu训练机器上执行
