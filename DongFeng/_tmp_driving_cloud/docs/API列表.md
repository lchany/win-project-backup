<table align="center" border="1" cellpadding="8" cellspacing="0">
    <tr>
        <td align="center"><strong>API类型</strong></td>
        <td align="center"><strong>API名称</strong></td>
        <td align="center"><strong>图模式</strong></td>
        <td align="center"><strong>Released</strong></td>
    </tr>
    <tr>
        <td rowspan="3">通用</td>
        <td align="center">mx_driving_cloud.unique_dim</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.dynamic_scatter_cloud</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.scatter_max_v3_cloud</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td>采样</td>
        <td align="center">mx_driving_cloud.grid_sampler3d_cloud</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="2">检测</td>
        <td align="center">mx_driving_cloud.nms3d_filter</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.nms_rotated</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="3">稀疏</td>
        <td align="center">mx_driving_cloud.SparseConv2d</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.SubMConv2d</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.SparseInverseConv3d</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="5">融合</td>
        <td align="center">mx_driving_cloud.npu_max_pool2d</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.futr3d_target_single</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.batch_matmul_cloud</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.refline_head_get_target_single</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="center">mx_driving_cloud.conv_bn_eval</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td>math</td>
        <td align="center">mx_driving_cloud.linalg.qr</td>
        <td align="center">不支持</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td>math</td>
        <td align="center">mx_driving_cloud.linalg.grid_sampler2d_cloud</td>
        <td align="center">不支持</td>
        <td align="center">N</td>
    </tr>
</table>

## FAQ
1. Released标注为N代表使用场景受限