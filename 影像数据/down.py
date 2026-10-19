from modelscope import dataset_snapshot_download

dataset_dir = dataset_snapshot_download(
    'jmz001/Examples',
    local_dir=r'G:\DSH\CH+AI\低空政务巡查智能 Agent\影像数据'
)

print(dataset_dir)