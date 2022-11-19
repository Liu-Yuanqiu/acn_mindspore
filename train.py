import os
os.environ['GLOG_v'] = '3'
import sys
import argparse

from coco_dataset import CocoDataset
from capsulexlan import CapsuleXlan, CapsuleXlanWithLoss, CapsuleXlanTrainOneStepCell

import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from  mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint, SummaryCollector
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='ACN in MindSpore')
    parser.add_argument("--device", type=str, default='Ascend')
    parser.add_argument("--device_id", type=int, default=0)
    # Train
    parser.add_argument("--works", type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--result_folder', type=str, default='./ACN_RESULT')
    parser.add_argument("--dataset_path", type=str, default='./mscoco')
    parser.add_argument("--epochs", type=int, default=50)
    # Model
    parser.add_argument("--vocab_size", type=int, default=9487)
    parser.add_argument("--seq_per_img", type=int, default=5)
    parser.add_argument("--encode_layer_num", type=int, default=3)
    parser.add_argument("--decode_layer_num", type=int, default=3)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    ms.set_seed(1)
    # PYNATIVE_MODE GRAPH_MODE
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target=args.device, device_id=args.device_id)

    coco_train_set = CocoDataset(            
        image_ids_path = os.path.join(args.dataset_path, 'txt', 'coco_train_image_id.txt'),
        input_seq = os.path.join(args.dataset_path, 'sent', 'coco_train_input.pkl'),
        target_seq = os.path.join(args.dataset_path, 'sent', 'coco_train_target.pkl'),
        att_feats_folder = os.path.join(args.dataset_path, 'feature', 'up_down_36'),
        seq_per_img = args.seq_per_img,
        max_feat_num = -1
    )
    # ds.config.set_enable_shared_mem(False)
    dataset_train = ds.GeneratorDataset(coco_train_set, 
        column_names=["indices", "input_seq", "target_seq", "att_feats"],
        shuffle=True,
        python_multiprocessing=True,
        num_parallel_workers=args.works)
    dataset_train = dataset_train.batch(args.batch_size, drop_remainder=True)
    step_per_epoch = dataset_train.get_dataset_size()

    net = CapsuleXlan(args.vocab_size, args.seq_per_img, args.encode_layer_num, args.decode_layer_num)
    net = CapsuleXlanWithLoss(net)
    warmup_lr = nn.WarmUpLR(args.lr, args.warmup)   
    optim = nn.Adam(params=net.trainable_params(), learning_rate=warmup_lr, beta1=0.9, beta2=0.98, eps=1.0e-9)
    net = CapsuleXlanTrainOneStepCell(net, optim)
    net.set_train(True)
    model = ms.Model(network=net, boost_level="O1")
    
    loss_cb = LossMonitor(per_print_times=1)
    time_cb = TimeMonitor(data_size=step_per_epoch)
    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=args.epochs)
    ckpoint_cb = ModelCheckpoint(prefix='ACN', directory=os.path.join(args.result_folder, 'checkpoints'), config=config_ck)
    specified={"collect_metric": True, "collect_graph":False, "collect_train_lineage":False, "collect_eval_lineage":False, "collect_input_data":False, "collect_dataset_graph":False, "histogram_regular":None}
    summary_cb = SummaryCollector(summary_dir=os.path.join(args.result_folder, 'summarys'), collect_specified_data=specified, collect_freq=1, keep_default_action=False, collect_tensor_freq=200)
    cbs = [loss_cb, time_cb, ckpoint_cb, summary_cb]
    model.train(epoch=args.epochs, train_dataset=dataset_train, callbacks=cbs, dataset_sink_mode=False)
    print("well done!")