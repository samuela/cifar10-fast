import argparse
import os.path

import torch.profiler
from torch.profiler import ProfilerActivity

from core import *
from dawn_utils import net, tsv
from torch_backend import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')


def main():
    args = parser.parse_args()

    print('Downloading datasets')
    dataset = cifar10(args.data_dir)

    epochs = 24
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = Network(net()).to(device).half()
    loss = x_ent_loss
    random_batch = lambda batch_size: {
        'input': torch.Tensor(np.random.rand(batch_size, 3, 32, 32)).cuda().
        half(),
        'target': torch.LongTensor(np.random.randint(0, 10, batch_size)).cuda(
        )
    }
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(dataset['valid']['targets']) % batch_size]:
        warmup_cudnn(model, loss, random_batch(size))

    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)

    print('Preprocessing training data')
    transforms = [
        partial(normalise,
                mean=np.array(cifar10_mean, dtype=np.float32),
                std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'),
    ]
    train_set = list(
        zip(*preprocess(dataset['train'], [partial(pad, border=4)] +
                        transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')

    train_batches = DataLoader(Transform(train_set, train_transforms),
                               batch_size,
                               shuffle=True,
                               set_random_choices=True,
                               drop_last=True)
    test_batches = DataLoader(test_set,
                              batch_size,
                              shuffle=False,
                              drop_last=False)
    opts = [
        SGD(
            trainable_params(model).values(), {
                'lr': (lambda step: lr_schedule(step / len(train_batches)) /
                       batch_size),
                'weight_decay':
                Const(5e-4 * batch_size),
                'momentum':
                Const(0.9)
            })
    ]
    logs, state = Table(), {MODEL: model, LOSS: loss, OPTS: opts}
    print("warmup epochs...")
    for epoch in range(3):
        logs.append(
            union({'epoch': epoch + 1},
                  train_epoch(state, timer, train_batches, test_batches)))

    print("profile epochs...")

    def trace_handler(p):
        # See https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        p.export_chrome_trace(f"profiling/chrome_trace_{p.step_num}.json")
        p.export_stacks(f"profiling/stacks_{p.step_num}.txt",
                        "self_cuda_time_total")

    # tensorboard_trace_handler doesn't seem to work AFAICT
    with torch.profiler.profile(
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(
            #     "./pytorch-profile"),
            on_trace_ready=trace_handler,
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
    ) as profiler:
        for epoch in range(5):
            logs.append(
                union({'epoch': epoch + 1},
                      train_epoch(state, timer, train_batches, test_batches)))
            profiler.step()


main()
