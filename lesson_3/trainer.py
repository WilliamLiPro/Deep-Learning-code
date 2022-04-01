import os
import torch
from. import (StatsLogger, TableLogger, Timer, union,)


def acc(out, target):
    return out.max(dim=1)[1] == target


def run_batches(model, criterion, batches, training, optimizer=None, regularizer=None, stats=None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stats = stats or StatsLogger(('loss', 'correct'))

    for batch in batches:
        inp, target = batch
        inp = inp.to(device)
        target = target.to(device)

        if training:
            model.train()
            output = model(inp)
            # if torch.isnan(output).any():
            #     print('there is nan')
            output = {"loss": criterion(output, target), "correct": acc(output, target)}

            loss_out = output['loss'].sum()
            if regularizer is not None:
                loss_out = loss_out + regularizer(model)
            loss_out.backward()

            optimizer.step()
            model.zero_grad()
        else:
            model.eval()
            with torch.no_grad():
                output = model(inp)
            output = {"loss": criterion(output, target), "correct": acc(output, target)}

        stats.append(output)
    return stats


def train_epoch(model, criterion, train_batches, test_batches, optimizer, timer, regularizer=None,
                test_time_in_total=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    train_stats, train_time = \
        run_batches(model, criterion, train_batches, True, optimizer, regularizer, device=device), timer()

    test_stats, test_time = \
        run_batches(model, criterion, test_batches, False, device=device), timer(test_time_in_total)
    return {
        'train time': train_time, 'train loss': train_stats.mean('loss'),
        'train acc': train_stats.mean('correct'),
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time,
    }


def train(model, criterion, opt, train_batches, test_batches, epochs, regularizer=None,
          loggers=(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          save_path=None, test_time_in_total=True, timer=None, ):
    timer = timer or Timer()
    summaries = []

    if save_path is not None:
        print('Save model: ON. Save path = {}'.format(save_path))

    for epoch in range(epochs):
        epoch_stats = train_epoch(model, criterion, train_batches, test_batches, opt, timer, regularizer,
                                  test_time_in_total=test_time_in_total, device=device)
        summary = union({'epoch': epoch + 1,}, epoch_stats)
        summaries.append(summary)
        for logger in loggers:
            logger.append(summary)

        # save the model
        if save_path is not None:
            (filepath, temp_filename) = os.path.split(save_path)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            print('Saving model to the path = {}'.format(save_path))
            torch.save(model, save_path)
            print('Save finished')

    return summaries


def simple_trainer(model, criterion, optimizer, batch,
                   epochs, save_path=None,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    train_batch, test_batch = batch
    parameter_n = sum([param.nelement() for param in model.parameters()])
    print('Initialization: total parameter of model is {}'.format(parameter_n))

    return train(model, criterion, optimizer, train_batch, test_batch, epochs,
                 loggers=(TableLogger(),), device=device, save_path=save_path)
