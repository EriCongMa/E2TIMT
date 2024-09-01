import torch

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print('Stopped! Because Only CPU could be available ...')
    exit()
else:
    print('Now is using device: {}'.format(device))
criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

def contrastive_loss_cal_moco_v3(q, k, tau):
    assert q.shape == k.shape

    logits = torch.mm(q, k.t())
    labels = torch.arange(0, q.shape[0])
    
    preds = (logits / tau).contiguous().to(device)
    labels = labels.contiguous().view(-1).to(device)
    
    loss = criterion(preds, labels)

    return 2 * tau * loss
