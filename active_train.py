import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import *
from eval import evaluate
from utils import *

import numpy as np
import argparse 
import torch.nn.functional as F



# Data parameters
data_folder = './voc/'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Learning parameters
# checkpoint = None
checkpoint = './checkpoint_ssd300.pth.tar'
# checkpoint = './SSD_40_epochs_dropout_0.1.pth.tar'
batch_size = 8  # batch size
epochs = 27  # number of epochs for each acquisition iteration
workers = 4 # number of workers for loading data in the DataLoader
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation


cudnn.benchmark = True


# active learning params
init_train_size = 500
print_freq = np.floor(init_train_size/batch_size) # print training or validation status every __ batches
acquisition_iterations = 14
num_of_queries = 250
pool_subset = 500
assert(pool_subset > num_of_queries)
save_al_checkpoints = False


# eval params
min_score = 0.01
max_overlap = 0.45
top_k = 200


def main():
    """
    Training and validation.
    """
    # Training settings
    parser = argparse.ArgumentParser(description='SSD VOC AL')
    parser.add_argument('--trial_number', type=int, default=1, metavar='N',
                        help='trial number for given acquisition function (default: 1)')
    parser.add_argument('--acquisition_function', type=str, default='RANDOM', metavar='N',
                        help='type of acquisition. Options are: RANDOM, MARGIN_SAMPLING')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--save_dir', type=str, default='./results/', metavar='N',
                         help='directory to save to  (default: ./result/)')
    parser.add_argument('--reset_weight',type=bool, default=False, metavar='N',
                        help='reset network weights (default: False')
    args = parser.parse_args()

    print("Training with the following acquisition function: ", args.acquisition_function)
    print("Training for trial #: ", args.trial_number)

    global epochs_since_improvement, start_epoch, label_map, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SSD300(n_classes=n_classes)     
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}],
                                    lr=args.lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        print("Loading checkpoint model.")
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

        # use lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr



    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Read data files, partition train/pool and test set
    with open(os.path.join(data_folder, 'TRAIN_images.json'), 'r') as j:
        train_images = json.load(j)
    with open(os.path.join(data_folder, 'TRAIN_objects.json'), 'r') as j:
        train_objects = json.load(j)
    all_train_indices = np.arange(len(train_images))

    
    train_indices = all_train_indices[:init_train_size]
    pool_indices = all_train_indices[init_train_size:5011] #ONLY VOC 2007, remove 5011 to use all data
    
    with open(os.path.join(data_folder, 'TEST_images.json'), 'r') as j:
        test_images = json.load(j)
    with open(os.path.join(data_folder, 'TEST_objects.json'), 'r') as j:
        test_objects = json.load(j)
    all_test_indices = np.arange(len(test_images))
    test_indices = all_test_indices

    # Custom dataloaders
    train_dataset = TrainDataset(train_images, train_objects, train_indices,
                                    keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    pool_dataset = PoolDataset(train_images, train_objects, pool_indices,
                                    pool_subset, num_of_queries,
                                    keep_difficult=keep_difficult)

    val_dataset = TestDataset(test_images, test_objects, test_indices,
                                    keep_difficult=keep_difficult)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=val_dataset.collate_fn, num_workers=workers,
                                                pin_memory=True)

    training_data = list()
    training_data.append(train_indices)
    mAP_list = list()
    APs_list = list()

    for i in range(acquisition_iterations):
        #grab new data from pool 
        img_indices = list()
        if (args.acquisition_function == 'MARGIN_SAMPLING'):
            print("Doing one iteration of margin sampling")
            img_indices = marginSampleAcquisition(pool_dataset, model)
        elif (args.acquisition_function == 'RANDOM'):
            print("Doing one iteration of random sampling")
            img_indices = randomSampleAcquisition(pool_dataset)
        elif (args.acquisition_function == 'ENTROPY' or args.acquisition_function == 'BALD' or 
            args.acquisition_function == 'VAR_RATIO' or args.acquisition_function == 'MEAN_STD' or 
            args.acquisition_function == 'MEAN_STD_WITH_BBOX'):
            print("Doing one iteration of ", args.acquisition_function)
            img_indices = dropoutAcquisition(pool_dataset, model, args.acquisition_function, dropout_iterations=10)
        elif (args.acquisition_function == 'LOCALIZATION_STABILITY'):
            img_indices = localizationAwareAcquisiton(pool_dataset, model)
        elif (args.acquisition_function == 'QBC'):
            img_indices = queryByCommittee(pool_dataset, model)
        else:
            print("UNKNOWN ACQUISITION FUNCTION")
            exit()

       
        # reset weights before training
        if args.reset_weight:
            # checkpoint = torch.load(checkpoint)
            print("Loading checkpoint model.")
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            # use lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        pool_dataset.removeFromPool(img_indices)
        train_dataset.addFromPool(img_indices)
        training_data.append(img_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn, num_workers=workers,
                                            pin_memory=True)  
                                            # note that we're passing the collate function here



        # We can train on new data for certain number of iterations rather than epochs
        # XXXX iterations
        # epochs = int(80000 / len(train_dataset.indices))
        # print("training_data len: ", len(train_dataset.indices))
        # decay_epoch = int(60000 / len(train_dataset.indices))

        for epoch in range(epochs):
            print('Epochs: ', epoch, ' / ', epochs)
            print('Learning rate is: ', optimizer.param_groups[1]['lr'])

            # if epoch == decay_epoch:
            #     adjust_learning_rate(optimizer, 0.1) # decay by factor of 0.1

            # One epoch's training
            train(train_loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch)

        # check performance
        APs, mAP = evaluate(val_loader, model)
        mAP_list.append(mAP)
        APs_list.append(APs)

        if save_al_checkpoints:
            save_checkpoint_active_learning(args.acquisition_function, i,  model, optimizer, mAP, training_data)

    # store accuracy
    print('Storing Accuracy Values over experiments')
    save_str = args.save_dir + args.acquisition_function + '_' + str(args.trial_number) + '_test_acc.npy'
    mAP_list = np.array(mAP_list)
    np.savez(save_str, mAP_list=mAP_list, training_data=training_data, APs_list=APs_list)

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    # free some memory since their histories may be stored
    del predicted_locs, predicted_scores, images, boxes, labels  

def randomSampleAcquisition(pool_dataset):
    start = time.time()

    pool_dataset.resampleSubsetIndices()
    
    total_time = time.time() - start
    print("Time required for random sampling: ", total_time)
    print("Time per image, ", total_time/pool_subset)

    img_indices = pool_dataset.indices[:pool_dataset.num_of_queries]
    return img_indices

def marginSampleAcquisition(pool_dataset, model):
    """iterate through pool, compute 
    
    :param train_loader: DataLoader for training data
    :param model: model
    """
    # extract pool subset randomly and initialize pool_loader
    pool_dataset.resampleSubsetIndices()
    pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32, shuffle=False,
                                               collate_fn=pool_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    model.eval()  # eval mode disables dropout

    margins = list()
    image_indices = list()

    start = time.time()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties, indices) in enumerate(pool_loader):
            print("batch #: ", i)

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

            # operating on all bboxes
            sorted_scores, sorted_scores_ind = torch.sort(predicted_scores, dim=2)     
            batch_margin = 1.0 - sorted_scores[:, :, -1] - sorted_scores[:, :, -2] # (N, 8732)
            
            batch_margin = torch.mul(batch_margin, batch_margin)            
            batch_margin = torch.mean(batch_margin, axis=1)
            margins.append(batch_margin)

            image_indices.extend(indices)

    # get index of K smallest margins
    margins = torch.cat(margins)
    margins = margins.cpu().numpy()

    # look for images with the smallest margins
    best_ind = np.argsort(margins)[:num_of_queries]

    total_time = time.time() - start
    print("Time required for margin sampling: ", total_time)
    print("Time per image, ", total_time/pool_subset)

    return [image_indices[i] for i in best_ind]

def dropoutAcquisition(pool_dataset, model, acquisition_type, dropout_iterations=20):
    # extract pool subset randomly and initialize pool_loader
    pool_dataset.resampleSubsetIndices()
    pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32, shuffle=False,
                                               collate_fn=pool_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  
                                               # seems like you can use larger batch size for eval

    model.train()  # enable dropout

    metrics = list()
    image_indices = list()

    print("Dropout Acquisition with, ", acquisition_type)
    print("Dropout iterations: ", dropout_iterations)

    start = time.time()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties, indices) in enumerate(pool_loader):
            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            locs = list()
            scores = list()

            # Forward prop.
            print("Starting dropout")
            for j in range(dropout_iterations):
                predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
                predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
                locs.append(predicted_locs)
                scores.append(predicted_scores)
            print("Finished dropout")

            locs = torch.stack(locs) # (dropout_iterations, N, 8732, 4)
            scores = torch.stack(scores) # (dropout_iterations, N, 8732, n_classes)

            locs_mean = torch.mean(locs, axis=0) #(N, 8732, 4)
            scores_mean = torch.mean(scores, axis=0) #(N, 8732, n_classes)

            #we will look for min SUM qlog(q)
            if (acquisition_type == 'ENTROPY'):
                detection_entropy = torch.sum(torch.mul(scores_mean, torch.log(scores_mean)), axis=2 ) #(N, 8732)
                entropy = torch.sum(detection_entropy, axis=1) # (N)
                metrics.append(entropy)

            elif (acquisition_type == 'BALD'):
                detection_entropy = torch.sum(torch.mul(scores_mean, torch.log(scores_mean)), axis=2 ) #(N, 8732)
                entropy = torch.sum(detection_entropy, axis=1) # (N)

                expected_entropy = torch.sum(torch.mul(scores, torch.log(scores)), axis=0 ) #(N, 8732, n_classes)
                expected_entropy = torch.sum(expected_entropy, axis=2) # (N, 8732)
                expected_entropy = torch.sum(expected_entropy, axis=1) # (N)
                expected_entropy = torch.div(expected_entropy, dropout_iterations)

                bald = entropy - expected_entropy
                metrics.append(bald)

            elif (acquisition_type == 'MEAN_STD'):
                scores_var = torch.var(scores, axis=0) #(N, 8732, n_classes) 
                scores_var = torch.mean(scores_var, axis=2) # mean across classes
                scores_var = torch.mean(scores_var, axis=1) # mean of var across bboxes in training ex.

                # we will look for samples with the greatest variance
                # or min -var
                metrics.append(-scores_var)

            elif (acquisition_type == 'VAR_RATIO'):
                max_cls_score, max_ind = torch.max(scores_mean, dim=2)
                max_cls_score, max_ind = torch.max(max_cls_score, axis=1)
                var_ratio = 1 - max_cls_score
                metrics.append(-var_ratio)

            elif (acquisition_type == 'MEAN_STD_WITH_BBOX'):
                scores_var = torch.var(scores, axis=0) #(N, 8732, n_classes) 
                scores_var = torch.mean(scores_var, axis=2) # mean across classes
                scores_var = torch.mean(scores_var, axis=1) # mean of var across bboxes in training ex.

                locs_var = torch.var(locs, axis=0) #(N, 8732, 4) 
                locs_var = torch.mean(locs_var, axis=2)
                locs_var = torch.mean(locs_var, axis=1)

                metrics.append(-scores_var - locs_var)

            else:
                print("Unsupported dropout acquisition type")

            image_indices.extend(indices)

            # Print status
            print('Batch: [{0}/{1}]\t'.format(i, len(pool_loader)))

    metrics = torch.cat(metrics)
    metrics = metrics.cpu().numpy()

    # look for images with the smallest score
    best_ind = np.argsort(metrics)[:num_of_queries]

    
    total_time = time.time() - start
    print("Time required for ", acquisition_type, " ", total_time)
    print("Time per image, ", total_time/pool_subset)

    return [image_indices[i] for i in best_ind]

def localizationAwareAcquisiton(pool_dataset, model, use_loc_stab=False):
    """implementation of "Localization-Aware Active Learning for Object Detection"
    using the localization stability and classication uncertainty metric mentioned in the paper.
    
    """
    # extract pool subset randomly and initialize pool_loader
    pool_dataset.resampleSubsetIndices()
    pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32, shuffle=False,
                                               collate_fn=pool_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    model.eval()  # eval mode disables dropout

    scores = list()
    image_indices = list()

    # let P = pool size
    reference_locs = list() # P x (8732 x 4)
    max_prob = list() # P x 8732, max_prob of each bounding box for each img
    noisy_locs = list() # N x P x (8732 x 4)
    classification_uncertainty = list()

    print("Acquisition with localization stability w/ classification uncertainty")

    noise_levels = [8, 16, 24, 32, 40, 48]

    start = time.time()

    with torch.no_grad():
        # acquire reference bboxes and classification scores
        print("Computing reference boxes")
        for i, (images, boxes, labels, difficulties, indices) in enumerate(pool_loader):
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            image_indices.extend(indices)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

            # Store max prob for each bbox and store all bbox locations
            # compute classifcation uncertainty for image as well
            for b in range(predicted_scores.shape[0]):
                cls_scores = predicted_scores[b]
                max_scores, _ = torch.max(cls_scores, dim=1) # (8732), max returns (vals, inds)
                max_prob.append(max_scores) 

                filtered_locs = model.decode_locations(predicted_locs[b])
                reference_locs.append(filtered_locs)

                max_uncertainty = torch.max(max_scores) 
                classification_uncertainty.append(1.0 - max_uncertainty) 

        classification_uncertainty = torch.stack(classification_uncertainty) # (P)

        # acquire bboxes for various noise levels
        for level in noise_levels:
            print("Computing bbox at noise level ", level)

            pool_dataset.gaussian_noise = level 
            # note that there is no shuffle, so order of images acquired remains the same
            pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32, shuffle=False,
                                               collate_fn=pool_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

            locs = list()
            for i, (images, boxes, labels, difficulties, indices) in enumerate(pool_loader):
                images = images.to(device)  # (N, 3, 300, 300)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                # Forward prop.
                predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
                predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

                for b in range(predicted_scores.shape[0]):
                    filtered_locs = model.decode_locations(predicted_locs[b])
                    locs.append(filtered_locs)    

            noisy_locs.append(locs)                 

    #restore noise levels
    pool_dataset.gaussian_noise = 0

    # find iou between noisy bbox and reference bbox
    all_ious = list()
    for noisy_bboxes in noisy_locs:
        image_ious = list()
        for i in range(len(reference_locs)): # iterate through imgs
            iou = find_jaccard_overlap(reference_locs[i], noisy_bboxes[i]) #(8732, 8732)
            iou = torch.diag(iou)
            image_ious.append(iou)
        image_ious = torch.stack(image_ious)
        all_ious.append(image_ious)
    all_ious = torch.stack(all_ious)
   
    loc_stab_per_box = torch.mean(all_ious, dim=0) # (Px8732), S_B(B_0^j) in paper
    max_prob = torch.stack(max_prob) # (P x 8732)

    loc_stab = torch.sum(max_prob * loc_stab_per_box, dim=1) / torch.sum(max_prob, dim=1) # (P)
   
    # we want samples with high classiciation uncertainty and low loc_stab score
    # in other words, get samples with highest scores 
    scores = classification_uncertainty - loc_stab
    scores = scores.cpu().numpy()

    # argsort goes from smallest to largest
    best_ind = np.argsort(scores)[-num_of_queries:]

    total_time = time.time() - start
    print("Time required for LOCSTAB ", total_time)
    print("Time per image, ", total_time/pool_subset)


    return [image_indices[i] for i in best_ind]



def queryByCommittee(pool_dataset, model):
    """iterate through pool, compute 
    
    :param train_loader: DataLoader for training data
    :param model: model
    """

    # extract pool subset randomly and initialize pool_loader
    pool_dataset.resampleSubsetIndices()
    pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32, shuffle=False,
                                               collate_fn=pool_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    model.eval()  # eval mode disables dropout

    image_indices = list()
    image_margins = list()
  
    print("Acquisition w/ query by committee")

    start = time.time()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties, indices) in enumerate(pool_loader):
            print("batch #: ", i)

            batch_start = time.time()

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)  
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

            image_indices.extend(indices)

            # go through each image and compute image_margin
            for image_num in range(predicted_scores.shape[0]):

                # compute iou mtx
                filtered_locs = model.decode_locations(predicted_locs[image_num]) # get locations in xy form
                iou = find_jaccard_overlap(filtered_locs, filtered_locs)

                class_scores = predicted_scores[image_num]
                class_scores_max_vals, class_scores_max_ind = torch.max(class_scores, dim=1) # sort by class_score for each bbox

                # #compute bbox margins for each class
                # iterate through iou mtx and cluster as needed
                bbox_margins = [[] for _ in range(n_classes)]
                not_visited = torch.ones(8732, dtype=torch.bool).cuda() #all true

                correct_prob = (class_scores_max_vals > 0.1) & (class_scores_max_vals < 0.9)

                for row in range(0, 8732):
                    # skip counting if we can
                    if not_visited[row] == False:
                        continue

                    # bboxes of interest will have iou > 0.3, same cls, not visited
                    correct_iou = (  iou[row, :] > 0.3  )
                    correct_cls = (class_scores_max_ind == class_scores_max_ind[row])
                    correct_bbox = (correct_iou & correct_cls & not_visited & correct_prob).nonzero() #indices with high overlap
                    
                    # rmbr bboxes used
                    not_visited[correct_bbox] = False         

                    softmax_scores = class_scores_max_vals[correct_bbox].squeeze()

                    if (softmax_scores.numel() > 1):
                        values, indices = torch.topk(softmax_scores, 2) 

                        bbox_margin = abs(values[0] - values[1])

                        bbox_class = class_scores_max_ind[row]
                        bbox_margins[bbox_class].append(bbox_margin)
                    
                # get avg. bbox margin for each class
                class_confidences = list()
                class_margins = list()
                for class_num in range(n_classes):
                    if (len(bbox_margins[class_num]) > 0):
                        class_margin = sum(bbox_margins[class_num]) / len(bbox_margins[class_num])
                        class_confidence = max(bbox_margins[class_num])
                        class_margins.append(class_margin)
                        class_confidences.append(class_confidence)

                class_confidences = torch.FloatTensor(class_confidences)
                class_margins = torch.FloatTensor(class_margins)
                image_margin = torch.sum(class_confidences * class_margins ) / torch.sum(class_confidences)

                image_margins.append(image_margin.cpu().item())
    
            batch_time = batch_start - time.time()
            print("batch time:", batch_time)

    #pick images with highest score
    best_ind = np.argsort(image_margins)[-num_of_queries:]

    total_time = time.time() - start
    print("Time required for QBC ", total_time)
    print("Time per image, ", total_time/pool_subset)


    return [image_indices[i] for i in best_ind]

if __name__ == '__main__':
    main()
