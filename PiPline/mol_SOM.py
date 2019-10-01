import os, math,time
from rdkit import Chem, DataStructs
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from tqdm import tqdm, trange
import numpy as np
import torch
import argparse
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from som import SOM
import os, csv, random
from torch.utils.data import Dataset, DataLoader

gpu_n=2
device = torch.device(f'cuda:{gpu_n}' if torch.cuda.is_available() else 'cpu')
#直接从网上下载的 CDK smiles rdkit 可能有未识别的 修正
def save_smiles_property_file(path, smiles, labels, delimiter=','):
    f = open(path, 'w')
    n_targets = labels.shape[1]#if n>1 multitask
    for i in range(len(smiles)):
        f.writelines(smiles[i])
        for j in range(n_targets):
            f.writelines(delimiter + str(labels[i, j]))
        f.writelines('\n')
    f.close()

def make_fingerprint(smiles: str,
                     fingerprint: str='morgan',
                     radius: int = 2,
                     num_bits: int = 2048,
                     use_counts: bool = False) -> np.ndarray:
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if fingerprint=='morgan':
        if use_counts:
            fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
        else:
            fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
        fp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_vect, fp)
    else:
        raise ValueError(f'for now only support morgan fingerprint')
    return fp

def read_smiles_property_file(args,# args will be use in the read_split_scale_write() do not remove
                              path, cols_to_read, #in multitask use multi cols
                              delimiter=',',  # change , the inputfile
                              fingerprint=True,
                              keep_header=False):  # todo merge args path or something
    print(f'path={path}')
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    header = next(reader)
    #cols_to_read=[col for col in range(len(header))] #if use all col, else just selected

    data_full = np.array(list(reader))#should also consider 0
    print(f'data_full.shape{data_full.shape}\n{data_full}--<')

    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        print(i,f'cols_to_read{cols_to_read}--->{header[i]}')
        data[i] = data_full[start_position:, col]
        if i >=1:#fill the miss or NA with value
            data[i]=np.where(data[i]=='None',None,data[i] )
            data[i]=np.where(data[i]=='nan',None,data[i] )
            data[i]=np.where(data[i]=='',None, data[i])#'' the datatype as U109,  astype('float32') will be used next， then None will be nan in np float()
    return data

def worker_init_fn():
        np.random.seed(args.seed)#+worker_id_)#todo 最好由args.seed 控制
class DGLDataset(Dataset):
    def __init__(self, dataf, training=True):
        print(f'Loading data...{dataf}')
        self.data_file = dataf  # use the Path() in all systems mergeinto args
        with open(self.data_file) as f:
            self.data = [line.strip("\r\n") for line in f]
        print('Loading finished.')
        print('\tNum samples:', len(self.data))
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx].split(',')[0]
        lab_val=self.data[idx].split(',')##TypeError: float() argument must be a string or a number, not 'NoneType'
        #avoid miss 'nan'
        label=[None if lab =='nan' or lab ==''or lab=='None' else lab for lab in lab_val][1:]#keep prop col
        label=[float(x) if x is not None else None for x in label]
        #print(f'check label:\n{label}')
        #todo change when in multi task
        result = {
            'label': label,
            'sm': smiles
        }  # keep sm for write the sm predic label file
        return result

def _unpack_field(examples, field):
    return [e[field] for e in examples]


class DGLCollator(object):
    def __init__(self, training):
        self.training = training

    def __call__(self, examples):
        # TODO: either support pickling or get around ctypes pointers using scipy
        # batch molecule graphs #use
        #print(f'examples\n{examples}')
        mol_labels = _unpack_field(examples, 'label')
        mol_sms = _unpack_field(examples, 'sm')
        mol_fps=[make_fingerprint(Chem.MolFromSmiles(str(sm))) for sm in mol_sms]
        result_batch = {
            'sm': mol_sms,  # list str
            'labels': mol_labels,  # list of list of float
            'mol_fp':mol_fps
        }
        # if not self.training:#just encoding in QSAR no generation process
        #print(f'the labels_batch size is \t',len(result['labels']))
        return result_batch


def seed_torch(seed):#如果同一台pc 上同时跑GPU任务，都用此code 也会导致同seed 不同结果，可能是多次设置sed
    random.seed(seed)#分到不同PC上看效果
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU. 可能会重复给多个GPU 设seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False# the results are very close, but some times not match对于GPU必不可少
    #https://discuss.pytorch.org/t/random-seed-initialization/7854/11
    print(f'used seed in seed_torch={seed}<+++++++++++')

def read_split_scale_write( args,  data_path=None,#todo change the dir
                      tmp_data_dir = None, cols_to_read=None):
    if args.data_path !=None:
        data_path=args.data_path
    if args.tmp_data_dir !=None:
        tmp_data_dir=args.tmp_data_dir
    #data_file= Path(os.getcwd())/dataf #use the Path() in all systems
    data = read_smiles_property_file(args,args.data_path,
                                     args.cols_to_read,#1=smile,2=property #todo 得事先知道多少列when multitasks change
                                     keep_header=True)
    #todo when multitasks change
    smiles = data[0]                  #array([ nan,  nan,  nan, ..., 5.81, 2.68, 2.06], dtype=float32)
    print(f'data={len(data)}')
    if len(data) > 1:
        labels = np.array(data[1:], dtype='float')#ar1 ar2 ... one by one then T just one ar ele by ele
        labels = labels.T
        args.n_task=n_task=len(data)-1
    else:
        labels = None
        n_task=None
    try:
        os.stat(tmp_data_dir)
    except:
        os.mkdir(tmp_data_dir)

    cross_validation_split = KFold(n_splits=10, shuffle=True,random_state=args.seed)#todo same random_state, when do para change all
    data = list(cross_validation_split.split(smiles, labels))#merge all labels into one ar= N*n_tasks
    i = 0
    sizes=(0.8,0.1,0.1)
    scalers=[]#the num=fold
    train_steps=[]
    args.class_weights=[]
    for split in data:
        if args.split_type == 'random':
            print('Cross validation with random split, fold number ' + str(i) + ' in progress...')
            train_val, test = split  # 小数位数比总数位数多就会得到8:1:1
            train, val =train_test_split(train_val, test_size=0.11111111111,random_state=args.seed)#shuffle : boolean, optional (default=True

        X_train = smiles[train]
        train_steps.append(len(X_train)//args.batch_size)
        y_train = labels[train]#.reshape(-1)#np array merge into one row todo change to multi
        #print(f'y_train looks {y_train}<<<<')
        X_val=smiles[val]
        y_val=labels[val]
        X_test=smiles[test]
        y_test=labels[test]
        args.train_size=len(X_train)
        args.val_size=len(X_val)
        args.test_size=len(X_test)
        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        scaler = None
        scalers.append(scaler)
        assert n_task != None
        save_smiles_property_file(f'{tmp_data_dir}{args.data_filename}_{i}_train',
                                  X_train, y_train.reshape(-1, n_task))#as comparing with scaled Todo change when use multitask
        if args.dataset_type=='classification':
            if args.class_balance:
                train_labels=y_train.reshape(-1, n_task).tolist()
                #print(f'train_labels{train_labels},len{len(train_labels)},y_train.reshape(-1, n_task){y_train.reshape(-1, n_task).shape}')
                valid_targets = [[] for _ in range(args.n_task)]
                for ij in range(len(train_labels)):
                    for task_num in range(args.n_task):
                        if not math.isnan(train_labels[ij][task_num]):#np.nan can be see as math.isnan
                            valid_targets[task_num].append(train_labels[ij][task_num])
                train_class_sizes = []
                #print(f'valid_targets{valid_targets},valid_targets--len{len(valid_targets)}')
                for task_targets in valid_targets:
                    #print(task_targets)
                    assert set(np.unique(task_targets)) <= {0, 1}# Make sure we're dealing with a binary classification task
                    try:
                        ones = np.count_nonzero(task_targets) / len(task_targets)#pos / total in one task
                    except ZeroDivisionError:
                        ones = float('nan')
                        print('Warning: class has no targets')
                    train_class_sizes.append([1 - ones, ones])#[neg,pos]
                class_batch_counts = torch.Tensor(train_class_sizes) * args.batch_size
                #print(f'class_batch_counts is {class_batch_counts}')#todo 实际每个batch 正负类个数是不定的，这里相当于由总体的百分数X batchSize
                args.class_weights.append(1 / torch.Tensor(class_batch_counts))#just use the train data's information
        #print(args.class_weights,'<<<<args.class_weights')
        #only train need to scaled as this part data used to train model, after train, when evaluate need scaled predictions back then score
        save_smiles_property_file(f'{tmp_data_dir}{args.data_filename}_{i}_test',
                                  X_test, y_test.reshape(-1, n_task))
        save_smiles_property_file(f'{tmp_data_dir}{args.data_filename}_{i}_val',
                                  X_val, y_val.reshape(-1, n_task))
        i+=1
    print(f'train_steps:{train_steps}################################\n')
    return scalers, train_steps

def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def predict_raw(som, X):
    #X = format_data(X)#意维度的变化
    return X.dot(som.weight)#相当于自注意力机制的score:b, map_size todo 点乘 based on dims,self.weight  X 至少 dim 维度一致，W i j, value each X 与 each output node的关系
    #w 是与距离有关   与个X 最像的 node 应该是 predict 最大的位置
def predict(som,X):
    w_=som.weight.data.cpu()
    raw_output = X.cpu().numpy().dot(w_.numpy())# 这里所谓预测就是 把输入与 out里的W 相乘
    output = np.zeros(raw_output.shape, dtype=np.int0)#b, map_size
    max_args = raw_output.argmax(axis=1)
    output[range(raw_output.shape[0]), max_args] = 1
    #print(f'output{output}')
    return output.argmax(axis=1)# 选取 输入X 与W dot后最大的得分的位置 todo  这种预测是个什么算法 ？ 请教时老师

from itertools import product
def iter_neighbours(weights, hexagon=False):
    _, grid_height, grid_width = weights.shape
    hexagon_even_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, 1))
    hexagon_odd_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1))
    rectangle_actions = ((-1, 0), (0, -1), (1, 0), (0, 1))
    for neuron_x, neuron_y in product(range(grid_height), range(grid_width)):
        neighbours = []
        if hexagon and neuron_x % 2 == 1:
            actions = hexagon_even_actions
        elif hexagon:
            actions = hexagon_odd_actions
        else:
            actions = rectangle_actions

        for shift_x, shift_y in actions:
            neigbour_x = neuron_x + shift_x
            neigbour_y = neuron_y + shift_y

            if 0 <= neigbour_x < grid_height and 0 <= neigbour_y < grid_width:
                neighbours.append((neigbour_x, neigbour_y))

        yield (neuron_x, neuron_y), neighbours#当前neruron 以及 上下左右 的邻居 位置

def compute_heatmap(weight,GRID_HEIGHT, GRID_WIDTH,save_f):
    heatmap = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for (neuron_x, neuron_y), neighbours in iter_neighbours(weight):
        total_distance = 0

        for (neigbour_x, neigbour_y) in neighbours:
            neuron_vec = weight[:, neuron_x, neuron_y]#weight  form todo
            neigbour_vec = weight[:, neigbour_x, neigbour_y]

            distance = np.linalg.norm(neuron_vec - neigbour_vec)
            total_distance += distance

        avg_distance = total_distance / len(neighbours)
        heatmap[neuron_x, neuron_y] = avg_distance#每个 grid 与其 nei 的weight_平均距离
    # save heatmap to file then use origin or other w
    #np.save('heatmapPlot.dat', heatmap)#this binary file
    np.savetxt(save_f+'_heatmapPlot.csv', heatmap, delimiter=',')#save as csv file todo pytorch 化 结合 原文pipline
    #print(f'heatmap shape {heatmap.shape}')
    return heatmap

class_parameters = [
    dict(
        marker='o',#o=SPHERE
        markeredgecolor='w',#w white
        markersize=3,
        markeredgewidth=2,
        markerfacecolor='w',#w white
    ),
    dict(
        marker=None,#use none means this class mark not plot on fig
        markeredgecolor='#348ABD',
        markersize=14,
        markeredgewidth=2,
        markerfacecolor='None',
    ),
]
def plotHeatmapSOM(sofm,n_inputs, GRID_HEIGHT, GRID_WIDTH,target, clusters,
                   save_f,
                   fig_title=f'Embedded molecule fingprint using SOM for ValdationDataset' ):
    plt.figure(figsize=(13, 13))
    plt.title(fig_title)
    #todo 先计算heatmap 然后根据heatmap 里的值来选择大于多少才标记mark
    weight = sofm.weight.cpu().numpy().reshape((n_inputs, GRID_HEIGHT, GRID_WIDTH))
    heatmap = compute_heatmap(weight,GRID_HEIGHT, GRID_WIDTH,save_f)
    max_val=np.amax(heatmap)## Maximum of the flattened array  https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
    #heatmap = max_val - heatmap#todo keep 与NB 文章一样的效果

    for actual_class, cluster_index in zip(target, clusters): #todo clusters元素类由map_size决定，所以不一定和target 类数一致，但是 b个数一致
        cluster_x, cluster_y = divmod(cluster_index, GRID_HEIGHT)#divmod(10, 3)-->(3, 1)  得到map 坐标
        if heatmap[cluster_x, cluster_y] >= 0.6:#todo  设置个阈值方便 重复 paper的图
            parameters = class_parameters[int(actual_class)]#actual_class就定义了 2类  S, C
        else:
            parameters = class_parameters[1]#第2类 不标记 用None
        if args.expanded_heatmap:
            plt.plot(2 * cluster_x, 2 * cluster_y, **parameters)
        else:
            plt.plot(cluster_x, cluster_y, **parameters)#todo when want to plot marks 白点
            continue
    plt.imshow(heatmap, cmap='jet', interpolation='nearest')#color choose https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
    plt.grid(b=None)
    plt.colorbar()
    #plt.show()
    plt.savefig(save_f)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--color', dest='dataset', action='store_const',
                        const='color', default=None,
                        help='use color')
    parser.add_argument('--mnist', dest='dataset', action='store_const',
                        const='mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--fashion_mnist', dest='dataset', action='store_const',
                        const='fashion_mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--train', action='store_const',
                        const=True, default=False,
                        help='train network')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.3, help='input learning rate')
    parser.add_argument('--epoch', type=int, default=100
                        , help='input total epoch')
    parser.add_argument('--data_dir', type=str, default='datasets', help='set a data directory')
    parser.add_argument('--res_dir', type=str, default='results', help='set a result directory')
    parser.add_argument('--model_dir', type=str, default='model', help='set a model directory')
    parser.add_argument('--row', type=int, default=20, help='set SOM row length')
    parser.add_argument('--col', type=int, default=20, help='set SOM col length')
    args = parser.parse_args()

    # Hyper parameters change args here#################
    args.data_dir='datasets'
    args.dataset='mnist'
    args.train=True
    args.save_dir='save'
    save_f_val=args.save_dir+'/'+'val_heatmap.png'
    args.class_balance,args.dataset_type = True, 'classification'
    args.data_path, args.tmp_data_dir = 'SOM_fp/MergeLable_2col.csv','SOM_fp/tmp/'
    args.cols_to_read = [0,1]
    args.seed=0
    seed_torch(args.seed)
    args.fold=1#use 10 cross-vaildation, but can just try 1
    args.data_filename=os.path.basename(args.data_path)+f'_seed{args.seed}'
    print(f'{args.data_filename}########')
    args.split_type='random'
    ####################################################################################################################
    args.map=(50,50)
    GRID_HEIGHT, GRID_WIDTH = args.map[0], args.map[1]
    args.input_dim=2048
    args.expanded_heatmap=False
    print(args.data_dir,args.dataset,args.res_dir,args.model_dir )
    # Create results dir

    dataset = args.dataset
    batch_size = args.batch_size
    total_epoch = args.epoch
    row = args.row
    col = args.col
    train = args.train

    scalers, models_stepsEachepoch = read_split_scale_write(args, args.data_path, args.tmp_data_dir,
                                                            cols_to_read=args.cols_to_read)  # use in ensamble_train
    fold=args.fold
    for f_i in range(fold):
        args.Foldth=f_i
        fold_path=os.path.join(args.save_dir, f'fold_{f_i}')
        makedirs(fold_path)
        DGLval = DGLDataset(f'{args.tmp_data_dir}{args.data_filename}_{f_i}_val', training=False)
        DGLtest = DGLDataset(f'{args.tmp_data_dir}{args.data_filename}_{f_i}_test', training=False)
        if args.dataset_type == 'classification':
            DGLtrain = DGLDataset(f'{args.tmp_data_dir}{args.data_filename}_{f_i}_train', training=True)
        else:
            DGLtrain = DGLDataset(f'{args.tmp_data_dir}{args.data_filename}_{f_i}_trainScaled', training=True)
        # clean data as dataloader
        train_dataloader = DataLoader(DGLtrain, batch_size=args.batch_size,  # set size
                                      shuffle=True, num_workers=0,
                                      collate_fn=DGLCollator(training=True),  # make batch
                                      drop_last=False, worker_init_fn=worker_init_fn)
        val_dataloader = DataLoader(DGLval, batch_size=args.batch_size,  # set size
                                    shuffle=True, num_workers=0,
                                    collate_fn=DGLCollator(training=False),  # make batch
                                    drop_last=False,worker_init_fn=worker_init_fn)
        val=[]
        val_lab=[]
        val_lab_list=[]
        for b in val_dataloader:
            val.extend(b['mol_fp'])
            #val_lab.extend(b['labels'][0])
            val_lab_list.extend(b['labels'])
        for b in val_lab_list:
            val_lab.extend(b)
        #print(f'len(val_lab)={len(val_lab)},val_lab_list={len(val_lab_list)},{val_lab}')
        #with open('test.txt','a+') as wf:
        #    wf.write(str(val_lab))
        #    wf.write(str(val_lab_list))

        test_dataloader = DataLoader(DGLtest, batch_size=args.batch_size,  # set size
                                     shuffle=False, num_workers=0,  # in test shuffle=F 同SMILES TESTDataloader
                                     collate_fn=DGLCollator(training=False),  # make batch
                                     drop_last=False,# 不同fold测试数据不完全一样了，没关系但是不用model还是用了同样数据
                                     worker_init_fn=worker_init_fn)#if worker_init_fn=None, it use the torch.initial seed()
        test=[]
        test_lab=[]
        test_lab_list=[]
        for b in test_dataloader:
            test.extend(b['mol_fp'])
            test_lab_list.extend(b['labels'])#todo considring multi_lab in feature
        for b in test_lab_list:
            test_lab.extend(b)
        #print(f'len(test)={len(test)},test[0]={len(test[1])},{test[1]}')
        #print(f'len(test_lab)={len(test_lab)},test[0]={len(test_lab[1])},{test_lab[1]}')

        #todo add train func,predict func,合并 nepy  sofm  and here
        input_dim=args.input_dim
        som = SOM(input_size=input_dim, out_size=args.map)
        som = som.to(device)
        if train == True:
            losses = list()
            for epoch in tqdm(range(total_epoch)):
                running_loss = 0
                start_time = time.time()
                for idx, b in enumerate(train_dataloader):
                    X=torch.FloatTensor(b['mol_fp'])
                    #print(f'X.size{X.size()}')
                    X = X.view(-1, input_dim).to(device)    # flatten
                    loss = som.self_organizing(X, epoch, total_epoch)    # train som
                    running_loss += loss
                losses.append(running_loss)
                print('epoch = %d, loss = %.2f, time = %.2fs' % (epoch + 1, running_loss, time.time() - start_time))
                if epoch % 100 == 0:
                    # save TODO som.save_result for molecules save
                    torch.save(som.state_dict(), f'{args.save_dir}/som_F{f_i}Epoch{epoch}.pth')
            torch.save(som.state_dict(), f'{args.save_dir}/som_F{f_i}.pth')
            #plot loss trend for each fold training
            plt.title('SOM loss')
            plt.plot(losses)
            #plt.show()
            plt.savefig(f'{args.save_dir}/train_loss{f_i}.png')
            plt.clf()#https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
            plt.close()

            save_f_val = args.save_dir + '/' + f'val_heatmapF{f_i}.png'
            save_f_test = args.save_dir + '/' + f'test_heatmapF{f_i}.png'
            # testing   and pliot heatmap each fold
            x_val=torch.FloatTensor(val)
            x_val = x_val.view(-1, input_dim).to(device)
            val_clusters = predict(som,x_val)
            plotHeatmapSOM(som,input_dim, GRID_HEIGHT, GRID_WIDTH,val_lab, val_clusters,save_f_val,
                           fig_title=f'Embedded molecule fingprint using SOM for ValdationDataset')

            x_test=torch.FloatTensor(test)
            x_test = x_test.view(-1, input_dim).to(device)
            test_clusters = predict(som,x_test)
            plotHeatmapSOM(som,input_dim, GRID_HEIGHT, GRID_WIDTH,test_lab, test_clusters,save_f_test,
                           fig_title=f'Embedded molecule fingprint using SOM for testDatasets')
            #val=[b for b in train_dataloader]
            #print(f'len(val)={len(val)}')
            #   X=torch.FloatTensor(b['mol_fp'])
            #    print(f'X.size{X.size()}')
            #predict(som,)


#todo  复制到linux 下 末尾不能有多空行,不然  data_full = np.array(list(reader)) 只有1维， 因为末尾的空行没有 ',' 来分割